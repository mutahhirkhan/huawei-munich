# Pinna-85 Landmark Extraction — Developer Implementation Guide

**Target hardware: 1× NVIDIA Tesla T4 (16 GB), free Google Colab.**
Everything below is scoped so it fits that budget. Anything that does not fit is listed in §2 with the reason it was cut.

---

## 0. What we are building, in one paragraph

We never feed the 365 k-vertex head mesh into a network. We crop each ear, rotate it into a canonical frame, mirror every left ear so the model only ever sees "right ears", and then **render the ear geometry from 5 virtual cameras** as image-like buffers (depth, normals, curvature). A standard 2D heatmap CNN (HRNet) predicts 85 keypoints in each rendered view. Because we control the cameras exactly, each 2D detection is a known line in 3D space — so **5 views triangulate back to 85 3D points** by plain least squares. A PCA shape prior fixes outliers, a small point-cloud network refines each landmark locally where the views were occluded, and finally every point is snapped onto the mesh surface.

```
                    ┌────────────────────────────────────────────┐
   head.ply         │  S1  ingest + sanity checks                │
   (365k verts)  ─► │  S2  ear ROI crop → canonicalize → mirror  │  CPU, one-time cache
                    └────────────────────────────────────────────┘
                                       │  ear mesh (~30k faces), right-ear frame
                                       ▼
                    ┌────────────────────────────────────────────┐
                    │  S3  multi-view raycast render (5 views)   │  CPU, in dataloader
                    │      6 channels: depth1, depth2, nx,ny,nz, │
                    │      shape-index                           │
                    └────────────────────────────────────────────┘
                                       │  (5, 6, 256, 256)
                                       ▼
                    ┌────────────────────────────────────────────┐
                    │  S5  HRNet-W18 → 85 heatmaps per view      │  GPU (this is the only
                    │      + soft-argmax → (u,v) + confidence    │   heavy training step)
                    └────────────────────────────────────────────┘
                                       │  (5, 85, 2) + (5, 85) conf
                                       ▼
                    ┌────────────────────────────────────────────┐
                    │  S7  weighted least-squares triangulation  │  closed form, 1 ms
                    └────────────────────────────────────────────┘
                                       │  (85, 3) coarse
                                       ▼
                    ┌────────────────────────────────────────────┐
                    │  S8  PCA shape prior (MAP fit + blend)     │  closed form
                    │  S9  local patch refinement net (Δxyz)     │  GPU, small
                    │  S10 project onto mesh surface             │  Open3D closest-point
                    └────────────────────────────────────────────┘
                                       │
                                       ▼
                        85 × 3 per ear, in original head coords
```

**Expected error band:** mean-shape baseline ≈ 3.5–5 mm → Stage-1 only ≈ 2.0–2.5 mm → + PCA + refinement ≈ **1.5–2.0 mm MD**. Annotation noise is the floor; do not expect < 1 mm.

---

## 1. Why this architecture and not the other two proposals

| Proposal in the deck | Verdict on a free T4 |
|---|---|
| **MPAN** (InfoGNN + PTv3 dual-stream, cross-attention fusion) | **Cut.** Two backbones on 365 k-vertex graphs. PTv3 wants Flash-Attention (Ampere+ / sm_80); the T4 is sm_75 and falls back to vanilla attention — 3–5× slower. GNN message passing over a mesh graph of this size will not fit 16 GB without aggressive decimation, which defeats the point of the mesh branch. Also needs `spconv` + `torch-scatter` + `pointops` compiled — a day of Colab dependency pain each session. |
| **Mutahhir's PTv3 + Bayesian CPD** | **Partly kept.** The heatmap-instead-of-regression insight and the shape-prior refinement are correct and we keep both. The PTv3 backbone and BCPD are dropped: BCPD on 85 points is overkill versus a closed-form weighted PCA fit that runs in microseconds and is much easier to debug. |
| **Rizwan's multi-view HRNet + PCA** | **This is the backbone of our plan.** Mature 2D machinery, pretrained ImageNet weights (matters enormously with only 400 ears), small memory footprint, and it maps cleanly onto Colab. |

**What we add on top of Rizwan's version** (the three things that will decide whether we place):

1. **Depth peeling (2 depth layers per view).** The inner helix and concha sit *under* the helix rim. A single z-buffer never sees them, which is exactly why a pure multi-view method loses on 50 of the 85 landmarks. Casting a second ray past the first hit exposes the hidden surface for free.
2. **In-graph differentiable triangulation** so we can put a loss directly on the 3D error — the actual competition metric — instead of only on 2D pixels.
3. **A point-cloud patch refiner** as a second stage, because rendered views will always be weakest in the concha cavity.

---

## 2. Facts about the data you must verify on day 1

Write `notebooks/00_data_audit.ipynb` and confirm each of these before anyone writes model code. These numbers drive every hyperparameter downstream.

| # | Check | Why it matters | Expected |
|---|---|---|---|
| 1 | Units are millimetres | Everything below assumes mm | Ear height ≈ 60–65 mm |
| 2 | Sign of Y for left vs right ear | Mirroring code depends on it | Sample shows left-ear Y ≈ +64…+91 |
| 3 | Std-dev of the 85-landmark centroid across 200 subjects | Decides how big the initial crop box must be | Hope for < 8 mm |
| 4 | Distance from each GT landmark to the **mesh surface** (not nearest vertex) | If ≈ 0, S10 surface projection is free accuracy. If not, S10 will *hurt* | Nearest-*vertex* distance is 0.247 mm mean, so surface distance should be ≈ 0.05 mm |
| 5 | Are landmark indices consistently ordered along each contour, same start point every subject? | If ordering flips on some subjects, every model will be poisoned | Plot contour 0→24 as a polyline for 20 random ears and eyeball it |
| 6 | Per-landmark std-dev after centring | Tells you which landmarks are intrinsically noisy | Concha will be worst |
| 7 | Duplicate / degenerate faces, non-manifold edges | Raycasting needs a clean mesh | Run `mesh.remove_degenerate_triangles()` etc. |

> **If check 5 fails on any subject, flag and quarantine it.** Silent annotation-order flips are the single most common reason a landmark model plateaus.

Also compute **Baseline B0** on day 1: predict the training mean shape, translated to the GT ear centroid. Report its MD. Every later number is judged against it. If a model does not beat B0 by 2×, something is wrong in the geometry code, not the network.

---

## 3. Stage-by-stage spec

Each stage lists its **input → output contract** with exact tensor shapes. Build them as independent modules with their own unit test; do not chain anything until the test passes.

### S1 — Ingest and caching

```python
import open3d as o3d, numpy as np, pandas as pd

def load_head(ply_path):
    m = o3d.io.read_triangle_mesh(str(ply_path))          # ~1–2 s for 365k verts
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m

def load_landmarks(csv_path):
    # CSV rows look like:  0 [-0.75018139 74.57184769 31.7642007 ]
    rows = []
    for line in open(csv_path):
        nums = line.replace('[',' ').replace(']',' ').split()
        rows.append([float(v) for v in nums[-3:]])
    L = np.asarray(rows, dtype=np.float64)
    assert L.shape == (85, 3), f"{csv_path}: got {L.shape}"
    return L
```

**Contour index map** — hardcode once, import everywhere:

```python
CONTOURS = {
    "outer_helix":        slice(0, 25),    # 25
    "concha_outline":     slice(25, 55),   # 30
    "inner_helix":        slice(55, 75),   # 20
    "superior_antihelix": slice(75, 85),   # 10
}
```

> Verify this against Figure 3 in the problem statement before trusting it — the figure numbers outer helix 0–24, concha 25–54, inner helix 55–74, antihelix 75–84, which matches. Confirm on real data anyway.

---

### S2 — Ear crop, canonicalization, mirroring

**In:** head mesh + (train only) 85 GT landmarks.
**Out:** `ear_mesh` (~30 k faces) in right-ear canonical frame, `T` (4×4 transform back to head coords), `L_canon` (85, 3).

Three sub-steps:

1. **Coarse localisation.** Because meshes are already interaurally aligned, start with a fixed anatomical box derived from the training statistics from check 3:
   `centre = mean_landmark_centroid ± 3σ`, box size **90 × 50 × 90 mm**. At inference this is your pass-1 crop. Pass 2 re-crops tightly (70 mm cube) around the pass-1 prediction centroid — see S11.
2. **Canonical frame.** Translate so the ear centroid is at the origin. Keep the axis directions from the interaural frame (already consistent) — do **not** run PCA on the ear itself to define axes; PCA axes flip sign between subjects and will silently scramble your dataset. **Do not rescale**: the metric is in millimetres, so stay in millimetres.
3. **Mirror all left ears → right ears.**

```python
def mirror_to_right(vertices, faces, landmarks):
    V = vertices.copy(); V[:, 1] *= -1                 # negate the interaural axis
    F = faces[:, ::-1].copy()                          # flip winding so normals stay outward
    L = landmarks.copy(); L[:, 1] *= -1
    return V, F, L
```

Landmark **indices do not change** under mirroring — index *i* is the same anatomical point on both ears.

4. **Decimate** to ~30 k faces (`mesh.simplify_quadric_decimation(30000)`). Measure the GT-to-decimated-surface distance; it must stay under 0.05 mm. Cache as `.npz` (V float32, F int32, L float32, T float32) → ~400 files, ~500 MB total.

**Unit test:** un-mirror and un-transform a cached ear; the vertices must match the original head mesh subset to < 1e-4 mm.

---

### S3 — Multi-view geometry rendering

This is the module that decides your accuracy ceiling, so get it exactly right.

**Cameras.** Orthographic, 5 views. Define view *k* by a rotation `R_k` (world→camera). Base view looks straight down the lateral axis at the ear; the other four are yaw/pitch offsets:

```python
VIEWS = [(  0,   0),      # lateral
         (+30,   0),      # rotated toward the face
         (-30,   0),      # rotated toward the back of head
         (  0, +25),      # from above
         (  0, -25)]      # from below
```

**Resolution and scale.** 256 × 256 px at **0.28 mm/px** → 71.7 mm field of view, comfortably covering a 65 mm ear with margin for jitter. Pixel quantisation error is 0.14 mm — an order of magnitude below the target, so this is not your bottleneck.

**Rendering by ray casting** (Open3D + Embree; no OpenGL, no PyTorch3D, nothing to compile):

```python
import open3d as o3d, numpy as np

def make_scene(V, F):
    m = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V, o3d.core.float32),
                                    o3d.core.Tensor(F, o3d.core.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(m)
    return scene

def render_view(scene, R, res=256, mm_per_px=0.28, far=200.0):
    """Returns depth1, depth2, normals(3), hit mask — all in camera space."""
    lin = (np.arange(res) - (res - 1) / 2) * mm_per_px
    gx, gy = np.meshgrid(lin, lin, indexing='xy')
    # camera-space rays travel along +z_cam; start well outside the mesh
    origins_cam = np.stack([gx, gy, np.full_like(gx, -far)], -1).reshape(-1, 3)
    dirs_cam    = np.tile(np.array([0, 0, 1.0], np.float32), (res * res, 1))

    Rt = R.T                                        # camera→world
    rays = np.concatenate([origins_cam @ Rt.T, dirs_cam @ Rt.T], -1).astype(np.float32)
    ans  = scene.cast_rays(o3d.core.Tensor(rays))

    t1   = ans['t_hit'].numpy()                      # distance along ray
    hit  = np.isfinite(t1)
    nrm  = ans['primitive_normals'].numpy() @ R.T    # rotate normals into camera space

    # ---- depth peeling: second layer, revealing surfaces under the helix rim ----
    eps = 0.5
    origins2 = rays[:, :3] + rays[:, 3:] * (np.nan_to_num(t1, nan=far)[:, None] + eps)
    rays2 = np.concatenate([origins2, rays[:, 3:]], -1).astype(np.float32)
    t2 = scene.cast_rays(o3d.core.Tensor(rays2))['t_hit'].numpy()

    d1 = np.where(hit, t1 - far, 0.0).reshape(res, res)
    d2 = np.where(np.isfinite(t2), t1 + eps + t2 - far, 0.0).reshape(res, res)
    return d1, d2, nrm.reshape(res, res, 3), hit.reshape(res, res)
```

**Channels fed to the CNN (6):** `depth1`, `depth2 − depth1` (thickness under the overhang), `nx, ny, nz`, `shape_index`.
Shape index (Koenderink) from principal curvatures is more stable than raw mean curvature; compute it once per vertex on the cropped mesh and sample it at the ray hit point via barycentric interpolation of `primitive_ids` + `primitive_uvs`.

Normalise depth per-sample: `d = (d − d[hit].mean()) / 15.0`, zeros where `~hit`. Add the hit mask as a 7th channel if you like; it is cheap.

**Cost:** 5 views × 65 k rays × 2 layers ≈ 650 k rays ≈ 25–40 ms per ear on one Colab core. With `num_workers=2` this keeps pace with the GPU. Cache the `RaycastingScene` objects in an LRU dict (400 ears × 30 k faces ≈ 400 MB RAM — fine in 12.7 GB).

**Unit test (do not skip this one):** project GT landmarks into all 5 views (§S7 formulas), then triangulate them straight back. Round-trip error must be **< 0.05 mm**. If it isn't, your `R_k` convention is wrong and no amount of training will save you.

---

### S4 — Ground-truth targets

For view *k* with rotation `R_k` and mm/px `s`, landmark `X` (canonical mm):

```
p     = R_k @ X                    # camera space
u     = p[0]/s + (res-1)/2         # pixel column
v     = p[1]/s + (res-1)/2         # pixel row
depth = p[2]
```

Then:

* **Heatmap target:** 2D Gaussian at `(u,v)`, σ = 2.0 px, rendered at heatmap resolution (128 × 128 = stride 2 → 0.56 mm/px; use stride 2, not the HRNet default stride 4).
* **Visibility mask:** landmark is visible in view *k* if `depth − depth_rendered(u,v) < 1.0 mm` (first layer) **or** it falls between layer 1 and layer 2. Otherwise mask it out of the 2D loss. Typically 3–4 of the 5 views see any given landmark; concha points may only get 1–2, which is precisely why S9 exists.
* **Out-of-frame:** if `(u,v)` leaves the image, mask it out too.

---

### S5 — Stage-1 network

```
backbone : timm HRNet-W18 (pretrained=True)   ~9.6M params
input    : (B, 7, 256, 256)     # inflate the 3-channel stem to 7 by weight-tiling /7×3
head     : 1×1 conv → 85 channels at 128×128  (heatmaps)
         : 1×1 conv → 85 channels at 128×128  (depth map, optional/aux)
```

Use **HRNet-W18, not W32.** You have 400 ears. W32 (28 M params) overfits and doubles the step time for < 0.1 mm gain in our regime. If W18 saturates and val error is still falling with more augmentation, then try W32.

Alternative if `timm` HRNet gives trouble: a U-Net with a ResNet-34 encoder from `segmentation_models_pytorch` is within ~0.1 mm and installs cleanly.

**Coordinate extraction (differentiable):**

```python
def soft_argmax_2d(logits, beta=1.0):
    B, K, H, W = logits.shape
    p = torch.softmax(logits.flatten(2) * beta, -1).view(B, K, H, W)
    xs = torch.arange(W, device=logits.device, dtype=p.dtype)
    ys = torch.arange(H, device=logits.device, dtype=p.dtype)
    u = (p.sum(2) * xs).sum(-1)
    v = (p.sum(3) * ys).sum(-1)
    conf = p.flatten(2).max(-1).values          # peak sharpness → triangulation weight
    return torch.stack([u, v], -1), conf        # (B,K,2), (B,K)
```

---

### S6 — Losses

```
L = 1.0 * L_coord2d        # SmoothL1(soft-argmax uv, gt uv), masked by visibility
  + 1.0 * L_heat           # JS-divergence between predicted heatmap and Gaussian target
  + 0.1 * L_depth          # SmoothL1 on the aux depth head, masked
  + 2.0 * L_3d             # SmoothL1(triangulate(uv), gt_xyz)   ← enable from epoch 150
  + 0.05 * L_smooth        # Σ ||L[i-1] - 2L[i] + L[i+1]|| within each contour
  + 0.02 * L_pca           # ||x - Π_pca(x)||, keeps predictions in the plausible shape space
```

`L_coord2d + L_heat` is the DSNT/JS recipe (Nibali et al., 2018) — soft-argmax alone drifts, the JS term keeps the heatmap unimodal.

`L_3d` is the one that directly optimises the competition metric. It requires all 5 views of one ear in the same batch, which costs memory — so **train in two phases** (see §4).

---

### S7 — Multi-view fusion (closed form, differentiable)

Orthographic projection is linear, so each view gives two exact linear constraints on the 3D point. For view *k* with rows `r1, r2, r3` of `R_k`:

```
r1 · X = s·(u_k − c)
r2 · X = s·(v_k − c)
```

Stack over K views → a 2K × 3 system, solve by weighted least squares with per-view weights `w_k` = confidence × visibility:

```python
def triangulate(uv, conf, Rs, mm_per_px, res):
    """uv: (K,85,2)  conf: (K,85)  Rs: (K,3,3) -> (85,3)"""
    c = (res - 1) / 2
    A, b, w = [], [], []
    for k, R in enumerate(Rs):
        A.append(R[None, :2, :].expand(85, 2, 3))                 # (85,2,3)
        b.append(mm_per_px * (uv[k] - c))                         # (85,2)
        w.append(conf[k][:, None].expand(85, 2))
    A = torch.cat(A, 1); b = torch.cat(b, 1); w = torch.cat(w, 1) # (85,2K,3),(85,2K)
    Aw = A * w[..., None]; bw = b * w
    return torch.linalg.lstsq(Aw, bw[..., None]).solution[..., 0] # (85,3)
```

Keep the **residual** of this solve — it is your best per-landmark uncertainty estimate, better than the heatmap peak, and it feeds S8.

The aux depth head can be added as a third weighted row per view (`r3 · X = depth_k`) but weight it at ~0.2; depth regression is noisier than the 2D position.

---

### S8 — PCA shape prior (MAP fit)

Build once from the training fold: stack the mirrored, ear-centred `85×3 → 255` vectors, run PCA, keep components covering 98 % variance (expect **K ≈ 25–35**).

MAP estimate under per-landmark noise σᵢ (from S7 residuals) and Gaussian prior with eigenvalues λ:

```
b = (Uᵀ W U + Λ⁻¹)⁻¹ Uᵀ W (ŷ − μ)          W = diag(1/σᵢ²) repeated ×3
x_pca = μ + U b
```

Then **blend per landmark instead of replacing outright**:

```
αᵢ = σᵢ² / (σᵢ² + σ₀²)          σ₀ ≈ 1.0 mm
xᵢ = αᵢ · x_pcaᵢ + (1 − αᵢ) · ŷᵢ
```

Confident landmarks keep their measured position; uncertain ones (concha, occluded inner helix) get pulled to the anatomically plausible shape. Tune σ₀ on validation only — this single scalar is worth ~0.2 mm.

> Also fit a **joint left+right PCA** later. The two ears of one subject are strongly correlated; a joint model regularises the harder ear using the easier one. Cheap experiment, real gain.

---

### S9 — Local patch refinement

The views cannot see into the concha floor. This stage does.

For each landmark *i*:

* Gather the **256 nearest mesh vertices within 8 mm** of the coarse prediction.
* Centre them on the prediction, divide by 8 mm.
* Feed to a shared PointNet-style encoder: `MLP(6→64→128→256)` per point → **conditioned heatmap head** producing one logit per point, conditioned on a learned 85-way landmark embedding (concatenated to every point feature).
* Softmax over the 256 points → soft-argmax → refined position; multiply by 8 mm and add back the centre.

Predicting a distribution over patch points beats regressing Δxyz directly (CHaRM, arXiv:2501.13073) and gives you a second, independent confidence signal.

Batch shape at training: `(B=4 ears × 85 landmarks, 256 points, 6 features)` = 340 patches — trivial on a T4, trains in under an hour. Train it *after* Stage 1 is frozen, on Stage-1's own validation-fold outputs, so it learns to fix realistic errors rather than synthetic ones.

---

### S10 — Surface projection

```python
scene = make_scene(V, F)
q = o3d.core.Tensor(pred.astype(np.float32))
proj = scene.compute_closest_points(q)['points'].numpy()
```

Then transform back to original head coordinates with `T⁻¹` (and un-mirror if it was a left ear).

**Gate this on validation.** Apply it only if it reduces val MD. If data-audit check 4 shows GT landmarks sit slightly *off* the surface, projection will cost you accuracy — in that case, project only landmarks whose distance-to-surface exceeds the training mean by 2σ.

---

### S11 — Inference, end to end

```
head.ply
  → fixed anatomical crop box (both ears)          [S2]
  → render 5 views                                  [S3]
  → HRNet pass 1 → triangulate                      [S5,S7]
  → RE-CROP tightly around the pass-1 centroid      ← cascade: removes crop-position error
  → render 5 views again → HRNet pass 2 → triangulate
  → PCA MAP blend                                   [S8]
  → patch refinement                                [S9]
  → surface projection + un-mirror + un-transform   [S10]
  → 85×3 left, 85×3 right
```

Two passes double inference cost (~2 s per head, irrelevant) and remove your single largest failure mode. Train Stage 1 with **±8 mm translation and ±10° rotation jitter** so pass 1 tolerates a sloppy initial crop.

---

### S12 — Evaluation harness

Implement the official metric exactly and report more than the headline number:

* **MD** (mean over subjects and both ears) — the official number.
* **Per-contour MD** — outer helix will be best, concha worst. This tells you where to spend effort.
* **Per-landmark MD** heat-strip (85 bars) — exposes individual bad annotations.
* **Median and 90th percentile** — the mean hides catastrophic single ears.
* **Worst-10 ears rendered as images** — look at them every week. Most of your remaining error will be two or three broken ears, not a uniform blur.
* Left vs right MD separately, to catch mirroring bugs.

---

## 4. Training recipe on a T4

Free Colab gives ~12 h max, idle-disconnects at ~90 min, and can preempt at any moment. Design around it: **short epochs, checkpoint every epoch to Drive, resume automatically.**

**Phase A — per-view training (bulk of the time).**
Treat `(ear, view)` as the sample. 400 ears × 5 views = 2 000 images/epoch.

| Setting | Value |
|---|---|
| Batch | 16 images |
| Precision | AMP **fp16** + `GradScaler` (T4 is Turing — **no bf16**) |
| Optimiser | AdamW, lr 3e-4, wd 1e-4 |
| Schedule | 5-epoch warmup → cosine to 1e-6 |
| Epochs | 300 |
| Step time | ~0.35 s → **~45 s/epoch** → **≈ 4 h total** |
| VRAM | ~6 GB |

**Phase B — 3D-consistency fine-tune.**
Switch the sample unit to a whole ear (5 views together) so `L_3d` can be computed in-graph.

| Setting | Value |
|---|---|
| Batch | 2 ears = 10 images (grad-accum ×4 for an effective 8) |
| lr | 5e-5, cosine, 60 epochs |
| Time | ~1.5 h |
| VRAM | ~9 GB |

**Phase C — patch refiner.** 40 epochs, ~40 min.

**Total per fold ≈ 6 h.** One fold fits in a single good Colab session. A 5-fold ensemble is ~30 h spread across several days — do folds 0 and 1 first, ensemble those, and only add folds if time allows.

**Augmentation.** Key trick: for an orthographic camera, a **2D in-plane affine transform of the render is exactly equivalent to rolling/shifting/zooming the camera**. So rotation (±180°), translation (±20 px) and scale (0.9–1.1×) applied on-GPU to the rendered tensor are *geometrically exact* 3D augmentations, and free. Out-of-plane variation comes from jittering the camera yaw/pitch by ±8° at render time in the dataloader. Add depth-channel noise (σ = 0.1 mm) and random channel dropout to survive scanner variation.

**Colab survival kit:**

```python
# every epoch
torch.save({'ep': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
            'scaler': scaler.state_dict(), 'best': best_md},
           '/content/drive/MyDrive/pinna/ckpt/fold0_last.pt')
```
Resume by loading `fold0_last.pt` at the top of the notebook, unconditionally. Keep raw `.ply` files on Drive; copy only the ~500 MB cropped-ear cache to `/content` at session start (a `.tar` copy takes ~1 min, versus 20 min to re-derive it).

---

## 5. Resources needed

### 5.1 Compute

| | Free Colab T4 | Notes |
|---|---|---|
| GPU | Tesla T4, 16 GB, sm_75 | fp16 yes, bf16 no, Flash-Attention no |
| RAM | 12.7 GB | enough; do not load all meshes at once |
| vCPU | 2 | **the real bottleneck** — `num_workers=2`, `persistent_workers=True` |
| Disk | ~78–107 GB ephemeral at `/content` | wiped every session |
| Session | ~12 h, idle timeout ~90 min | checkpoint every epoch |

**Free upgrade path worth taking: Kaggle Notebooks.** 30 GPU-hours/week, P100 or **2× T4**, 12 h sessions, 4 vCPU, and — crucially — **20 GB of persistent Dataset storage**, so you upload the meshes once instead of re-syncing Drive. Run long training there, iterate interactively on Colab.

If you eventually get an A100/L4 (Colab Pro, ~$10), the only thing that changes is that you can raise batch size and try HRNet-W32. The architecture does not need to change — that's deliberate.

### 5.2 Storage budget

| Item | Size |
|---|---|
| Raw dataset: 200 × ~15 MB `.ply` | **~3.1 GB** |
| Landmark CSVs: 400 files | ~2 MB |
| Cropped + decimated ear cache (400 `.npz`) | ~500 MB |
| Checkpoints (HRNet-W18 + optimiser, ×2 kept/fold) | ~250 MB/fold |
| Logs, predictions, figures | ~200 MB |
| **Total on a free 15 GB Drive** | **~5 GB — fits** |

Do **not** pre-render and cache the view images. 400 ears × 12 pose-sets × 5 views at 256²×6 ch would be 15–19 GB. On-the-fly ray casting is both smaller and more diverse.

### 5.3 Software

```bash
# Colab already ships torch 2.x + cu12x; do not reinstall it.
pip -q install open3d==0.18.0 trimesh timm==1.0.* einops \
               scikit-learn scipy pandas tqdm tensorboard
```

| Package | Used for |
|---|---|
| `open3d` | PLY I/O, decimation, **RaycastingScene** (Embree), closest-point projection |
| `trimesh` | curvature helpers, quick sanity plots |
| `timm` | HRNet-W18 pretrained weights |
| `torch` (Colab preinstalled) | everything trainable |
| `scikit-learn` | PCA shape model |
| `tensorboard` | loss + val-MD curves |

Explicitly **not** used, and this is a feature: PyTorch3D, nvdiffrast, spconv, torch-scatter, pointops, flash-attn. Every one of those is a compile-from-source risk that costs an hour at the start of each Colab session.

### 5.4 Repository layout

```
pinna85/
├── configs/            base.yaml, fold0.yaml
├── data/
│   ├── audit.py        the day-1 checks from §2
│   ├── crop.py         S2
│   ├── render.py       S3  (+ its round-trip unit test)
│   └── dataset.py      torch Dataset, on-the-fly render, GPU augment
├── models/
│   ├── hrnet_head.py   S5
│   ├── refiner.py      S9
│   └── losses.py       S6
├── geom/
│   ├── project.py      S4 / S7 projection + triangulation
│   ├── pca_prior.py    S8
│   └── snap.py         S10
├── train_stage1.py     phases A + B
├── train_refiner.py    phase C
├── infer.py            S11
├── evaluate.py         S12
└── notebooks/          00_audit, 01_render_debug, 02_results
```

### 5.5 People and time (3 developers, 4 weeks)

| Week | Dev A — geometry | Dev B — model | Dev C — priors & eval |
|---|---|---|---|
| 1 | §2 audit, S1, S2, cache built | env, dataloader, HRNet skeleton overfits 1 ear | metric harness, **B0 mean-shape baseline**, PCA model |
| 2 | S3 renderer + round-trip test **< 0.05 mm** | Phase A training runs, augmentation | S7 triangulation + confidence, per-contour reporting |
| 3 | S10 projection, 2-pass cascade | Phase B 3D loss, hyperparameter sweep | S8 MAP blend, tune σ₀, joint L/R PCA |
| 4 | inference packaging, timing | S9 refiner, fold 1 | ensembling, worst-case analysis, write-up |

**The single most important scheduling rule:** nobody trains anything until Dev A's round-trip test passes. A 0.5 mm bug in the camera convention is indistinguishable from "the model isn't learning yet", and teams lose weeks to it.

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Concha / inner-helix landmarks are occluded in every view** | High | Depth peeling (2 layers), two extra views angled into the cavity, and S9 patch refiner which works on true 3D. Track per-contour MD from week 2 so you see this early. |
| Ear crop misses at inference | Medium | 2-pass cascade + ±8 mm translation augmentation. Fallback: tiny 3D CNN on a 64³ occupancy grid regressing both ear centres (trains in 20 min). |
| Annotation-order flips in some subjects | Medium | Data-audit check 5; quarantine offenders; a contour-smoothness loss also suppresses their damage. |
| Overfitting on 400 ears | High | Mirroring (2×), exact in-plane affine augmentation, ImageNet-pretrained W18 not W32, PCA prior loss, 5-fold ensembling. |
| Colab preemption mid-run | Certain | Per-epoch checkpoint to Drive + unconditional resume. Never rely on a run finishing. |
| Dataloader starves the GPU (2 vCPU) | Medium | Cache RaycastingScenes in RAM, render at 256 not 384, `persistent_workers=True`, and if needed drop to 3 views during Phase A and use 5 only in Phase B. |
| Surface projection *increases* error | Low–Medium | Gate on validation (S10); apply selectively. |
| Full dataset arrives late (NDA) | Medium | Build and validate the whole pipeline on the single sample subject with synthetic augmentation; every module has a unit test that does not need 200 subjects. |

---

## 7. Definition of done, per milestone

- **M1:** audit report + B0 baseline MD number, on the board.
- **M2:** render round-trip error < 0.05 mm; a debug notebook that overlays projected GT landmarks on all 5 rendered views for any subject.
- **M3:** Stage-1 overfits a single ear to < 0.3 mm (proves the whole chain is wired correctly).
- **M4:** val MD < 2.5 mm from Stage 1 alone.
- **M5:** val MD < 2.0 mm with PCA + refinement + projection.
- **M6:** `infer.py head.ply → two 85×3 CSVs`, under 5 s per head, reproducible from a clean Colab runtime with one command.

---

## 8. References for the design choices

1. **Zhou, Z. et al.** *3D Landmark Localization in Point Clouds for the Human Ear.* IEEE FG 2020. — closest published analogue; offset-voting + 3DMM augmentation. https://ieeexplore.ieee.org/document/9320194/
2. **Pausch, F., Perfler, F., Holighaus, N., Majdak, P.** *Prediction of parameters of a pinna model from synthetic geometries using a vision transformer.* JASA 159(6), 2026, 5578–5598. — validates multi-view rendering of pinnae; 1.34–1.98 mm Hausdorff. https://doi.org/10.1121/10.0043919
3. **Dai, H., Pears, N., Smith, W.** *A Data-Augmented 3D Morphable Model of the Ear.* FG 2018 / Pattern Recognition Letters 2019. — the York ear PCA model; justifies S8. https://ieeexplore.ieee.org/document/8373858/
4. **Sun, K. et al.** *Deep High-Resolution Representation Learning for Human Pose Estimation (HRNet).* CVPR 2019. — the S5 backbone. https://arxiv.org/abs/1902.09212
5. **Nibali, A. et al.** *Numerical Coordinate Regression with Convolutional Neural Networks (DSNT).* arXiv:1801.07372, 2018. — soft-argmax + JS regularisation, the S6 loss recipe.
6. **Iskakov, K. et al.** *Learnable Triangulation of Human Pose.* ICCV 2019. — differentiable multi-view triangulation with learned confidences; S7.
7. **Lang, Y. et al.** *TS-MDL: Two-Stage Mesh Deep Learning for Landmark Localization.* arXiv:2109.11941. — crop-then-refine cascade; S11.
8. **CHaRM: Conditioned Heatmap Regression on Point Clouds.** arXiv:2501.13073, 2025. — patch-conditioned heatmaps; S9.
9. **Zhang, F. et al.** *D-CPT: 3D landmark detection on human point clouds — benchmark and dual cascade point transformer.* arXiv:2401.07251. — anchor-free heatmap detection on dense human point clouds.
10. **Engel, I., Daugintis, R. et al.** *The SONICOM HRTF Dataset.* JAES 71(5), 2023. — the data domain and downstream HRTF motivation. https://arxiv.org/abs/2507.05053
11. **Koenderink, J.J., van Doorn, A.J.** *Surface shape and curvature scales.* Image and Vision Computing 10(8), 1992. — shape index used as a render channel in S3.
