# HELIX — Engineering Playbook

Companion to `HELIX_Proposal_2026-07-29.pdf`. The proposal is the 3-page submission artefact; this is
the internal build document. Nothing here needs to be shown to the organisers.

---

## 0. What I took from each of your three drafts

| Source | Kept | Dropped, and why |
|---|---|---|
| **Mouazzama (MPAN)** | Heatmap prediction instead of direct coordinate regression; soft-argmax decoding; local patch refinement; anatomical constraint layer; barycentric surface projection. The observation that GT landmarks are **off-vertex** is the single most useful finding in all three drafts. | The InfoGNN mesh-graph branch. A message-passing GNN over a 366 k-vertex graph is a memory and wall-clock disaster, and PTv3 already sees local topology through its neighbourhood attention. The dual-stream cross-attention fusion adds parameters you cannot afford to fit with 400 ears. |
| **Rizwan (multi-view HRNet)** | Left→right mirroring; multi-view 2.5D rendering with depth/normal/curvature channels; confidence-weighted view fusion; PCA shape correction; honest expected-accuracy numbers. This is the most *realistic* of the three. | Nothing, but demoted: multi-view is ensemble member **C**, not the primary. Rendering cannot see into the concha under the helix overhang, and 30 of the 85 points live there. |
| **Mutahhir (PTv3 + literature review)** | PTv3 backbone; the voxel-then-FPS downsampling order; the literature survey, which is the best-researched part of any draft. The conclusion "no single published method covers the task; the winning recipe is a hybrid" is correct. | BCPD for the shape prior. Under a Gaussian prior with Gaussian observation noise, the MAP estimate is closed-form weighted least squares. BCPD is an iterative solver for a problem you do not have (unknown correspondence) — your 85 points are already indexed. |

**The load-bearing idea none of the three drafts had:** the 85 points are not 85 detections. They are ordered samples of four curves. That single reframing is what licenses the PCA prior, the arc-length resampling, the contour-ID embeddings, and averaging ensemble members directly in landmark space.

---

## 1. Do this before you write any model code

Everything below runs on the one sample you already have. It takes an afternoon and it will change your architecture decisions. **Do not skip it.** More competition points are lost to a silent convention mismatch than to a weak backbone.

```python
# step0_forensics.py
import re, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree

CONTOURS = {                    # 0-indexed, read off Figure 3 of the brief — ASSERT, don't assume
    "outer_helix":    (0, 25),
    "concha":         (25, 55),
    "inner_helix":    (55, 75),
    "sup_antihelix":  (75, 85),
}

def load_landmarks(path):
    """Parses '0 [-0.75018139 74.57184769 31.7642007 ]' style rows."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
        rows.append(nums[-3:])                       # drop the leading index
    L = np.asarray(rows, dtype=np.float64)
    assert L.shape == (85, 3), f"expected (85,3), got {L.shape}"
    return L

def report(ply_path, csv_left, csv_right):
    mesh = trimesh.load(ply_path, process=False)
    print(f"mesh: {len(mesh.vertices)} verts  {len(mesh.faces)} faces  "
          f"watertight={mesh.is_watertight}  bbox={mesh.bounds.round(1).tolist()}")
    print(f"median edge length: {np.median(mesh.edges_unique_length):.3f}")

    for side, csv in (("left", csv_left), ("right", csv_right)):
        L = load_landmarks(csv)

        # --- Q1: are the landmarks ON the surface? point-to-TRIANGLE, not point-to-vertex ---
        closest, d_surf, tri = trimesh.proximity.closest_point(mesh, L)
        d_vert, _ = cKDTree(mesh.vertices).query(L)
        n = mesh.face_normals[tri]
        signed = np.einsum("ij,ij->i", L - closest, n)       # + = outside the surface

        print(f"\n[{side}]  y-range {L[:,1].min():.1f}..{L[:,1].max():.1f}  "
              f"(sign tells you the L/R convention)")
        print(f"  dist to nearest VERTEX   mean {d_vert.mean():.4f}  max {d_vert.max():.4f}")
        print(f"  dist to nearest TRIANGLE mean {d_surf.mean():.4f}  max {d_surf.max():.4f}")
        print(f"  signed normal offset     mean {signed.mean():+.4f}  std {signed.std():.4f}")

        # --- Q2: is the spacing along each contour uniform in arc length? ---
        for name, (a, b) in CONTOURS.items():
            step = np.linalg.norm(np.diff(L[a:b], axis=0), axis=1)
            print(f"  {name:14s} n={b-a:2d}  step mean {step.mean():5.2f}  "
                  f"CV {step.std()/step.mean():.3f}  max/min {step.max()/step.min():.2f}")
```

### How to read the output — this is the whole point

| Observation | Conclusion | Action |
|---|---|---|
| `d_surf` ≈ 0.00–0.05 | GT lies exactly on the surface | **Barycentric surface projection is free accuracy.** Turn it on. |
| `d_surf` ≈ 0.2–0.3, `signed` mean ≈ 0, std large | Annotations are noisy about the surface | Do *not* project. Projection would inject bias. |
| `d_surf` small but `signed` mean clearly > 0 | Systematic outward offset (annotated on a smoothed/offset surface) | Learn a per-landmark signed normal offset and add it back. |
| `d_vert` ≈ 0.25 but `d_surf` ≈ 0 | On-surface, between vertices — as Mouazzama found | Confirms you must predict **continuous** coordinates. Vertex classification caps you at ~0.25 mm error before you start. |
| Contour `CV` < 0.10 | Uniform arc-length spacing | **Fit a spline per contour and resample to uniform arc length as a post-process.** This is nearly free MD. |
| Contour `CV` > 0.25 | Points are anchored to semantic features | Do not resample. The prior must carry the spacing instead. |

Also run, across all 400 files once you have the full set:

1. **Index-order consistency.** Procrustes-align every ear to the running mean, then look at per-ear
   residual. Ears with a residual > 3σ usually have a reversed contour or a swapped block — real
   label noise that will otherwise poison training.
2. **Mirror validity.** `L_left @ diag(1,-1,1)` should Procrustes-align to the mean right ear with a
   residual comparable to right-vs-right. If it doesn't, the mirror axis or the index order differs
   between sides, and blind mirroring will silently halve your data quality.
3. **Units.** Y ranging 63–91 with the head centre at the origin means the ear canal sits ≈ 74 mm
   lateral — half an interaural distance. So **coordinates are millimetres and MD is in millimetres.**
   Never scale-normalise per sample without unscaling the output; the metric is absolute.

---

## 2. Repo layout

```
helix/
  data/       forensics.py  canonicalise.py  roi.py  dataset.py  augment_tps.py
  models/     ptv3_head.py  sparse_unet.py  multiview.py  refine.py
  prior/      pca.py  fit.py  contour.py
  train.py    infer.py  eval.py
  configs/    ptv3.yaml  sparse.yaml  mv.yaml
  cache/      *.npz          # ROI clouds + features, built once
```

Cache aggressively. Preprocessing 200 × 366 k-vertex meshes is minutes per epoch if you redo it and
seconds if you don't.

---

## 3. Stage specifications

### 3.1 ROI extraction (`data/roi.py`)

The meshes are pre-aligned, so start with a statistical box, not a detector:

```python
# from the training landmarks, per side, in the global frame
lo = np.percentile(all_landmarks, 1, axis=0) - 15.0     # 15 mm margin
hi = np.percentile(all_landmarks, 99, axis=0) + 15.0
```

Then refine with FPFH + RANSAC + point-to-plane ICP against a mean-ear template, **rigid only** —
allowing scale would destroy the metric. Gate on ICP fitness; if it fails, fall back to the box.
Store the rigid transform so you can map predictions back to the global frame.

Downsample in this order, always: **crop → voxel downsample (O(N)) → FPS to 24 k**. Running FPS on
366 k points is O(N²) and is the bottleneck Mutahhir correctly identified.

### 3.2 Per-point features

`[xyz_local (3), normal (3), κ₁, κ₂, shape index, curvedness (4), HKS (8–16)]`

Heat-kernel signatures are intrinsic and rigid-invariant, which makes them unusually good at marking
the helix ridge and the antihelix fork. Compute once, cache.

### 3.3 Heatmap targets — use geodesic distance, not Euclidean

This matters more for ears than for almost any other anatomy. The helix rim **folds over** the
scapha, so two points a millimetre apart in Euclidean space can be 15 mm apart along the surface. A
Euclidean Gaussian heatmap bleeds straight through the fold and teaches the model to vote for points
on the wrong side of the rim.

```python
import potpourri3d as pp3d
solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
geo = solver.compute_distance(nearest_vertex_index_of_landmark_k)
target_k = np.exp(-geo**2 / (2 * sigma**2))
```

Anneal σ from 3.0 mm to 1.5 mm over training — wide early for a usable gradient signal, tight later
for precision.

### 3.4 Losses

```python
def coord_loss(pred, gt, delta=1.0):
    """The challenge metric itself, Huberised near zero so gradients behave at convergence."""
    e = torch.linalg.norm(pred - gt, dim=-1)                    # (B, 85), millimetres
    return torch.where(e < delta, 0.5 * e ** 2 / delta, e - 0.5 * delta).mean()

def smooth_loss(pred, contours):
    """Second difference along each contour — penalises kinks, not curvature."""
    total = 0.0
    for a, b in contours:
        c = pred[:, a:b]
        total = total + torch.linalg.norm(c[:, :-2] - 2 * c[:, 1:-1] + c[:, 2:], dim=-1).mean()
    return total / len(contours)
```

Full objective: `L = 1.0·L_coord + 0.5·L_heat(focal) + 0.5·L_offset(smooth-L1, masked ≤6 mm)
+ 0.1·L_smooth + 0.05·L_prior(Mahalanobis on b) + 0.1·L_sym + 0.2·L_NLL`.

Tune `λ_coord` last and leave it at 1.0 — it is the thing you are actually scored on. **Do not use
MSE.** MSE optimises the mean of squared distances; you are scored on the mean of distances.

### 3.5 PCA shape prior (`prior/pca.py`, `prior/fit.py`)

```python
def build(X, K=40):
    """X: (n_ears, 255) canonicalised landmark vectors, after rigid GPA."""
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    lam = S ** 2 / (len(X) - 1)
    return mu, Vt[:K], lam[:K]                       # basis B is (K, 255)

def fit(lhat, sigma, mu, B, lam, damp=1.0):
    """Closed-form sigma-weighted MAP fit. lhat (85,3), sigma (85,)."""
    y = lhat.reshape(-1) - mu
    w = np.repeat(1.0 / np.maximum(sigma, 1e-3) ** 2, 3)
    A = B * w                                        # (K, 255)
    b = np.linalg.solve(A @ B.T + damp * np.diag(1.0 / lam), A @ y)
    return (mu + B.T @ b).reshape(-1, 3), b

def blend(lhat, l_pca, sigma, lo=0.5, hi=2.0):
    """Trust the network where it is confident; trust anatomy where it is not."""
    a = np.clip((sigma - lo) / (hi - lo), 0.0, 1.0)[:, None]
    return (1 - a) * lhat + a * l_pca
```

Report the variance spectrum in your final write-up — if 40 modes give > 98 %, that is a strong,
citable argument for the whole design.

Build **two** models: per-ear (255-dim) and joint left+right (510-dim). The joint one exploits
within-subject symmetry and is usually the better regulariser; validate both.

### 3.6 TPS shape-synthesis augmentation (`data/augment_tps.py`)

The highest-leverage component. Sample new landmark configurations from the PCA model, then drag the
mesh along with them:

```python
def tps_solve(S, T, lam=1e-3):
    """3D thin-plate spline. Kernel U(r)=|r| is the 3D biharmonic Green's function."""
    n = len(S)
    K = np.linalg.norm(S[:, None] - S[None], axis=-1)          # U(r) = r in 3D
    P = np.hstack([np.ones((n, 1)), S])
    M = np.block([[K + lam * np.eye(n), P], [P.T, np.zeros((4, 4))]])
    return np.linalg.solve(M, np.vstack([T, np.zeros((4, 3))]))

def tps_apply(W, S, V, chunk=20000):
    out = np.empty_like(V)
    for i in range(0, len(V), chunk):                          # keep the kernel matrix bounded
        v = V[i:i + chunk]
        K = np.linalg.norm(v[:, None] - S[None], axis=-1)
        P = np.hstack([np.ones((len(v), 1)), v])
        out[i:i + chunk] = K @ W[:len(S)] + P @ W[len(S):]
    return out
```

Two details that decide whether this works or produces garbage:

- **Add anchor points.** Include ~60 points sampled on the ROI boundary in both `S` and `T`,
  unchanged. Without them the warp extrapolates wildly outside the landmark hull and folds the mesh.
- **Reject implausible samples.** Draw `b ~ N(0, Λ)` truncated at ±2.5σ per mode, and discard any
  warp whose Jacobian determinant goes negative anywhere (self-intersection). Eyeball the first 50.

Target 30–50 k synthetic ears. Keep a real-only validation fold — never validate on synthetic data.

### 3.7 Cascaded refinement

Crop 1024 points within r of the current estimate, re-centre, feed a shared tiny-PTv3 conditioned on
`emb(landmark_id) + emb(contour_id)`, predict Δl. Two passes, r: 8 mm → 4 mm. The point is that each
landmark now sees the mesh at **full resolution** rather than the 24 k-point global subsample. Train
the refiner on perturbed ground truth (jitter the input by N(0, 2 mm)) so it learns to correct, not
to memorise.

---

## 4. Validation protocol

- **5-fold, subject-wise.** Both ears of a subject in the same fold. Left and right ears of one
  person are strongly correlated; splitting ear-wise leaks and will show you a fake 20 % improvement.
- Track: overall MD; per-contour MD; a per-landmark MD bar chart; p50/p90/p95; worst-subject MD.
- Keep a running **ablation table** from day one: `S3 → +S4 → +S5 → +S6`. If the prior or the cascade
  are not paying for themselves, delete them. They are strict post-processes, so this costs nothing.
- Plot a **reliability diagram** for σ̂ (predicted uncertainty vs. realised error). If σ̂ is not
  calibrated, the whole Step-4 weighting is meaningless — this is the one thing that can quietly
  break the design.

---

## 5. Eight-week plan

| Wk | Work | Owner |
|---|---|---|
| 1 | Step 0 forensics; canonicalisation; ROI; cache builder; **PCA model + variance spectrum** | Data |
| 2 | Dataset/loader; augmentation (rigid, noise, dropout); PTv3 skeleton training end-to-end on the sample | Data + Model |
| 3 | Heatmap + offset head; geodesic targets; first honest CV number | Model |
| 4 | TPS synthesis engine; retrain with synthetic data; sparse U-Net member | Data + Model |
| 5 | Uncertainty head + calibration; Step-4 prior fit; contour resampling; ablation table | Model + Eval |
| 6 | Cascade refiner; multi-view member; ensemble plumbing | All |
| 7 | Full 5-fold × 3 backbones; TTA; hyper-parameter sweep on the top member only | All |
| 8 | Freeze; package CLI + TorchScript; write-up; reproducibility check from a clean clone | All |

Get a *bad but complete* end-to-end number by end of week 3. A pipeline that scores 4 mm in week 3
beats a brilliant architecture that first runs in week 7.

---

## 6. Things that will cost you points if you get them wrong

1. **Scale-normalising per sample and forgetting to unscale.** The metric is absolute millimetres.
2. **Ear-wise CV splits.** Inflates your validation, then the leaderboard corrects you in public.
3. **FPS on the raw mesh.** Voxel first. Always.
4. **Euclidean heatmap targets.** The helix fold will punish you specifically in the region with the
   most landmarks.
5. **Projecting to the surface without checking whether GT is on the surface.** Section 1 tells you.
6. **Trusting the contour index blocks from a figure.** Assert them in code against the full dataset.
7. **Tuning on the test split.** Pick one number, one protocol, and defend it.
8. **Blind mirroring.** Verify the mirrored left ear actually aligns to right ears first.
