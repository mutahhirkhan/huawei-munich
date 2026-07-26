# Automatic Extraction of Anatomical Landmarks of Pinna Shape

## Problem Statement

The objective of this challenge is to automatically extract anatomical landmarks of the **human pinna (outer ear)** from a **3D mesh scan**.

The submitted model must:

- Accept a **3D mesh (.ply)** as input.
- Predict **85 anatomical landmarks** for **each ear** (left and right).
- Output accurate 3D coordinates corresponding to predefined pinna contours.

The predicted landmarks describe four important anatomical structures:

1. **Outer Helix** – 25 points
2. **Concha Outline** – 30 points
3. **Inner Helix** – 20 points
4. **Superior Antihelix** – 10 points

Total:

- **85 landmarks × 3 coordinates (X, Y, Z)** per ear.

---

# Resources

- [**Example Code, Sample Data & Proposal Template**](https://drive.google.com/drive/folders/19WmltlHFZUsf68STaIPT4_24F6EfpQJO?usp=drive_link) 

---

# Overall Pipeline

```text
3D Mesh (.ply)
        │
        ▼
Preprocessing
        │
        ▼
Pinna Detection
        │
        ▼
Landmark Estimation Model
        │
        ▼
85 Landmark Coordinates
(Left Ear + Right Ear)
```

---

## Figure 1 — Overall Landmark Estimation Process

![Figure 1 - Landmark Estimation Pipeline](https://cdn.fs.agorize.com/px5CEPUESguwaXRz1YkK)

*Figure 1. Block diagram of the pinna landmark estimation process.*

---

# Input

The dataset contains:

- 3D head and upper torso scans
- Mesh format: **PLY**
- Ground truth annotations:
  - Left ear
  - Right ear
- 85 landmarks per ear

The scans were captured using:

- **EinScan Pro 2X 2020**

After scanning, every mesh underwent:

- Removal of unwanted mesh regions
- Hole filling
- Anatomical alignment

---

# Anatomical Coordinate System

Every mesh is aligned using the **interaural anatomical reference frame**.

## Axes

### Y-axis

Runs from:

**Left ear canal → Right ear canal**

### X-axis

Runs from:

**Back of the head → Tip of the nose**

### Z-axis

Runs vertically upward toward the top of the head.

The head center is defined as the intersection of the X, Y, and Z axes.

This standardized alignment ensures consistent landmark annotation across all subjects.

---

## Figure 2 — Anatomical Coordinate System

![Figure 2 - Anatomical Coordinate System](https://cdn.fs.agorize.com/CSUZOH68Rsij5a1TMtDn)

*Figure 2. Anatomical coordinate system used to align all 3D meshes.*

---

# Landmark Categories

Each ear is represented using four anatomical contours.

| Structure | Number of Points |
|-----------|-----------------:|
| Outer Helix | 25 |
| Concha Outline | 30 |
| Inner Helix | 20 |
| Superior Antihelix | 10 |
| **Total** | **85** |

---

## Figure 3 — Pinna Contours

![Figure 3 - Pinna Contours](https://cdn.fs.agorize.com/1izn3bA2RhuuBgAv6HGM)

*Figure 3. The four annotated pinna contours: Outer Helix (25), Concha Outline (30), Inner Helix (20), and Superior Antihelix (10), totaling 85 landmarks.*


---

The landmarks are stored as an **85 × 3** matrix.

Each row contains:

```text
[x, y, z]
```

representing the coordinates of one anatomical landmark.

---

# Expected Output

For **both ears**, the model should output an:

```text
85 × 3
```

landmark matrix containing:

- Outer helix
- Concha outline
- Inner helix
- Superior antihelix

Each coordinate should accurately correspond to its anatomical location on the pinna.

---

# Evaluation Metric

Performance is evaluated using the **Mean Euclidean Distance (MD)** between predicted and ground-truth landmarks.

---

## Ground Truth

For subject **j** and one ear,

$$
L_{gt}^{j,\mathrm{ear}}
=
\left\{
l_{gt,1}^{j,\mathrm{ear}},
l_{gt,2}^{j,\mathrm{ear}},
\ldots,
l_{gt,N}^{j,\mathrm{ear}}
\right\}
$$

---

## Prediction

$$
L_{out}^{j,\mathrm{ear}}
=
\left\{
l_{out,1}^{j,\mathrm{ear}},
l_{out,2}^{j,\mathrm{ear}},
\ldots,
l_{out,N}^{j,\mathrm{ear}}
\right\}
$$

where:

- $N = 85$

Each landmark contains its 3D coordinates.

---

## Distance for One Ear

$$
d\!\left(
L_{out}^{j,\mathrm{ear}},
L_{gt}^{j,\mathrm{ear}}
\right)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
l_{out,i}^{j,\mathrm{ear}}
-
l_{gt,i}^{j,\mathrm{ear}}
\right\|
$$

This computes the average Euclidean distance between corresponding predicted and ground-truth landmarks.

A **lower distance** indicates better performance.

---

## Final Metric

Across all **M** subjects and both ears,

$$
MD
=
\frac{1}{2M}
\sum_{j=1}^{M}
\sum_{\mathrm{ear}}
d\!\left(
L_{out}^{j,\mathrm{ear}},
L_{gt}^{j,\mathrm{ear}}
\right)
$$

where:

- $M$ = number of subjects
- $\mathrm{ear} \in \{\mathrm{left}, \mathrm{right}\}$

The objective is to **minimize MD**.

---

# Dataset

## Sample Dataset

Initially provided:

- One 3D mesh
- Left ear annotations
- Right ear annotations
- Example notebook

The notebook demonstrates:

- Loading meshes
- Reading landmarks
- Basic visualization
- Starter implementation

---

## Full Dataset

The complete dataset contains:

- 200 subjects
- Full head and torso 3D meshes
- Annotated landmarks for both ears

Access requires:

- A signed NDA from **every team member**
- Huawei approval

Only after successful verification is access granted.


---

# References

1. Outer Ear Anatomy  
   https://en.wikipedia.org/wiki/Outer_ear

2. Engel, Isaac, et al. (2023).  
   *The SONICOM HRTF Dataset.* Journal AES.

3. Kahana Yuvi & Philip Nelson (2007).  
   *Boundary element simulations of the transfer function of human heads and baffled pinnae using accurate geometric models.* Journal of Sound and Vibration.