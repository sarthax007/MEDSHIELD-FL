# BraTS 2021 Data Dictionary

> **MedShield-FL — Task 11**
> Inspect and document the BraTS structure.

---

## 1. Overview

The **BraTS 2021** (Brain Tumor Segmentation) dataset contains multi-parametric
MRI (mpMRI) scans of brain tumor patients. Each patient case includes four MRI
modalities and one segmentation mask, all co-registered to the same anatomical
template (SRI24) and interpolated to a uniform isotropic resolution.

| Property             | Value                                  |
| -------------------- | -------------------------------------- |
| Source               | RSNA-ASNR-MICCAI BraTS 2021 Challenge |
| Number of cases      | ~1,251 training subjects               |
| File format          | NIfTI (`.nii.gz`)                      |
| Voxel dimensions     | 240 × 240 × 155                       |
| Voxel spacing        | 1 mm × 1 mm × 1 mm (isotropic)        |
| Data type            | float32 (modalities), uint8 (labels)   |
| Coordinate system    | LPS+ (Left-Posterior-Superior)         |

---

## 2. MRI Modalities

Each patient directory contains **four modality volumes** plus one segmentation
label volume. All five share the same spatial dimensions and affine transform.

| Filename suffix | Modality                                | Clinical purpose                                                    |
| --------------- | --------------------------------------- | ------------------------------------------------------------------- |
| `_t1.nii.gz`    | **T1-weighted** (native)                | Anatomy, grey/white matter contrast                                 |
| `_t1ce.nii.gz`  | **T1-weighted contrast-enhanced** (T1ce)| Highlights enhancing tumor (gadolinium uptake)                      |
| `_t2.nii.gz`    | **T2-weighted**                         | Edema and non-enhancing tumor appear hyperintense                   |
| `_flair.nii.gz` | **T2 FLAIR** (Fluid-Attenuated IR)      | Edema without CSF signal; best for peritumoral infiltration         |
| `_seg.nii.gz`   | **Segmentation label**                  | Voxel-level ground-truth tumor annotation                           |

### Intensity characteristics

| Modality | Typical range (after skull stripping) | Notes                                                           |
| -------- | ------------------------------------- | --------------------------------------------------------------- |
| T1       | 0 – ~3,000                            | Background = 0; brain tissue varies by scan                     |
| T1ce     | 0 – ~4,000                            | Enhancing tumor appears bright; range can be wider than T1      |
| T2       | 0 – ~5,000                            | Fluid (CSF, edema) is bright                                    |
| FLAIR    | 0 – ~4,000                            | CSF suppressed; edema remains bright                            |

> **Note:** Raw intensity ranges vary per scan. Normalisation (e.g., z-score
> per volume) is applied at the preprocessing stage (Task 14).

---

## 3. Segmentation Label Definitions

The segmentation mask (`_seg.nii.gz`) uses the following integer labels:

| Label value | Name                           | Clinical definition                                                    | BraTS region    |
| ----------- | ------------------------------ | ---------------------------------------------------------------------- | --------------- |
| **0**       | Background / healthy tissue    | Normal brain and non-brain voxels                                      | —               |
| **1**       | Necrotic / non-enhancing tumor | Necrotic tumor core (NCR) and non-enhancing tumor (NET)                | Tumor Core (TC) |
| **2**       | Peritumoral edema              | Edematous / invaded tissue surrounding the tumor                       | Whole Tumor (WT)|
| **4**       | GD-enhancing tumor             | Active tumor showing gadolinium uptake on T1ce                         | Enhancing Tumor (ET) |

> **Label 3 is intentionally absent.** The original BraTS annotation merged
> labels 1 and 3 into a single class (label 1). Code must handle {0, 1, 2, 4}
> not {0, 1, 2, 3}.

### Derived evaluation regions

| Region           | Label composition | Clinical meaning                                |
| ---------------- | ----------------- | ----------------------------------------------- |
| Whole Tumor (WT) | 1 + 2 + 4         | Everything abnormal                             |
| Tumor Core (TC)  | 1 + 4             | Core without surrounding edema                  |
| Enhancing Tumor (ET) | 4             | Only the active enhancing component             |

---

## 4. Expected Tumor Class Distribution (BraTS 2021 Training)

Based on published statistics of the BraTS 2021 training set:

| Tumor grade        | Approx. count | Percentage |
| ------------------- | ------------- | ---------- |
| High-Grade Glioma (HGG) | ~1,000   | ~80%       |
| Low-Grade Glioma (LGG)  | ~251     | ~20%       |

> The dataset is **imbalanced toward HGG**. This matters for federated
> partitioning (Task 17) and for the active-learning query strategy (Level 5).

### Per-voxel label distribution (approximate, across all subjects)

| Label | Mean % of brain voxels | Notes                           |
| ----- | ---------------------- | ------------------------------- |
| 0     | ~97–98%                | Healthy tissue dominates        |
| 1     | ~0.5–1.0%              | Necrotic / non-enhancing core   |
| 2     | ~1.0–2.0%              | Peritumoral edema               |
| 4     | ~0.3–0.5%              | Enhancing tumor (rarest class)  |

---

## 5. Directory Layout

```
data/raw/BraTS2021_Training_Data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00001/
│   └── ...
└── BraTS2021_01250/
    └── ...
```

---

## 6. Known Anomalies and Edge Cases

The inspection script (`medshield.data.inspect`) checks for the following and
logs any anomalies found:

| Anomaly type              | What is checked                                               |
| ------------------------- | ------------------------------------------------------------- |
| Missing modalities        | Each patient must have exactly 5 files (4 modalities + seg)   |
| Inconsistent shape        | All 5 volumes in a patient must share the same dimensions     |
| Unexpected label values   | Segmentation must only contain labels {0, 1, 2, 4}           |
| Non-finite intensities    | Modality volumes must not contain NaN or ±Inf values          |
| Zero-volume segmentation  | A patient with no tumor voxels at all is flagged              |

> Run `python -m medshield.data.inspect --data-dir data/raw` to generate the
> full anomaly report after downloading the dataset.
