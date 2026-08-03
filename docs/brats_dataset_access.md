# BraTS Dataset Access and Usage Terms

## Overview

The BraTS (Brain Tumor Segmentation) dataset is the standard benchmark for medical imaging tasks involving brain tumors. For MedShield-FL, we are utilizing the **BraTS 2021** dataset version for reproducibility.

## Obtaining Access

Access to the BraTS dataset is governed by its official source. To obtain legitimate access:
1. Register for an account on the official Synapse platform or through the RSNA/ASNR/MICCAI BraTS challenge portal (e.g., [Synapse ID: syn25829067](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863)).
2. Accept the data usage agreement which typically prohibits commercial use, re-identification of patients, and requires proper citation of the dataset publications.

## Storage Location

Once downloaded, the raw dataset should be stored in the following directory:
`data/raw/`

**Note:** The `.gitignore` is explicitly configured to exclude `data/raw/**`, `*.nii.gz`, and any other imaging formats. The raw dataset is excluded from version control as it is large and privacy-sensitive.

## Data-Access Note & Usage Conditions

- **Authorized Users:** Only authorized researchers and developers associated with this project who have individually agreed to the BraTS usage terms may download or use the raw data.
- **Conditions of Use:**
  - The dataset must **never** be committed to version control.
  - Users must not attempt to re-identify any patient.
  - The data must only be used for the development and testing of the MedShield-FL federated learning framework.
  - Appropriate citations to the BraTS benchmark papers must be included in any publications resulting from this project.

## Dataset Version

- **Version/Year:** BraTS 2021 (RSNA-ASNR-MICCAI BraTS 2021)
- **Format:** NIfTI (`.nii.gz`) format containing multiple 3D MRI modalities (T1, T1ce, T2, FLAIR) and corresponding segmentation labels.
