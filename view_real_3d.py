import os
import matplotlib.pyplot as plt
import nibabel as nib

# Point to both the MRI and the Segmentation Mask
flair_path = "data/raw/BraTS2021_00495_t2.nii.gz"
seg_path = "data/raw/BraTS2021_00495_seg.nii.gz"

if not os.path.exists(flair_path) or not os.path.exists(seg_path):
    print(
        "Oops! Could not find the files. Make sure both the flair and seg files are in data/raw/"
    )
else:
    print("Loading 3D Medical Volumes...")

    # Load the 3D NIfTI volumes using nibabel
    flair_img = nib.load(flair_path)
    seg_img = nib.load(seg_path)

    # Extract the data into standard Numpy arrays
    flair_data = flair_img.get_fdata()  # type: ignore
    seg_data = seg_img.get_fdata()  # type: ignore

    print(f"Successfully loaded scans with shape: {flair_data.shape}")

    # Extract the middle 2D slice to view (usually where the tumor is most visible)
    middle_slice_idx = 100
    flair_slice = flair_data[:, :, middle_slice_idx]
    seg_slice = seg_data[:, :, middle_slice_idx]

    # Display the real brain scan and the mask side-by-side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Plot the Grayscale MRI
    axes[0].imshow(flair_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Real BraTS MRI (Slice #{middle_slice_idx})")
    axes[0].axis("off")

    # 2. Plot the Colored Tumor Mask
    # The BraTS dataset uses specific integer labels (1, 2, 4) for different tumor regions.
    axes[1].imshow(seg_slice.T, cmap="nipy_spectral", origin="lower")
    axes[1].set_title("Ground-Truth Tumor Mask")
    axes[1].axis("off")

    print("Opening medical image viewer...")
    plt.tight_layout()
    plt.show()
