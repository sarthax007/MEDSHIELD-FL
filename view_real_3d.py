import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Pointing to the real BraTS 3D volume you just downloaded
file_path = "data/raw/BraTS2021_00495_flair.nii.gz"

if not os.path.exists(file_path):
    print(f"Oops! Could not find the file at: {file_path}")
else:
    print(f"Loading 3D Medical Volume from {file_path}...")

    # Load the 3D NIfTI volume using nibabel
    img = nib.load(file_path)

    # Extract the data into a standard Numpy array
    image_data = np.asanyarray(img.dataobj)  # type: ignore

    print(f"Successfully loaded 3D scan with shape: {image_data.shape}")

    # Extract the middle 2D slice to view
    middle_slice_idx = image_data.shape[2] // 2
    slice_2d = image_data[:, :, middle_slice_idx]

    # Display the real brain scan using matplotlib
    plt.figure(figsize=(6, 6))

    # Transpose and fix orientation
    plt.imshow(slice_2d.T, cmap="gray", origin="lower")

    plt.title(f"Real BraTS MRI Slice (Slice #{middle_slice_idx})")
    plt.axis("off")

    print("Opening medical image viewer...")
    plt.show()
