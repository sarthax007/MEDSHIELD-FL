import matplotlib.pyplot as plt
import numpy as np
import os

# Pointing to one of your synthetic slices
image_path = "data/synthetic_baseline/7_slice5.npy"

if not os.path.exists(image_path):
    print(f"Oops! Could not find the file at: {image_path}")
else:
    # Load the Numpy array directly
    img_array = np.load(image_path)

    # If the array has extra dimensions (like channels), flatten it for viewing
    if img_array.ndim > 2:
        img_array = img_array.squeeze()
        if img_array.ndim > 2:
            img_array = img_array[0]  # Grab just the first channel if still 3D

    plt.figure(figsize=(6, 6))
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Synthetic MRI Slice: {os.path.basename(image_path)}")
    plt.axis("off")

    print("Opening synthetic image viewer...")
    plt.show()
