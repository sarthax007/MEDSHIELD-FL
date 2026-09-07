import os
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.widgets import Slider, RadioButtons

patient_id = "BraTS2021_00495"
base_dir = "data/raw/"
seg_path = os.path.join(base_dir, f"{patient_id}_seg.nii.gz")

if not os.path.exists(seg_path):
    print("Oops! Could not find the files. Make sure files are in data/raw/")
    exit()

print("Loading 3D Medical Volumes...")

modalities = ["t1", "t1ce", "t2", "flair"]
volumes = {}
for mod in modalities:
    path = os.path.join(base_dir, f"{patient_id}_{mod}.nii.gz")
    if os.path.exists(path):
        volumes[mod] = nib.load(path).get_fdata()  # type: ignore

seg_data = nib.load(seg_path).get_fdata()  # type: ignore

# Default modality
current_mod = "t2"
flair_data = volumes[current_mod]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plt.subplots_adjust(bottom=0.25, left=0.25)

initial_slice = 100

# 1. Plot the Grayscale MRI
img1 = axes[0].imshow(flair_data[:, :, initial_slice].T, cmap="gray", origin="lower")
axes[0].set_title(f"MRI Modality: {current_mod.upper()}")
axes[0].axis("off")

# 2. Plot the Colored Tumor Mask
img2 = axes[1].imshow(
    seg_data[:, :, initial_slice].T, cmap="nipy_spectral", origin="lower"
)
axes[1].set_title("Ground-Truth Tumor Mask")
axes[1].axis("off")

# Interactive Slider for 3D Depth
ax_slice = plt.axes([0.3, 0.05, 0.5, 0.03])  # type: ignore
slice_slider = Slider(
    ax=ax_slice,
    label="3D Depth (Z-Axis)",
    valmin=0,
    valmax=flair_data.shape[2] - 1,
    valinit=initial_slice,
    valstep=1,
)

# Radio Buttons for MRI Modalities
ax_radio = plt.axes([0.05, 0.4, 0.12, 0.2])  # type: ignore
radio = RadioButtons(
    ax_radio, [m.upper() for m in volumes.keys()], active=modalities.index(current_mod)
)


def update(val):
    idx = int(slice_slider.val)
    mod = radio.value_selected.lower()
    img1.set_data(volumes[mod][:, :, idx].T)
    img2.set_data(seg_data[:, :, idx].T)
    axes[0].set_title(f"MRI Modality: {mod.upper()}")
    fig.canvas.draw_idle()


slice_slider.on_changed(update)
radio.on_clicked(update)

print("Opening Interactive 3D multi-modal medical image viewer...")
plt.show()
