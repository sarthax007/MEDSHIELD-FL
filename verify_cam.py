import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.widgets import RadioButtons, Slider
from shared.medshield.model.vit import TumorClassifier
from shared.medshield.explain.vit_cam import (
    ViTGradCAM,
    reshape_transform_vit_timm,
    show_cam_on_image,
)


def test_vit_cam():
    print("Loading AI Model and Medical Data...")
    model = TumorClassifier(pretrained=False)
    model.eval()

    target_layer = model.backbone.blocks[-1].norm1  # type: ignore
    cam_extractor = ViTGradCAM(
        model=model,
        target_layer=target_layer,
        reshape_transform=reshape_transform_vit_timm,
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    plt.subplots_adjust(left=0.35, bottom=0.2)

    # Load Real 3D Data Volume (for the Real 3D mode)
    real_path = "data/raw/BraTS2021_00495_t2.nii.gz"
    if os.path.exists(real_path):
        real_volume = nib.load(real_path).get_fdata()  # type: ignore
        max_slices = real_volume.shape[2] - 1
    else:
        real_volume = None
        max_slices = 100

    def get_synthetic_image(idx):
        path = f"data/synthetic_baseline/{idx}_slice5.npy"
        if os.path.exists(path):
            arr = np.load(path)
            img = arr[0] if arr.ndim > 2 else arr
            img = img[:224, :224]
            return (img - img.min()) / (img.max() - img.min() + 1e-8)
        return np.zeros((224, 224))

    def get_real_image(slice_idx):
        if real_volume is not None:
            img = real_volume[:, :, slice_idx].T
            img = img[:224, :224]
            if img.max() != img.min():
                return (img - img.min()) / (img.max() - img.min() + 1e-8)
        return np.zeros((224, 224))

    def compute_cam(img_norm):
        img_3d = np.stack([img_norm] * 3, axis=0)
        input_tensor = torch.tensor(img_3d, dtype=torch.float32).unsqueeze(0)
        cam = cam_extractor(input_tensor)
        vis_img = np.stack([img_norm] * 3, axis=-1)
        return show_cam_on_image(vis_img, cam, use_rgb=True)

    # Initial plot
    initial_img = compute_cam(get_synthetic_image(7))
    img_plot = ax.imshow(initial_img)
    ax.set_title("Heatmap: Synthetic Baseline (Patient 7)")
    ax.axis("off")

    # UI Elements
    ax_radio = plt.axes([0.05, 0.5, 0.25, 0.15])  # type: ignore
    radio = RadioButtons(ax_radio, ["Synthetic Baseline", "Real 3D MRI Slice"])

    ax_slider = plt.axes([0.35, 0.05, 0.5, 0.03])  # type: ignore
    slider = Slider(ax_slider, "Image ID / Z-Slice", 1, 30, valinit=7, valstep=1)

    # Dynamically change the slider bounds based on the mode selected
    def update_mode(label):
        if label == "Synthetic Baseline":
            slider.valmin = 1
            slider.valmax = 30
            slider.ax.set_xlim(1, 30)
            slider.set_val(7)
        else:
            slider.valmin = 0
            slider.valmax = max_slices
            slider.ax.set_xlim(0, max_slices)
            slider.set_val(100)
        fig.canvas.draw_idle()

    # Dynamically compute the heatmap based on the slider value
    def update_slider(val):
        idx = int(slider.val)
        mode = radio.value_selected

        if mode == "Synthetic Baseline":
            img = get_synthetic_image(idx)
            cam = compute_cam(img)
            img_plot.set_data(cam)
            ax.set_title(f"Heatmap: Synthetic Baseline (Patient {idx})")
        else:
            img = get_real_image(idx)
            cam = compute_cam(img)
            img_plot.set_data(cam)
            ax.set_title(f"Heatmap: Real 3D MRI (Z-Slice {idx})")

        fig.canvas.draw_idle()

    radio.on_clicked(update_mode)
    slider.on_changed(update_slider)

    print("Opening Interactive Grad-CAM window...")
    plt.show()


if __name__ == "__main__":
    test_vit_cam()
