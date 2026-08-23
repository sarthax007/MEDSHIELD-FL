import os
import cv2
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from shared.medshield.model.vit import TumorClassifier
from shared.medshield.explain.vit_cam import (
    ViTGradCAM,
    reshape_transform_vit_timm,
    show_cam_on_image,
)


def validate_explanations():
    # Setup paths
    out_dir = "docs/explainability_evidence"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load an untrained ViT model to establish baseline
    print("Loading initialized ViT model (random weights)...")
    model = TumorClassifier(pretrained=False)
    model.eval()

    target_layer = model.backbone.blocks[-1].norm1  # type: ignore

    cam_extractor = ViTGradCAM(
        model=model,
        target_layer=target_layer,
        reshape_transform=reshape_transform_vit_timm,
    )

    # 2. Load Sample Data
    image_path = "data/raw/BraTS2021_00495_t2.nii.gz"
    if not os.path.exists(image_path):
        # fallback to flair if t2 was missing
        image_path = "data/raw/BraTS2021_00495_flair.nii.gz"
        if not os.path.exists(image_path):
            print(f"Sample data not found. Looked for {image_path}")
            return

    print(f"Loading sample scan from: {image_path}")
    nii_img = nib.load(image_path)
    img_data = nii_img.get_fdata()  # type: ignore

    # Get a middle slice
    slice_idx = img_data.shape[2] // 2
    img_2d = img_data[:, :, slice_idx]

    # Resize to 224x224
    img_2d = cv2.resize(img_2d, (224, 224))

    # Normalize to [0, 1]
    img_2d_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)

    # Convert to tensor [1, 3, 224, 224]
    img_3d = np.stack([img_2d_norm] * 3, axis=0)
    input_tensor = torch.tensor(img_3d, dtype=torch.float32).unsqueeze(0)

    # 3. Generate CAM
    cam = cam_extractor(input_tensor)

    # Visual Overlay
    vis_img = np.stack([img_2d_norm] * 3, axis=-1)
    cam_vis = show_cam_on_image(vis_img, cam, use_rgb=True)

    # 4. Generate Side-by-Side Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_2d_norm, cmap="gray", origin="lower")
    axes[0].set_title("Original Scan (T2/FLAIR)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(cam_vis, origin="lower")
    axes[1].set_title("Untrained Baseline Grad-CAM Heatmap", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "baseline_gradcam_comparison.png")
    plt.savefig(out_file)
    print(f"Saved visual comparison to {out_file}")


if __name__ == "__main__":
    validate_explanations()
