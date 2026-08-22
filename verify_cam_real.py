import os
import cv2
import torch
import numpy as np
import nibabel as nib
from shared.medshield.model.vit import TumorClassifier
from shared.medshield.explain.vit_cam import (
    ViTGradCAM,
    reshape_transform_vit_timm,
    show_cam_on_image,
)


def test_vit_cam_real():
    # Load model
    model = TumorClassifier(pretrained=False)  # Random init for testing the shape
    model.eval()

    # The target layer in timm ViT-Base is typically the norm1 in the last block
    target_layer = model.backbone.blocks[-1].norm1  # type: ignore

    # Initialize CAM
    cam_extractor = ViTGradCAM(
        model=model,
        target_layer=target_layer,
        reshape_transform=reshape_transform_vit_timm,
    )

    # Load real image
    image_path = "data/raw/BraTS2021_00495_flair.nii.gz"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}. Skipping.")
        return

    # Load the NIfTI file
    nii_img = nib.load(image_path)
    img_data = nii_img.get_fdata()  # type: ignore

    # Get a middle slice from the 3D volume
    # Shape is typically [H, W, D]. We slice along the depth dimension.
    slice_idx = img_data.shape[2] // 2
    img_2d = img_data[:, :, slice_idx]

    # Resize to 224x224
    img_2d = cv2.resize(img_2d, (224, 224))

    # Normalize for show_cam_on_image [0, 1]
    img_2d_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)

    # Convert to 3 channels for PyTorch [3, H, W]
    img_3d = np.stack([img_2d_norm] * 3, axis=0)

    # Convert to tensor
    input_tensor = torch.tensor(img_3d, dtype=torch.float32).unsqueeze(
        0
    )  # [1, 3, 224, 224]

    # Generate CAM
    cam = cam_extractor(input_tensor)

    print(f"Generated CAM map of shape {cam.shape}. Max: {cam.max()}, Min: {cam.min()}")

    if cam.shape == (224, 224):
        print("SUCCESS! Output shape matches the expected spatial extent (224x224).")
    else:
        print("FAILED! Shape mismatch.")

    # Visual Overlay
    # Original image for show_cam_on_image needs to be [H, W, 3] in [0, 1]
    vis_img = np.stack([img_2d_norm] * 3, axis=-1)

    # Show cam on image
    cam_vis = show_cam_on_image(vis_img, cam, use_rgb=True)

    out_path = "real_brain_heatmap_overlay.jpg"
    # OpenCV expects BGR for saving
    cv2.imwrite(out_path, cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":
    test_vit_cam_real()
