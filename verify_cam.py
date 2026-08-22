import torch
import numpy as np
from shared.medshield.model.vit import TumorClassifier
from shared.medshield.explain.vit_cam import ViTGradCAM, reshape_transform_vit_timm
import os


def test_vit_cam():
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

    # Load synthetic image
    image_path = "data/synthetic_baseline/7_slice5.npy"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}. Skipping.")
        return

    # Image is [4, 240, 240] or similar. We need [1, 3, 224, 224] for standard ViT.
    img_array = np.load(image_path)

    # Simple preprocessing: take 1st channel, resize to 224x224, repeat to 3 channels
    img_2d = img_array[0] if img_array.ndim > 2 else img_array

    # Just center crop/resize to 224 for a quick test
    img_2d = img_2d[:224, :224]

    # Convert to 3 channels
    img_3d = np.stack([img_2d] * 3, axis=0)

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


if __name__ == "__main__":
    test_vit_cam()
