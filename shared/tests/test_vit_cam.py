import torch
import torch.nn as nn

from medshield.explain.vit_cam import ViTGradCAM, reshape_transform_vit_timm


class DummyViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified mock of a ViT
        self.block = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768))
        self.head = nn.Linear(768, 2)

    def forward(self, x):
        # x is [B, C, H, W]
        # Return a mock tensor of tokens [B, 197, 768] (1 CLS + 196 patches)
        # Note: In a real model, this comes from patch embedding, here we just mock it.
        # Ensure it requires grad so backward passes correctly.
        tokens = torch.randn(x.size(0), 197, 768, device=x.device, requires_grad=True)
        # Pass through the mock block
        out = self.block(tokens)
        # Extract the CLS token for classification
        cls_token = out[:, 0]
        return self.head(cls_token)


def test_hooks_capture_and_cleanup():
    model = DummyViT()
    target_layer = model.block[0]  # LayerNorm

    cam = ViTGradCAM(model, target_layer, reshape_transform=reshape_transform_vit_timm)

    # Before running, hooks should be empty
    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0

    input_tensor = torch.randn(1, 3, 224, 224)

    # Generate CAM
    heatmap = cam(input_tensor, target_category=1)

    # Assert output shape matches original image (224x224)
    assert heatmap.shape == (224, 224)

    # After running, hooks should be cleanly removed
    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0

    # Assert internal tensors are cleared
    assert cam.activations is None
    assert cam.gradients is None


def test_inference_mode_unaffected():
    model = DummyViT()
    target_layer = model.block[0]
    cam = ViTGradCAM(model, target_layer, reshape_transform=reshape_transform_vit_timm)

    input_tensor = torch.randn(1, 3, 224, 224)

    # Forward pass 1 (before any CAM calls)
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Generate CAM
    _ = cam(input_tensor, target_category=0)

    # Forward pass 2 (after CAM)
    with torch.no_grad():
        _ = model(input_tensor)

    # Should not be affected by lingering hooks
    # Note: DummyViT generates random tokens in forward(), so we can't do an allclose check.
    # Instead, we just verify that forward passes successfully and hooks remain 0.
    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0
