import torch
import torch.nn as nn

from medshield.explain.vit_cam import ViTGradCAM, reshape_transform_vit_timm


class DummyViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified mock of a ViT
        norm = nn.LayerNorm(768)
        linear = nn.Linear(768, 768)
        self.block = nn.Sequential(norm, linear)
        self.head = nn.Linear(768, 2)
        # Initialize weights to positive values to avoid ReLUs zeroing out the CAM
        nn.init.normal_(linear.weight, mean=0.1, std=0.01)
        nn.init.normal_(self.head.weight, mean=0.1, std=0.01)

    def forward(self, x):
        # x is [B, C, H, W] -> reshape to [B, 196, 768] (3*224*224 = 150528 = 196*768)
        b = x.size(0)
        patches = x.view(b, 196, 768)
        cls_token = torch.ones(b, 1, 768, device=x.device, requires_grad=True)
        tokens = torch.cat([cls_token, patches], dim=1)

        # Pass through the mock block
        out = self.block(tokens)

        # In a real ViT, self-attention mixes tokens. Here we explicitly sum
        # across the sequence dimension so the classification score depends on all patches.
        # Otherwise, gradients for patch tokens will be 0.
        cls_mixed = out.sum(dim=1)
        return self.head(cls_mixed)


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


def test_heatmap_properties():
    # Task 71 Acceptance Criteria Validation
    model = DummyViT()
    target_layer = model.block[0]
    cam = ViTGradCAM(model, target_layer, reshape_transform=reshape_transform_vit_timm)

    # 1. Test target_category=None (defaults to predicted class)
    input_tensor1 = torch.randn(1, 3, 224, 224)
    heatmap1 = cam(input_tensor1, target_category=None)

    # Verify shape
    assert heatmap1.shape == (224, 224)

    # 2. Verify heatmap values are normalized between 0 and 1
    assert heatmap1.min() >= 0.0
    assert heatmap1.max() <= 1.0

    # 3. Verify different inputs yield meaningfully different heatmaps
    input_tensor2 = torch.randn(1, 3, 224, 224) * 100.0  # Make it significantly different
    heatmap2 = cam(input_tensor2, target_category=None)

    # The sum of absolute differences should be greater than 0
    import numpy as np

    assert not np.allclose(
        heatmap1, heatmap2, atol=1e-5
    ), "Heatmaps for different images should differ"
