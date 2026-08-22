import numpy as np
import pytest

from medshield.explain.vit_cam import show_cam_on_image


def test_overlay_shape_and_type():
    # Mock image (H, W, 3) in [0, 1]
    img = np.random.rand(224, 224, 3).astype(np.float32)
    # Mock heatmap (H, W) in [0, 1]
    heatmap = np.random.rand(224, 224).astype(np.float32)

    overlay = show_cam_on_image(img, heatmap)

    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_overlay_grayscale_image():
    # Mock grayscale image (H, W) in [0, 1]
    img = np.random.rand(224, 224).astype(np.float32)
    heatmap = np.random.rand(224, 224).astype(np.float32)

    overlay = show_cam_on_image(img, heatmap)

    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_overlay_invalid_image_range():
    # Mock image with values > 1.0
    img = np.random.rand(224, 224, 3).astype(np.float32) * 255.0
    heatmap = np.random.rand(224, 224).astype(np.float32)

    with pytest.raises(Exception, match="The input image should np.float32 in the range"):
        show_cam_on_image(img, heatmap)
