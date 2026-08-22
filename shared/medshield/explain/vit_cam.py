import torch
import torch.nn.functional as F  # noqa: N812


class ViTGradCAM:
    """
    Grad-CAM for Vision Transformers.
    Extracts gradients and activations from the final Transformer block
    to compute a saliency map.
    """

    def __init__(self, model, target_layer, reshape_transform=None):
        """
        Args:
            model: The ViT model (e.g., TumorClassifier)
            target_layer: The target module (usually the LayerNorm before MLP in the last block)
            reshape_transform: A function to reshape the tokens back to a 2D spatial map.
                               For a standard ViT (14x14 tokens + CLS), this drops the CLS token
                               and reshapes the 196 tokens into 14x14.
        """
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

        self.activations = None
        self.gradients = None
        self.handlers = []

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handlers:
            h.remove()
        self.handlers = []

    def __call__(self, input_tensor, target_category=None):
        """
        Generates the CAM for a given input tensor and target category.
        Args:
            input_tensor (torch.Tensor): Input batch of shape (1, C, H, W)
            target_category (int, optional): The class to generate CAM for. If None, uses the highest scoring class.
        Returns:
            np.ndarray: Upsampled spatial map of shape (H, W) normalized to [0, 1].
        """
        self._register_hooks()
        try:
            self.model.eval()

            # We need gradients on the input (not strictly required for Grad-CAM on intermediate layers,
            # but required for the backward pass to run)
            if not input_tensor.requires_grad:
                input_tensor.requires_grad = True

            # Forward pass
            logits = self.model(input_tensor)

            if target_category is None:
                target_category = logits.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            score = logits[0, target_category]
            score.backward()

            # Get activations and gradients
            acts = self.activations
            grads = self.gradients

            if acts is None or grads is None:
                raise RuntimeError(
                    "Activations or gradients are None. The target layer might not have been executed, or hooks were not called."
                )

            if self.reshape_transform is not None:
                acts = self.reshape_transform(acts)
                grads = self.reshape_transform(grads)

            # acts and grads are now [batch, channels, height, width]
            # Global Average Pooling on gradients
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)

            # Weighted combination of forward activation maps
            cam = torch.sum(weights * acts, dim=1).squeeze(0)

            # ReLU to keep only positive influences
            cam = F.relu(cam)

            # Normalize between 0 and 1
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 0:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            cam = cam.cpu().numpy()

            # Upsample to input spatial resolution
            input_h, input_w = input_tensor.shape[2:]
            cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
            cam = F.interpolate(cam, size=(input_h, input_w), mode="bilinear", align_corners=False)
            cam = cam.squeeze().numpy()

            return cam
        finally:
            self.remove_hooks()
            self.activations = None
            self.gradients = None


def reshape_transform_vit_timm(tensor, height=14, width=14):
    """
    Reshape transform for timm's ViT.
    The input tensor has shape [batch, num_tokens, hidden_dim].
    For ViT-Base/16 with 224x224 images, num_tokens = 1 + (224/16)*(224/16) = 1 + 196 = 197.
    """
    # Exclude the CLS token (the first token)
    result = tensor[:, 1:, :]
    # Reshape from [batch, sequence_length, hidden_dim] to [batch, hidden_dim, height, width]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    # Transpose to [batch, hidden_dim, height, width]
    result = result.permute(0, 3, 1, 2)
    return result
