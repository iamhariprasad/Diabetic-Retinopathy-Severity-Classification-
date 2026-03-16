import torch
import torch.nn as nn
import timm

class DRVisionTransformer(nn.Module):
    """
    Vision Transformer architecture for Diabetic Retinopathy Classification.
    Uses a 4-channel input: 3 RGB channels + 1 Vessel Mask channel.
    Classification head modified for 5 severe classes (0 to 4).
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=5, pretrained=True, in_chans=4):
        super(DRVisionTransformer, self).__init__()
        
        # Load Timm's ViT. 
        # Setting in_chans=4 automatically replaces the patch_embed layer 
        # to accept 4 channels instead of 3. If pretrained=True, timm will 
        # sum/average or copy weights from the 3-channel version so pretrained weights 
        # are preserved where possible.
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=in_chans
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Tensor of shape (B, 4, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 5)
        """
        return self.model(x)

    def get_attention_maps(self, x):
        """
        Helper to fetch attention maps if needed for interpretability.
        Note: True Grad-CAM computation requires hooks. This is a placeholder
        for fetching internal features if the user implements a custom attention 
        roll-out or calls hooks externally.
        """
        # We can use timm's `forward_features` to get the tokens before classification head
        features = self.model.forward_features(x)
        return features

def build_model(num_classes=5, in_chans=4, device='cuda'):
    """
    Instantiates the model and moves it to the target device.
    """
    model = DRVisionTransformer(num_classes=num_classes, in_chans=in_chans)
    model = model.to(device)
    return model
