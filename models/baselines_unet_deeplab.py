import numpy as np
import segmentation_models_pytorch as smp
import torch

from data.dataset import box_to_mask_channel


def build_model(name, in_channels=4):
    if name == "unet":
        return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=in_channels, classes=1)
    if name == "deeplab":
        return smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", in_channels=in_channels, classes=1)
    raise ValueError(f"unknown baseline {name}")


def make_input(image, box):
    """image [B,3,H,W] in [0,1], box [B,4] in HxW coords -> [B,4,H,W] (RGB + binary box-mask channel)."""
    _, _, H, W = image.shape
    masks = [np.asarray(box_to_mask_channel(b, H, W), dtype=np.float32).reshape(H, W) for b in box.cpu().numpy()]
    m = torch.from_numpy(np.stack(masks))[:, None].to(image.device)
    return torch.cat([image, m], dim=1)
