import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry


class MedSAM(nn.Module):
    def __init__(self, checkpoint, model_type="vit_b", freeze_encoder=True, freeze_prompt=True):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        if freeze_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt:
            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad = False

    def encode(self, image):
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16, enabled=image.is_cuda):
            emb = self.sam.image_encoder(image)
        return emb.float()

    def _mask_decode(self, embedding, sp, dn):
        low, _ = self.sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dn,
            multimask_output=False,
        )
        return F.interpolate(low, size=(1024, 1024), mode="bilinear", align_corners=False)

    def decode(self, embedding, boxes):
        sp, dn = self.sam.prompt_encoder(points=None, boxes=boxes[:, None, :], masks=None)
        return self._mask_decode(embedding, sp, dn)

    def forward(self, image, boxes):
        return self.decode(self.encode(image), boxes)

    def point_decode(self, embedding, points, labels):
        sp, dn = self.sam.prompt_encoder(points=(points[:, None, :], labels[:, None]), boxes=None, masks=None)
        return self._mask_decode(embedding, sp, dn)
