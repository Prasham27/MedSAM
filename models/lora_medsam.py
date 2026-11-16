import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

LORA_KEYS = {"Aq", "Bq", "Av", "Bv"}


class LoRAQKV(nn.Module):
    def __init__(self, qkv, rank=4, alpha=4):
        super().__init__()
        self.qkv = qkv
        for p in self.qkv.parameters():
            p.requires_grad = False
        dim = qkv.in_features
        self.dim = qkv.out_features // 3
        self.scale = alpha / rank
        self.Aq = nn.Parameter(torch.empty(dim, rank))
        self.Bq = nn.Parameter(torch.zeros(rank, dim))
        self.Av = nn.Parameter(torch.empty(dim, rank))
        self.Bv = nn.Parameter(torch.zeros(rank, dim))
        for A in (self.Aq, self.Av):
            nn.init.kaiming_uniform_(A.t(), a=math.sqrt(5))

    def forward(self, x):
        d = self.dim
        base = self.qkv(x)
        lq = (x @ self.Aq @ self.Bq) * self.scale
        lv = (x @ self.Av @ self.Bv) * self.scale
        return torch.cat([base[..., :d] + lq, base[..., d:2 * d], base[..., 2 * d:] + lv], dim=-1)


def _checkpointed(blk):
    orig = blk.forward

    def forward(x):
        if blk.training:
            return checkpoint(orig, x, use_reentrant=False)
        return orig(x)

    blk.forward = forward


def apply_lora(sam, rank=4, alpha=4, use_checkpoint=True):
    for p in sam.parameters():
        p.requires_grad = False
    for blk in sam.image_encoder.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, rank, alpha)
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
    if use_checkpoint:
        for blk in sam.image_encoder.blocks:
            _checkpointed(blk)
    count_params(sam)
    return sam


def lora_state_dict(sam):
    return {k: v.detach().cpu() for k, v in sam.state_dict().items()
            if k.split(".")[-1] in LORA_KEYS or "mask_decoder." in k}


def load_lora(sam, path):
    sd = torch.load(path, map_location="cpu")
    result = sam.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys in {path}: {result.unexpected_keys[:5]}")
    print(f"loaded {len(sd)} tensors from {path}")
    return sam


def count_params(sam):
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam.parameters())
    print(f"trainable params: {trainable:,} / total: {total:,}")
    return trainable, total
