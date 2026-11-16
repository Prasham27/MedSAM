import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.common import resolve


def get_bbox(mask, jitter=0, rng=None):
    """Tight xyxy box (inclusive pixel indices) of a non-empty 2D mask, each side jittered by U[0, jitter]."""
    ys, xs = np.nonzero(mask)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    if jitter > 0:
        rng = rng if rng is not None else np.random.default_rng(42)
        box += np.array([-1, -1, 1, 1], np.float32) * rng.uniform(0, jitter, 4).astype(np.float32)
    H, W = mask.shape
    box[[0, 2]] = np.clip(box[[0, 2]], 0, W - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, H - 1)
    return box


def box_to_mask_channel(box, H, W):
    """Binary float32 [H,W] mask, 1 inside the xyxy box (inclusive)."""
    x0, y0, x1, y1 = box
    x0, y0 = max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0)
    x1, y1 = min(int(np.ceil(x1)), W - 1), min(int(np.ceil(y1)), H - 1)
    m = np.zeros((H, W), np.float32)
    m[y0:y1 + 1, x0:x1 + 1] = 1
    return m


class MSDSliceDataset(Dataset):
    def __init__(self, npz_dir, split, target_name, jitter=20, image_size=1024, embed_cache_dir=None, out_size=1024):
        self.npz_dir = resolve(npz_dir)
        self.split, self.target_name = split, target_name
        self.jitter, self.image_size, self.out_size = jitter, image_size, out_size
        self.embed_cache_dir = resolve(embed_cache_dir) if embed_cache_dir else None
        self.rng = np.random.default_rng(42)
        self.items = [(rel, names.index(target_name)) for rel, names in self._index()
                      if target_name in names]

    def _index(self):
        cache = self.npz_dir / f"{self.split}_index.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        index = []
        for f in sorted((self.npz_dir / self.split).glob("*.npz")):
            with np.load(f) as d:
                index.append([f.relative_to(self.npz_dir).as_posix(), [str(n) for n in d["names"]]])
        cache.write_text(json.dumps(index), encoding="utf-8")
        return index

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, t = self.items[i]
        path = self.npz_dir / rel
        with np.load(path) as d:
            img, gt, z = d["img"], d["gts"][t], int(d["z"])
        name = path.stem

        box = get_bbox(gt, self.jitter, self.rng) * (self.out_size / self.image_size)

        if self.out_size != self.image_size:
            size = (self.out_size, self.out_size)
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
        image = np.repeat(img[None].astype(np.float32) / 255.0, 3, axis=0)
        gt = (gt > 0).astype(np.float32)[None]

        out = {
            "image": torch.from_numpy(image),
            "gt": torch.from_numpy(gt),
            "box": torch.from_numpy(box.astype(np.float32)),
            "name": name,
            "volume": name.rsplit("_z", 1)[0],
            "z": z,
        }
        if self.embed_cache_dir is not None:
            emb = np.load(self.embed_cache_dir / f"{name}.npy").astype(np.float32)
            out["embedding"] = torch.from_numpy(emb)
        return out
