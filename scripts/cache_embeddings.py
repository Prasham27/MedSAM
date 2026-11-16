import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from data.common import ROOT, get_logger, load_config, resolve
from models.medsam_wrapper import MedSAM

CKPTS = {"sam": "checkpoints/sam_vit_b_01ec64.pth", "medsam": "checkpoints/medsam_vit_b.pth"}
BATCH = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--model", required=True, choices=["sam", "medsam"])
    args = ap.parse_args()
    log = get_logger("cache_embeddings", args.task)

    cfg = load_config(args.task)
    npz_dir = resolve(cfg["npz_dir"]) / args.split
    out_dir = ROOT / "cache" / args.model / args.task / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(npz_dir.glob("*.npz")) if not (out_dir / f"{p.stem}.npy").exists()]
    log.info(f"{args.model} {args.task}/{args.split}: {len(todo)} to encode")
    if not todo:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedSAM(str(ROOT / CKPTS[args.model])).to(device).eval()
    for i in tqdm(range(0, len(todo), BATCH), desc=f"{args.model} {args.split}"):
        chunk = todo[i:i + BATCH]
        imgs = np.stack([np.load(p)["img"] for p in chunk]).astype(np.float32) / 255.0
        x = torch.from_numpy(np.repeat(imgs[:, None], 3, axis=1)).to(device)
        with torch.no_grad():
            emb = model.encode(x).cpu().numpy().astype(np.float16)
        for p, e in zip(chunk, emb):
            np.save(out_dir / f"{p.stem}.npy", e)
    log.info(f"done: wrote {len(todo)} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
