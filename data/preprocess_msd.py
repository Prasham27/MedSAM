import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

from data.common import ROOT, get_logger, load_config, resolve

SIZE = 1024


def normalize_ct(vol, width, level):
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(vol, lo, hi) - lo) / (hi - lo) * 255).round().astype(np.uint8)


def normalize_mri(vol, pclip):
    nz = vol[vol > 0]
    if nz.size == 0:
        return np.zeros(vol.shape, np.uint8)
    lo, hi = np.percentile(nz, pclip)
    return ((np.clip(vol, lo, hi) - lo) / max(hi - lo, 1e-6) * 255).round().astype(np.uint8)


def list_volumes(raw_dir):
    return sorted(p.name[: -len(".nii.gz")] for p in (raw_dir / "imagesTr").glob("*.nii.gz")
                  if not p.name.startswith("._"))


def make_splits(ids, seed=42):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_va = round(n * 0.8), round(n * 0.1)
    return {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}


def process_volume(vid, split, cfg, raw_dir, out_dir):
    img = nib.load(raw_dir / "imagesTr" / f"{vid}.nii.gz")
    lab = nib.load(raw_dir / "labelsTr" / f"{vid}.nii.gz")
    spacing = np.array(img.header.get_zooms()[:3], np.float32)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    seg = np.asanyarray(lab.dataobj)
    if seg.ndim == 4:
        seg = seg[..., 0]

    data = data.astype(np.float32)
    if cfg["modality"].upper() == "CT":
        vol = normalize_ct(data, cfg["window_width"], cfg["window_level"])
    else:
        vol = normalize_mri(data, cfg["percentile_clip"])

    targets = {n: ([v] if isinstance(v, int) else list(v)) for n, v in cfg["labels"].items()}
    masks = {n: np.isin(seg, v) for n, v in targets.items()}
    H, W, D = data.shape[0], data.shape[1], data.shape[2]

    (out_dir / split).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    n_npz = 0
    for z in range(D):
        if not any(masks[n][:, :, z].any() for n in targets):
            continue
        names, gts = [], []
        for n in targets:
            m = cv2.resize(np.ascontiguousarray(masks[n][:, :, z].T, dtype=np.uint8), (SIZE, SIZE),
                           interpolation=cv2.INTER_NEAREST)
            if m.sum() < cfg["min_mask_pixels"]:
                continue
            names.append(n)
            gts.append(m)
        if not names:
            continue
        img_s = cv2.resize(np.ascontiguousarray(vol[:, :, z].T), (SIZE, SIZE),
                           interpolation=cv2.INTER_CUBIC)
        np.savez_compressed(out_dir / split / f"{vid}_z{z:03d}.npz", img=img_s,
                            gts=np.stack(gts).astype(np.uint8), names=np.array(names),
                            spacing=spacing, z=z, orig_shape=np.array([H, W]))
        counts.update(names)
        n_npz += 1

    info = {"spacing": spacing.tolist(), "orig_shape": [H, W], "n_z_total": D}
    return info, n_npz, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    args = ap.parse_args()
    log = get_logger("preprocess_msd", args.task)

    cfg = load_config(args.task)
    raw_dir, out_dir = resolve(cfg["raw_dir"]), resolve(cfg["npz_dir"])
    ids = list_volumes(raw_dir)
    if cfg.get("max_volumes"):
        ids = ids[: cfg["max_volumes"]]
    splits = make_splits(ids)
    log.info(f"{args.task}: {len(ids)} volumes -> " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    split_path = ROOT / "datasets" / "splits.json"
    all_splits = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    all_splits[args.task] = splits
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(all_splits, indent=2), encoding="utf-8")

    meta, stats = {}, []
    for split, vids in splits.items():
        if (out_dir / split).exists():
            for f in (out_dir / split).glob("*.npz"):
                f.unlink()
        n_npz, targets = 0, Counter()
        for vid in tqdm(vids, desc=f"{args.task}/{split}"):
            info, k, counts = process_volume(vid, split, cfg, raw_dir, out_dir)
            meta[vid] = {"split": split, **info}
            n_npz += k
            targets.update(counts)
        log.info(f"{split}: {len(vids)} volumes, {n_npz} npz slices, targets={dict(targets)}")
        stats.append((split, len(vids), n_npz, targets))

    (out_dir / "volumes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stats_path = ROOT / "results" / "tables" / "dataset_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if stats_path.exists():
        with open(stats_path, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["task"] != args.task]
    for split, nv, nn_, targets in stats:
        for t in cfg["labels"]:
            rows.append({"task": args.task, "split": split, "n_volumes": nv, "n_npz_slices": nn_,
                         "target": t, "n_target_slices": targets[t]})
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "split", "n_volumes", "n_npz_slices", "target", "n_target_slices"])
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {out_dir / 'volumes.json'} and {stats_path}")


if __name__ == "__main__":
    main()
