import os, sys; from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, csv, json, random, time
import numpy as np
import torch
from tqdm import tqdm
from data.common import ROOT, get_logger, load_config
from data.dataset import MSDSliceDataset
from evaluation.metrics import dsc, nsd, reconstruct_volume
from models.medsam_wrapper import MedSAM

SEED = 42
COLS = ["task", "target", "volume", "k", "n_boxes", "dsc3d", "nsd3d", "seconds"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    ap.add_argument("--target", required=True, choices=["liver", "tumor", "prostate"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--ks", default="1,3,5,10")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    log = get_logger("sparse3d_propagation", a.task).info
    log(f"args {vars(a)} device={DEV}")

    cfg = load_config(a.task)
    npz_root = ROOT / cfg["npz_dir"]
    with open(npz_root / "volumes.json", encoding="utf-8") as f:
        volumes = json.load(f)
    cache = ROOT / "cache/medsam" / a.task / a.split
    ds = MSDSliceDataset(npz_dir=str(npz_root), split=a.split, target_name=a.target, jitter=0,
                         image_size=1024, embed_cache_dir=None, out_size=1024)

    per = {}
    for i in tqdm(range(len(ds)), desc="index"):
        it = ds[i]
        per.setdefault(it["volume"], []).append((int(it["z"]), it["name"],
                                                 np.asarray(it["box"], dtype=np.float64),
                                                 np.asarray(it["gt"])[0] > 0.5))

    model = MedSAM(str(ROOT / "checkpoints/medsam_vit_b.pth"), model_type="vit_b").to(DEV).eval()
    rows = []
    for vid in sorted(per):
        items = sorted(per[vid], key=lambda t: t[0])
        zs = np.array([t[0] for t in items], dtype=np.float64)
        boxes = np.stack([t[2] for t in items])
        gts = {t[0]: t[3] for t in items}
        n = len(items)
        for k in ks:
            idx = sorted(set(range(0, n, k)) | {n - 1})
            bi = np.stack([np.interp(zs, zs[idx], boxes[idx, c]) for c in range(4)], axis=1)
            preds = {}
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                for j, (z, name, _, _) in enumerate(items):
                    emb = torch.from_numpy(np.load(cache / f"{name}.npy").astype(np.float32)).to(DEV)[None]
                    box = torch.tensor(bi[j:j + 1], dtype=torch.float32, device=DEV)
                    preds[z] = (model.decode(emb, box)[0, 0] > 0).cpu().numpy()
            sec = time.perf_counter() - t0
            sp = volumes[vid]["spacing"]
            pred3 = reconstruct_volume(preds, volumes[vid])
            gt3 = reconstruct_volume(gts, volumes[vid])
            row = {"task": a.task, "target": a.target, "volume": vid, "k": k, "n_boxes": len(idx),
                   "dsc3d": dsc(pred3, gt3), "nsd3d": nsd(pred3, gt3, [sp[2], sp[1], sp[0]], tol_mm=2.0),
                   "seconds": sec}
            rows.append(row)
            log(f"{vid} k={k} n_boxes={len(idx)} dsc3d {row['dsc3d']:.4f} nsd3d {row['nsd3d']:.4f} {sec:.2f}s")

    out = ROOT / "results/tables/sparse3d.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f) if (r["task"], r["target"]) != (a.task, a.target)]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(keep)
        w.writerows(rows)
    log(f"wrote {len(rows)} rows to {out.name}")


if __name__ == "__main__":
    main()
