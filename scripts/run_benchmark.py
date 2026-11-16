import os, sys; from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, csv, json, random, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.common import ROOT, get_logger, load_config
from data.dataset import MSDSliceDataset
from evaluation.metrics import dsc, nsd, postprocess_3d, reconstruct_volume
from models.baselines_unet_deeplab import build_model, make_input
from models.lora_medsam import apply_lora, load_lora
from models.medsam_wrapper import MedSAM

SEED = 42
COLS = ["task", "target", "method", "tag", "jitter", "volume", "dsc3d", "nsd3d", "dsc3d_pp", "nsd3d_pp",
        "slice_dsc_mean", "n_slices", "sec_per_slice"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"
CACHED = {"sam", "medsam", "medsam_point"}


def load_model(a):
    if a.method in ("unet", "deeplab"):
        net = build_model(a.method, in_channels=4).to(DEV)
        ck = a.ckpt or ROOT / "checkpoints" / f"{a.method}_{a.task}_{a.target}.pth"
        net.load_state_dict(torch.load(ck, map_location=DEV))
        return None, net.eval()
    base = "checkpoints/sam_vit_b_01ec64.pth" if a.method == "sam" else "checkpoints/medsam_vit_b.pth"
    model = MedSAM(str(ROOT / base), model_type="vit_b").to(DEV)
    if a.method == "lora":
        apply_lora(model.sam, rank=a.rank, use_checkpoint=False)
        load_lora(model.sam, str(a.ckpt or ROOT / "checkpoints" / f"lora_{a.task}_{a.target}.pth"))
        model.to(DEV)
    return model.eval(), None


def load_gt(path, target):
    with np.load(path) as f:
        names = [str(n) for n in f["names"]]
        return f["gts"][names.index(target)] > 0


def infer(a, model, net, batch, cache, misses):
    img, box = batch["image"], batch["box"]
    if net is not None:
        x = make_input(img, box).to(DEV)
        return F.interpolate(net(x), size=(1024, 1024), mode="bilinear", align_corners=False)
    p = cache / f"{batch['name'][0]}.npy" if cache is not None else None
    if p is not None and p.exists():
        emb = torch.from_numpy(np.load(p).astype(np.float32)).to(DEV)[None]
    else:
        if p is not None:
            misses[0] += 1
        emb = model.encode(img.to(DEV))
    if a.method == "medsam_point":
        b = box[0]
        pt = torch.stack([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])[None].to(DEV)
        return model.point_decode(emb, pt, torch.ones(1, dtype=torch.long, device=DEV))
    return model.decode(emb, box.to(DEV))


def save_pred(a, name, z, vid, pred, box):
    d = ROOT / "results/preds" / a.task / a.target / a.method
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{name}.png"), cv2.resize(pred.astype(np.uint8) * 255, (256, 256),
                                                   interpolation=cv2.INTER_NEAREST))
    with open(d / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"volume": vid, "z": z, "box": box.tolist()}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    ap.add_argument("--target", required=True, choices=["liver", "tumor", "prostate"])
    ap.add_argument("--method", required=True,
                    choices=["sam", "medsam", "lora", "unet", "deeplab", "medsam_point"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_preds", action="store_true")
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    log = get_logger("run_benchmark", a.task).info
    log(f"args {vars(a)} device={DEV}")

    cfg = load_config(a.task)
    npz_root = ROOT / cfg["npz_dir"]
    with open(npz_root / "volumes.json", encoding="utf-8") as f:
        volumes = json.load(f)
    baseline = a.method in ("unet", "deeplab")
    jitter = 0 if a.method == "medsam_point" else a.jitter
    ds = MSDSliceDataset(npz_dir=str(npz_root), split=a.split, target_name=a.target, jitter=jitter,
                         image_size=1024, embed_cache_dir=None, out_size=256 if baseline else 1024)
    model, net = load_model(a)
    cache = None
    if a.method in CACHED:
        cache = ROOT / "cache" / ("sam" if a.method == "sam" else "medsam") / a.task / a.split

    P, G, BX, SD, T, misses = {}, {}, {}, {}, {}, [0]
    for batch in tqdm(DataLoader(ds, batch_size=1, num_workers=0), desc=f"{a.method} {a.task}/{a.target}"):
        name, vid, z = batch["name"][0], batch["volume"][0], int(batch["z"][0])
        box1024 = batch["box"][0].numpy().astype(np.float64) * (4.0 if baseline else 1.0)
        gt = load_gt(npz_root / a.split / f"{name}.npz", a.target)
        if AMP:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = (infer(a, model, net, batch, cache, misses)[0, 0] > 0).cpu().numpy()
        T.setdefault(vid, []).append(time.perf_counter() - t0)
        P.setdefault(vid, {})[z] = pred
        G.setdefault(vid, {})[z] = gt
        BX.setdefault(vid, {})[z] = box1024
        SD.setdefault(vid, []).append(dsc(pred, gt))
        if a.save_preds:
            save_pred(a, name, z, vid, pred, box1024)

    rows = []
    for vid in sorted(P):
        vj = volumes[vid]
        H, W = vj["orig_shape"]
        sx, sy = W / 1024, H / 1024
        sp = vj["spacing"]
        zyx = [sp[2], sp[1], sp[0]]
        pred3 = reconstruct_volume(P[vid], vj)
        gt3 = reconstruct_volume(G[vid], vj)
        boxes = {z: b * np.array([sx, sy, sx, sy]) for z, b in BX[vid].items()}
        pp = postprocess_3d(pred3, boxes)
        rows.append({"task": a.task, "target": a.target, "method": a.method, "tag": a.tag, "jitter": a.jitter,
                     "volume": vid, "dsc3d": dsc(pred3, gt3), "nsd3d": nsd(pred3, gt3, zyx, tol_mm=2.0),
                     "dsc3d_pp": dsc(pp, gt3), "nsd3d_pp": nsd(pp, gt3, zyx, tol_mm=2.0),
                     "slice_dsc_mean": float(np.mean(SD[vid])), "n_slices": len(SD[vid]),
                     "sec_per_slice": float(np.sum(T[vid]) / len(T[vid]))})
        log(f"{vid}: dsc3d {rows[-1]['dsc3d']:.4f} nsd3d {rows[-1]['nsd3d']:.4f} "
            f"dsc3d_pp {rows[-1]['dsc3d_pp']:.4f} slices {len(SD[vid])}")
    if cache is not None:
        log(f"cache misses (encoded on the fly): {misses[0]}")

    out = ROOT / "results/tables/benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    key = (a.task, a.target, a.method, a.tag, str(a.jitter))
    keep = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f)
                    if (r["task"], r["target"], r["method"], r["tag"], r["jitter"]) != key]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(keep)
        w.writerows(rows)
    log(f"wrote {len(rows)} rows to {out.name} (method={a.method}, tag='{a.tag}', jitter={a.jitter})")


if __name__ == "__main__":
    main()
