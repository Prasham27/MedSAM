import os, sys; from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.common import ROOT, get_logger, load_config
from data.dataset import MSDSliceDataset
from models.baselines_unet_deeplab import build_model, make_input
from evaluation.losses import bce_dice_loss, slice_dice

SEED = 42
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "deeplab"])
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    ap.add_argument("--target", required=True, choices=["liver", "tumor", "prostate"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    args = ap.parse_args()
    epochs = args.epochs or (25 if args.task == "prostate" else 15)
    cfg = load_config(args.task)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    log = get_logger("train_baseline", args.task).info
    tag = f"{args.model}/{args.task}/{args.target}"
    log(f"args {vars(args)} epochs={epochs} device={DEV}")

    def ds(split, jitter):
        return MSDSliceDataset(npz_dir=str(ROOT / cfg["npz_dir"]), split=split, target_name=args.target,
                               jitter=jitter, image_size=1024, embed_cache_dir=None, out_size=256)
    tr = DataLoader(ds("train", 20), batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=AMP,
                    generator=torch.Generator().manual_seed(SEED))
    va = DataLoader(ds("val", 0), batch_size=args.bs, shuffle=False, num_workers=4)

    net = build_model(args.model, in_channels=4).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    n_total = sum(p.numel() for p in net.parameters())
    ckpt = ROOT / "checkpoints" / f"{args.model}_{args.task}_{args.target}.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if AMP:
        torch.cuda.reset_peak_memory_stats()
    best, best_ep, hist, train_s = -1.0, 0, [], 0.0
    for ep in range(1, epochs + 1):
        net.train(); losses = []; t0 = time.time()
        for b in tqdm(tr, desc=f"{tag} ep{ep}", leave=False):
            x = make_input(b["image"], b["box"]).to(DEV, non_blocking=True)
            y = b["gt"].to(DEV, non_blocking=True)
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
                logits = net(x)
            loss = bce_dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        train_s += time.time() - t0

        net.eval(); dscs = []
        with torch.inference_mode(), torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            for b in va:
                logits = net(make_input(b["image"], b["box"]).to(DEV))
                dscs += slice_dice(logits, b["gt"].to(DEV))
        val = float(np.mean(dscs))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_dsc": val})
        log(f"{tag} epoch {ep} loss {np.mean(losses):.4f} val_dsc {val:.4f}")
        if val > best:
            best, best_ep = val, ep
            torch.save(net.state_dict(), ckpt)

    peak = torch.cuda.max_memory_allocated() / 2**20 if AMP else 0.0
    res = {"task": args.task, "target": args.target, "model": args.model, "epochs": epochs, "bs": args.bs,
           "lr": args.lr, "wd": args.wd, "best_val_dsc": best, "best_epoch": best_ep, "history": hist,
           "train_seconds": train_s, "peak_vram_mb": peak, "trainable_params": n_total,
           "total_params": n_total, "checkpoint": str(ckpt.relative_to(ROOT))}
    out = ROOT / "results/tables" / f"train_{args.model}_{args.task}_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log(f"{tag} done: best val_dsc {best:.4f} @ep{best_ep}, {train_s:.0f}s, peak {peak:.0f} MB")


if __name__ == "__main__":
    main()
