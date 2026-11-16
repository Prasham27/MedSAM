import os, sys; from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, gc, json, math, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from data.common import ROOT, get_logger, load_config
from data.dataset import MSDSliceDataset
from models.medsam_wrapper import MedSAM
from models.lora_medsam import apply_lora, count_params, lora_state_dict
from evaluation.losses import bce_dice_loss, slice_dice

SEED, WARMUP, FALLBACK_STEPS = 42, 50, 5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEV.type == "cuda"


class OOMFallback(Exception):
    pass


def make_ds(cfg, task, split, target, mode, jitter):
    cache = ROOT / "cache/medsam" / task / split if mode == "decoder_only" else None
    return MSDSliceDataset(npz_dir=str(ROOT / cfg["npz_dir"]), split=split, target_name=target, jitter=jitter,
                           image_size=1024, embed_cache_dir=str(cache) if cache else None, out_size=1024)


def run(args, cfg, mode, log, max_steps, fallback=None):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    tag = f"{args.task}/{args.target}/{mode}"
    ckpt = ROOT / "checkpoints" / f"lora_{args.task}_{args.target}.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    model = MedSAM(str(ROOT / "checkpoints/medsam_vit_b.pth"), model_type="vit_b").to(DEV)
    apply_lora(model.sam, rank=args.rank, use_checkpoint=(mode == "lora")); model.to(DEV)
    if mode == "decoder_only":
        for p in model.sam.image_encoder.parameters():
            p.requires_grad_(False)
    trainable, n_total = count_params(model.sam)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    tr_ds = make_ds(cfg, args.task, "train", args.target, mode, jitter=20)
    va_ds = make_ds(cfg, args.task, "val", args.target, mode, jitter=0)
    tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=AMP,
                    generator=torch.Generator().manual_seed(SEED))
    va = DataLoader(Subset(va_ds, range(min(200, len(va_ds)))), batch_size=args.bs, num_workers=4)

    planned = min(max_steps, args.epochs * math.ceil(len(tr) / args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARMUP if s < WARMUP
        else 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, planned - WARMUP))))

    def forward(batch):
        img = batch["image"].to(DEV, non_blocking=True)
        box = batch["box"].to(DEV)
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=AMP):
            if mode == "lora":
                logits = model(img, box)
            else:
                logits = model.decode(batch["embedding"].to(DEV), box)
        return logits.float()

    if AMP:
        torch.cuda.reset_peak_memory_stats()
    best, best_ep, hist, gstep, train_s, done = -1.0, 0, [], 0, 0.0, False
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n_loss, t0 = 0.0, 0, time.time()
        for i, batch in enumerate(tqdm(tr, desc=f"{tag} ep{ep}", leave=False)):
            try:
                loss = bce_dice_loss(forward(batch), batch["gt"].to(DEV))
                (loss / args.accum).backward()
            except torch.cuda.OutOfMemoryError:
                if mode == "lora" and gstep < FALLBACK_STEPS:
                    raise OOMFallback() from None
                raise
            run_loss += loss.item(); n_loss += 1
            if (i + 1) % args.accum == 0 or i + 1 == len(tr):
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 20 == 0:
                    log(f"{tag} step {gstep}/{planned} ep {ep} loss {run_loss / n_loss:.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e}")
                    run_loss, n_loss = 0.0, 0
                if gstep >= planned:
                    done = True
                    break
        train_s += time.time() - t0

        model.eval(); dscs = []
        with torch.inference_mode():
            for batch in va:
                dscs += slice_dice(forward(batch), batch["gt"].to(DEV))
        val = float(np.mean(dscs))
        hist.append({"epoch": ep, "steps": gstep, "val_dsc": val})
        log(f"{tag} epoch {ep} val_dsc {val:.4f} (n={len(dscs)})")
        if val > best:
            best, best_ep = val, ep
            torch.save(lora_state_dict(model.sam), ckpt)
        if done:
            break

    peak = torch.cuda.max_memory_allocated() / 2**20 if AMP else 0.0
    res = {"task": args.task, "target": args.target,
           "mode": "DECODER_ONLY" if mode == "decoder_only" else "LORA", "fallback": fallback,
           "rank": args.rank, "epochs": args.epochs, "lr": args.lr, "wd": args.wd, "accum": args.accum,
           "bs": args.bs, "max_steps": planned, "optimizer_steps": gstep, "best_val_dsc": best,
           "best_epoch": best_ep, "history": hist, "train_seconds": train_s,
           "steps_per_sec": gstep / train_s if train_s > 0 else 0.0, "peak_vram_mb": peak,
           "trainable_params": trainable, "total_params": n_total,
           "checkpoint": str(ckpt.relative_to(ROOT))}
    out = ROOT / "results/tables" / f"train_lora_{args.task}_{args.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log(f"{tag} done: best val_dsc {best:.4f} @ep{best_ep}, {gstep} opt steps, "
        f"{res['steps_per_sec']:.3f} steps/s, peak {peak:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    ap.add_argument("--target", required=True, choices=["liver", "tumor", "prostate"])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--decoder_only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.task)
    max_steps = args.max_steps or (600 if args.task == "prostate" else 1500)
    log = get_logger("train_lora", args.task).info
    log(f"args {vars(args)} device={DEV} max_steps={max_steps}")

    fallback = None
    try:
        run(args, cfg, "decoder_only" if args.decoder_only else "lora", log, max_steps)
    except OOMFallback:
        fallback = "DECODER_ONLY"
    if fallback:
        gc.collect()
        torch.cuda.empty_cache()
        log(f"CUDA OOM in first {FALLBACK_STEPS} steps; restarting as DECODER_ONLY")
        run(args, cfg, "decoder_only", log, max_steps, fallback)


if __name__ == "__main__":
    main()
