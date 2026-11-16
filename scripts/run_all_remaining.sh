#!/usr/bin/env bash
# Master chain for everything after prostate Phase 2. Idempotent-ish: each step logs; rerun from the failing step by hand if needed.
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")/.."
step(){ echo "=== $(date '+%H:%M:%S') STEP: $*"; }
step "prostate LoRA (150 opt steps, <=3 epochs)"
python scripts/train_lora.py --task prostate --target prostate --max_steps 150 --epochs 3
python scripts/run_benchmark.py --task prostate --target prostate --method lora --ckpt checkpoints/lora_prostate_prostate.pth --save_preds
step "liver embeddings val/test"
for m in medsam sam; do for s in val test; do python scripts/cache_embeddings.py --task liver --split $s --model $m; done; done
step "liver zero-shot"
for t in liver tumor; do for m in medsam sam; do python scripts/run_benchmark.py --task liver --target $t --method $m --save_preds; done; python scripts/run_benchmark.py --task liver --target $t --method medsam_point; done
step "liver jitter"
for t in liver tumor; do for j in 5 10 20 40; do for m in medsam sam; do python scripts/run_benchmark.py --task liver --target $t --method $m --jitter $j --tag jitter; done; done; done
step "liver sparse3d"
for t in liver tumor; do python evaluation/sparse_3d_propagation.py --task liver --target $t; done
step "liver baselines (6 epochs)"
for t in liver tumor; do for b in unet deeplab; do
  python scripts/train_baseline.py --task liver --target $t --model $b --epochs 6
  python scripts/run_benchmark.py --task liver --target $t --method $b --ckpt checkpoints/${b}_liver_${t}.pth --save_preds
done; done
step "liver LoRA (150 opt steps each)"
for t in tumor liver; do
  python scripts/train_lora.py --task liver --target $t --max_steps 150 --epochs 1
  python scripts/run_benchmark.py --task liver --target $t --method lora --ckpt checkpoints/lora_liver_${t}.pth --save_preds
done
step "phase 4 figures + report"
python evaluation/visualize_podium_overlays.py
python scripts/make_report.py
step "ALL DONE"
