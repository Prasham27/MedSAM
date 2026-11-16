# HANDOFF — MedSAM MSD Evaluation (Group 5, DAU CV Project)

This file is the single source of truth for *any* Claude Code session (or human) picking up this
project. Read it fully before doing anything. Keep it updated after every meaningful step.

## 0. How to resume (for a new Claude Code session)
1. Open this folder (`medsam-msd-evaluation/`) as the working directory.
2. Read `HANDOFF.md` (this file) and `SPEC.md` (exact technical spec for every module).
3. Check the **Status board** below. Find the first item not marked `[x]` and continue from there.
4. Check `results/logs/` for the latest log of any background job; check `nvidia-smi` for running jobs.
5. Use Haiku subagents for implementation, Sonnet/Opus only for debugging decisions. The owner's
   Claude usage budget is very limited — be economical (no exploratory file dumps, no re-reading).
6. After every completed step: tick the box, append a line to the **Run log**, and if a decision
   was changed, record it in **Decisions**.

## 1. Project context (from mid-evaluation)
- Course: Computer Vision project, Dhirubhai Ambani University. Group 5: Dhruv Patel (202301095),
  Kavish Patel (202301074), Prasham Shah (202301252).
- Paper: Jun Ma et al., "Segment anything in medical images", Nat. Commun. 15:654 (2024).
- Mid-eval defined the domain-shift problem and chose MSD `Task03_Liver` (CT, labels 1=liver,
  2=tumor) and `Task05_Prostate` (T2 MRI, labels 1=PZ, 2=TZ -> merged to whole prostate).
- Full original plan: `../medsam_final_implementation_plan.md` (one level above this folder).

## 2. Hardware / environment actually available (measured 2026-11-16)
- Windows 11, Python 3.13.9, conda 26.1.1, git available.
- GPU: NVIDIA RTX 3060 Laptop, **6 GB VRAM** (driver 546.30, CUDA 12.3). RAM 16 GB. Disk ~57 GB free.
- torch was CPU-only (2.12.1+cpu) -> must install CUDA build (see SPEC §1).
- Env var `KMP_DUPLICATE_LIB_OK=TRUE` is required to run python+torch on this machine (OpenMP clash).

## 3. Scope decisions (the plan listed 4 "better approaches"; we implement ONE + one cheap add-on)
Chosen as the "better approach": **Approach A — LoRA-MedSAM** (rank 4 LoRA in ViT-B Q/V + trainable
mask decoder). Rationale: it is the only extension that (a) is a genuine methodological improvement
over the paper, (b) is trainable on 6 GB VRAM (bf16 + grad-checkpointing + batch 1 + accumulation),
(c) needs no new external repo/checkpoint, (d) produces a direct apples-to-apples DSC/NSD comparison
with zero-shot MedSAM and the specialist baselines.
Cheap add-on if time allows: **Approach C — YOLOv8n auto-prompting** (ultralytics already installed,
YOLOv8n trains in minutes; shows the fully-automatic pipeline).
Dropped: **Approach B (MedSAM-2)** — needs SAM2/Hiera install + memory bank; too much VRAM/setup for
the budget. **Approach D (LiteMedSAM)** — pure latency benchmark, little scientific value here.
Fallback for LoRA if OOM on 6 GB: freeze encoder entirely, fine-tune mask decoder only on cached
embeddings ("MedSAM-DecFT"); document this honestly in the report.

Data scope (disk/time): Liver uses a **subset of 40 volumes** (deterministic: first 40 sorted
filenames) -> 32 train / 4 val / 4 test volumes. Prostate uses all 32 -> 26/3/3. Split is at the
3D-volume level, seed 42, saved to `datasets/splits.json`.

## 4. Status board  (tick when done; keep in order)
### Phase 0 — Environment & assets
- [x] 0.1 CUDA torch installed and `torch.cuda.is_available()` is True
- [x] 0.2 Packages installed (segment_anything, nibabel, SimpleITK, smp, surface-distance, cc3d, gdown, scikit-image, pyyaml)
- [x] 0.3 `checkpoints/sam_vit_b_01ec64.pth` downloaded (375 MB)
- [x] 0.4 `checkpoints/medsam_vit_b.pth` downloaded (375 MB)
- [x] 0.5 `datasets/raw/Task05_Prostate/` extracted (imagesTr + labelsTr)
- [x] 0.6 (40 vols extracted, tar deleted) `datasets/raw/Task03_Liver/` 40-volume subset extracted (tar is ~27 GB; extract only the subset)
  - NOTE: download running in background (resumable; restarted 03:18 after a connection reset). Run `bash scripts/extract_liver_subset.sh` when results/logs/liver_download.log shows 100%.
### Phase 1 — Preprocessing & zero-shot foundation baselines
- [x] 1.1 (prostate 377/51/47; liver 5678/908/625 slices, tumor on 2482/235/343) `data/preprocess_msd.py` written & run -> `datasets/npz/{liver,prostate}/{train,val,test}/*.npz`
- [x] 1.2 cache_embeddings run for prostate train/val/test (sam+medsam) — liver pending
- [x] 1.3 (prostate DONE: MedSAM DSC ~0.95 vs SAM ~0.92; point prompt fails ~0.06) Zero-shot SAM vs MedSAM evaluated (DSC + NSD tau=2, per-volume) -> `results/tables/zeroshot.csv`
### Phase 2 — Specialist baselines & prompt ablations
- [x] 2.1 (prostate DONE, DSC 0.930) 4-channel U-Net trained per task (`models/baselines_unet_deeplab.py`)
- [x] 2.2 (prostate DONE, DSC 0.921) 4-channel DeepLabV3+ (ResNet-50) trained per task
- [x] 2.3 (prostate DONE: medsam flat 0.945-0.959 up to 40px, sam drops 0.926->0.860) Box-jitter ablation {0,5,10,20,40}px + center-point prompt -> `results/tables/jitter.csv`
- [x] 2.4 (prostate DONE: k=1 0.945, k=3 0.899, k=5 0.874, k=10 0.608) 3-step sparse linear interpolation k in {3,5,10} on test volumes -> `results/tables/sparse3d.csv`
### Phase 3 — Extension
- [x] 3.1 LoRA-MedSAM (r=4) trained per task -> `checkpoints/lora_{liver,prostate}.pth`
- [x] 3.2 LoRA-MedSAM evaluated on test (DSC/NSD, VRAM, trainable params)
- [x] 3.3 YOLOv8n auto-prompt (yolo_medsam) trained + evaluated: recall liver 0.91 / prostate 0.91 / tumor 0.15; median DSC liver 0.946, prostate 0.882, tumor 0.074
### Phase 4 — Stats & figures
- [x] 4.1 Wilcoxon signed-rank tests across all pairs -> `results/tables/wilcoxon.csv`
- [x] 4.2 (scripts written & tested on prostate; rerun at end) Box plots, podium plots, qualitative overlays, jitter curve, timing bar -> `results/figures/`
- [x] 4.3 (make_report.py written; rerun at end) `results/REPORT.md` summarising all numbers + figure list for the final slides

## 4b. FINAL RESULTS SNAPSHOT (median over test volumes; full tables in results/REPORT.md)
| target | SAM | U-Net | DeepLabV3+ | MedSAM 0-shot | LoRA-MedSAM |
|---|---|---|---|---|---|
| liver DSC | 0.915 | 0.962 | 0.960 | 0.958 | 0.959 |
| tumor DSC | 0.264 | 0.538 | 0.671 | 0.352 | **0.602** |
| prostate DSC | 0.926 | 0.930 | 0.921 | 0.945 | **0.954** |

YOLOv8n+MedSAM fully automatic (no human box): liver 0.946, prostate 0.882, tumor 0.074 (detector recall 0.91/0.91/0.15 — small low-contrast lesions defeat a nano detector trained 12 epochs; honest limitation to report).
- Pooled Wilcoxon (11 vols, DSC): MedSAM vs SAM p=0.001; LoRA vs MedSAM p=0.001; MedSAM vs U-Net/DeepLab n.s. (in-distribution).
- Cross-task: liver-trained U-Net/DeepLab on prostate -> DSC 0.075/0.129; prostate-trained on liver -> 0.768/0.784. MedSAM zero-shot 0.945/0.958 unchanged (paper's core claim reproduced).
- Jitter: MedSAM flat up to 40 px on liver/prostate; SAM degrades (prostate 0.926->0.860). Center-point prompt fails (<0.08) for all targets.
- Sparse 3D interpolation: liver k=10 keeps DSC 0.951 with 19 boxes instead of 182; prostate (few slices) degrades to 0.608 at k=10.
- LoRA: 4.2M trainable / 93.9M (4.5%; LoRA ~0.15M + decoder 4.06M), peak VRAM 3.7 GB, 150 opt steps, ~43 min/target on RTX 3060 6 GB.
- Deliverables: results/REPORT.md, results/tables/*.csv, results/figures/*.png (19 figs: boxplots, podium, overlays, jitter curves, sparse3d), checkpoints/*.pth.

## 5. Decisions log
- 2026-11-16: Chose Approach A (LoRA) as the primary extension; C optional; B & D dropped (see §3).
- 2026-11-16: Liver limited to 40 volumes; images stored as uint8 1024x1024 single-channel in npz
  (3-channel repeat happens at load time) to save disk.

## 6. Run log (append newest at bottom: date/time, what ran, result, where the output is)
- 2026-11-16 01:05  Project skeleton + HANDOFF.md + SPEC.md created. Nothing downloaded yet.
- 2026-11-16 03:18  Phase 0 (steps 0.1-0.5 done): torch 2.6.0+cu124 (torchvision 0.21.0+cu124; a PyPI 2.15.0 torch was installed by mistake and replaced), CUDA available=True). Packages installed incl. segment-anything, nibabel, SimpleITK, smp, cc3d, gdown, skimage, pyyaml, surface-distance (all imports OK). SAM ckpt 375,042,383 B, MedSAM ckpt 375,049,145 B (both load, 314 keys). Prostate: 32 imagesTr + 32 labelsTr extracted to datasets/raw/Task05_Prostate (._ junk removed, tar deleted). Liver: curl started in background -> datasets/Task03_Liver.tar, log results/logs/liver_download.log; extract_liver_subset.sh written, NOT run yet.
- 2026-11-16 03:02  Wrote (py_compile OK, not executed: no GPU/torch yet) evaluation/losses.py, models/baselines_unet_deeplab.py, scripts/train_lora.py, scripts/train_baseline.py, scripts/run_benchmark.py, evaluation/sparse_3d_propagation.py (SPEC §7, §10, §11, §12).
- 2026-11-16 03:00  Code written (not run yet): configs/msd_{liver_ct,prostate_mri}.yaml, data/{common,preprocess_msd,dataset}.py, models/{medsam_wrapper,lora_medsam}.py, scripts/cache_embeddings.py, evaluation/metrics.py, empty __init__.py x3; pure-numpy unit tests passed (35 checks).

- 2026-11-16 03:53  Phase 4 scripts written and run on prostate (GPU job untouched, benchmark.csv not modified): evaluation/visualize_podium_overlays.py -> results/figures/ boxplot_dsc/nsd, podium, overlay (median slice), jitter_curve, sparse3d; scripts/make_report.py -> results/REPORT.md + results/tables/wilcoxon.csv (n=3 volumes, p floor 0.25). Liver/tumor figures pending liver data; no train_*.json yet.

## 7. Known issues / gotchas
- Always `set KMP_DUPLICATE_LIB_OK=TRUE` (PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`) before python.
- 6 GB VRAM: never run two GPU jobs at once. Check `nvidia-smi` first.
- 2026-11-16 03:20  All code modules written (SPEC §2-§12) by Haiku agents. Prostate preprocessed. Background job: cache embeddings (sam+medsam, prostate train/val/test) -> zero-shot benchmark medsam/sam/medsam_point. Log: results/logs/phase1_prostate.log (ends with EXIT=0 on success). Liver tar downloading (~6 MB/s, curl retry loop) -> then `bash scripts/extract_liver_subset.sh`, then `python -m data.preprocess_msd --task liver`.
- 2026-11-16 03:50  Prostate zero-shot done (benchmark.csv). Launched background chain results/logs/phase23_prostate.log: jitter ablation {5,10,20,40} sam+medsam -> sparse3d -> train unet -> bench -> train deeplab -> bench -> train_lora (600 steps) -> bench. Launched background waiter results/logs/liver_extract_preprocess.log: waits for curl to finish, extracts 40 liver vols, preprocesses liver. Each log ends with EXIT=0 on success. NEXT after both finish: cache liver embeddings (sam+medsam, train/val/test), zero-shot liver+tumor, then same phase23 chain for liver (targets liver and tumor; lora --max_steps 1500), then Phase 4 (visualize + make_report + wilcoxon). Phase 4 scripts (SPEC §14 visualize_podium_overlays.py, scripts/make_report.py) are NOT yet written.
- 2026-11-16 04:19  phase23_prostate chain: all OK except train_lora crashed (LoRA params created after .to(cuda) -> device mismatch). FIXED: added model.to(DEV) after apply_lora in scripts/train_lora.py L37 and scripts/run_benchmark.py L37. Re-running LoRA prostate (600 steps) + benchmark in background: results/logs/lora_prostate.log. Phase 4 scripts written (evaluation/visualize_podium_overlays.py, scripts/make_report.py), tested on prostate.
- 2026-11-16 04:49  Liver extracted+preprocessed OK. LoRA was too slow (2.9 s/micro-batch, bs1 accum8 => 600 opt steps ~4 h) so killed it and CAPPED budgets: prostate LoRA 150 opt steps/3 epochs, liver LoRA 150 opt steps/1 epoch per target, liver baselines 6 epochs. Launched MASTER CHAIN `scripts/run_all_remaining.sh` -> log results/logs/run_all_remaining.log (lines '=== HH:MM:SS STEP: ...' mark progress; ends with 'ALL DONE' and EXIT=0). It runs: prostate LoRA+bench -> liver embeddings (val/test only; train NOT cached to save disk) -> liver zero-shot -> jitter -> sparse3d -> baselines -> LoRA tumor+liver -> figures -> REPORT.md. Expected finish ~6-7 h. IF IT DIES: read the last STEP line in the log, fix, and rerun the remaining commands from that step (copy them from the script). GPU must be free first (nvidia-smi). Disk: ~34 GB free; don't cache liver train embeddings.
- 2026-11-16 09:46  MASTER CHAIN FINISHED 09:41 EXIT=0 (all liver/prostate experiments, figures, REPORT.md). Ran cross-task specialist eval (tag crosstask_*, results/logs/crosstask.log). Haiku agent adding pooled-Wilcoxon + cross-task sections to scripts/make_report.py and regenerating REPORT.md. PROJECT COMPLETE except optional 3.3 (YOLO). Possible polish if budget allows: write slides from REPORT.md; longer LoRA training (raise --max_steps) for tumor.
- 2026-11-16 10:41  YOLO auto-prompt done for both tasks (models/auto_prompt_yolo.py; logs results/logs/yolo_*.log). Liver YOLO first died silently after dataset caching with workers=8 (RAM); fixed with workers=2. Figures + REPORT.md regenerated with yolo_medsam. PROJECT COMPLETE — all status-board items ticked. Deliverables: results/REPORT.md, results/tables/, results/figures/ (19 png), checkpoints/.
