# Scripts, Execution Runbook, and CLI Reference

This document provides a comprehensive operational guide for setting up the environment, executing the end-to-end data processing and modeling pipelines, utilizing the master orchestration scripts, and troubleshooting hardware-specific issues.

---

## 1. Environment & Hardware Prerequisites

### 1.1 Hardware Specifications
- **Operating System**: Windows 11 (tested on Windows 11 Build 22631 / PowerShell).
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU with **6 GB VRAM** (CUDA 12.3 / 12.4).
- **Host RAM**: 16 GB DDR4/DDR5.
- **Disk Space**: At least 35 GB free (after deleting raw `.tar` archives).

### 1.2 Mandatory Environment Variable
On Windows systems with PyTorch and NumPy installed, multiple OpenMP runtime libraries (`libiomp5md.dll`) can conflict, throwing `OMP: Error #15: Initializing libiomp5md.dll...`.
To prevent this, you **must set the environment variable**:

```powershell
# PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```
```bash
# Bash / WSL
export KMP_DUPLICATE_LIB_OK=TRUE
```
Every Python entry script in this repository also includes `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` at the top of the file as a defensive measure.

### 1.3 Installation Commands
```bash
# 1. Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install Segment Anything (Meta AI)
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Install surface-distance (Google DeepMind)
pip install git+https://github.com/deepmind/surface-distance.git

# 4. Install supporting medical and computer vision packages
pip install nibabel SimpleITK segmentation-models-pytorch connected-components-3d gdown scikit-image pyyaml ultralytics seaborn tqdm
```

---

## 2. Checkpoint & Raw Data Acquisition

### 2.1 Model Checkpoints (`checkpoints/`)
Create the directory and download pre-trained weights:
```bash
mkdir -p checkpoints

# Segment Anything ViT-B (Meta AI, 375 MB)
curl -L -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# MedSAM ViT-B (Wang et al., 375 MB) via gdown
gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam_vit_b.pth
```

### 2.2 MSD Raw Datasets (`datasets/raw/`)
```bash
mkdir -p datasets/raw

# 1. Download Task05 Prostate (~230 MB) and extract
curl -L -o datasets/Task05_Prostate.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar
tar -xf datasets/Task05_Prostate.tar -C datasets/raw/
rm datasets/Task05_Prostate.tar
# Remove macOS metadata files if present
find datasets/raw -name '._*' -type f -delete

# 2. Download Task03 Liver (~27 GB)
# Because Liver is 27 GB, use the automated extraction script to extract only 40 volumes
curl -L -C - -o datasets/Task03_Liver.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar
bash scripts/extract_liver_subset.sh
```

---

## 3. End-to-End Execution Pipeline (Step-by-Step)

All scripts are executed from the repository root directory.

```
       Phase 1                    Phase 2                   Phase 3                 Phase 4
  Data Ingestion &          Specialist Models &         LoRA-MedSAM &         Reporting & Figures
 Zero-Shot Baselines         Prompt Ablations         Autonomous YOLO            Generation
┌────────────────────┐    ┌────────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ preprocess_msd.py  │───►│ train_baseline.py  │───►│ train_lora.py   │───►│ visualize_podium... │
│ cache_embeddings.py│    │ run_benchmark.py   │    │ auto_prompt_yolo│    │ make_report.py      │
│ run_benchmark.py   │    │ sparse_3d_prop...  │    └─────────────────┘    └─────────────────────┘
└────────────────────┘    └────────────────────┘
```

### Phase 1: Preprocessing & Zero-Shot Baselines

#### Step 1.1: Preprocess MSD Volumes to `.npz` Slices
```bash
python -m data.preprocess_msd --task prostate
python -m data.preprocess_msd --task liver
```
*Outputs: Slices in `datasets/npz/{liver,prostate}/{train,val,test}/*.npz`, `volumes.json`, and `results/tables/dataset_stats.csv`.*

#### Step 1.2: Cache ViT-B Image Embeddings
Precompute the $64 \times 64 \times 256$ feature maps for fast prompt evaluation:
```bash
# Prostate: Cache all splits for both SAM and MedSAM
for split in train val test; do
  python scripts/cache_embeddings.py --task prostate --split $split --model medsam
  python scripts/cache_embeddings.py --task prostate --split $split --model sam
done

# Liver: Cache val and test splits (skipping train to save disk space)
for split in val test; do
  python scripts/cache_embeddings.py --task liver --split $split --model medsam
  python scripts/cache_embeddings.py --task liver --split $split --model sam
done
```

#### Step 1.3: Run Zero-Shot Foundation Benchmarks (Jitter 0)
```bash
# Zero-shot Prostate
python scripts/run_benchmark.py --task prostate --target prostate --method medsam --save_preds
python scripts/run_benchmark.py --task prostate --target prostate --method sam --save_preds
python scripts/run_benchmark.py --task prostate --target prostate --method medsam_point

# Zero-shot Liver (Parenchyma and Tumor)
for target in liver tumor; do
  python scripts/run_benchmark.py --task liver --target $target --method medsam --save_preds
  python scripts/run_benchmark.py --task liver --target $target --method sam --save_preds
  python scripts/run_benchmark.py --task liver --target $target --method medsam_point
done
```

---

### Phase 2: Specialist Baselines & Prompt Ablations

#### Step 2.1: Train Specialist 4-Channel Baselines (U-Net & DeepLabV3+)
```bash
# Prostate (25 epochs)
python scripts/train_baseline.py --task prostate --target prostate --model unet --epochs 25
python scripts/train_baseline.py --task prostate --target prostate --model deeplab --epochs 25

# Evaluate Prostate baselines
python scripts/run_benchmark.py --task prostate --target prostate --method unet --save_preds
python scripts/run_benchmark.py --task prostate --target prostate --method deeplab --save_preds

# Liver & Tumor (6 epochs each to fit GPU budget)
for target in liver tumor; do
  for model in unet deeplab; do
    python scripts/train_baseline.py --task liver --target $target --model $model --epochs 6
    python scripts/run_benchmark.py --task liver --target $target --method $model --save_preds
  done
done
```

#### Step 2.2: Prompt Box-Jitter Sensitivity Sweeps
```bash
# Prostate Jitter {5, 10, 20, 40} px
for j in 5 10 20 40; do
  python scripts/run_benchmark.py --task prostate --target prostate --method medsam --jitter $j --tag jitter
  python scripts/run_benchmark.py --task prostate --target prostate --method sam --jitter $j --tag jitter
done

# Liver Jitter
for target in liver tumor; do
  for j in 5 10 20 40; do
    python scripts/run_benchmark.py --task liver --target $target --method medsam --jitter $j --tag jitter
    python scripts/run_benchmark.py --task liver --target $target --method sam --jitter $j --tag jitter
  done
done
```

#### Step 2.3: Sparse 3D Slice Propagation
```bash
python evaluation/sparse_3d_propagation.py --task prostate --target prostate --ks 1,3,5,10
python evaluation/sparse_3d_propagation.py --task liver --target liver --ks 1,3,5,10
python evaluation/sparse_3d_propagation.py --task liver --target tumor --ks 1,3,5,10
```

---

### Phase 3: LoRA-MedSAM & YOLOv8n Auto-Prompting

#### Step 3.1: Train LoRA-MedSAM
```bash
# Prostate: 150 optimizer steps (3 epochs, batch 1 x accum 8)
python scripts/train_lora.py --task prostate --target prostate --max_steps 150 --epochs 3
python scripts/run_benchmark.py --task prostate --target prostate --method lora --save_preds

# Liver: 150 optimizer steps per target
python scripts/train_lora.py --task liver --target liver --max_steps 150 --epochs 1
python scripts/run_benchmark.py --task liver --target liver --method lora --save_preds

python scripts/train_lora.py --task liver --target tumor --max_steps 150 --epochs 1
python scripts/run_benchmark.py --task liver --target tumor --method lora --save_preds
```

#### Step 3.2: YOLOv8n Auto-Prompting (Export, Train, Eval)
```bash
# Run end-to-end YOLO pipeline (export npz -> train yolov8n -> detect test slices -> MedSAM inference)
python models/auto_prompt_yolo.py --task prostate --stage all
python models/auto_prompt_yolo.py --task liver --stage all
```

---

### Phase 4: Reporting & Publication Visualizations

#### Step 4.1: Generate Publication Figures
```bash
python evaluation/visualize_podium_overlays.py
```
*Generates 19 figures in `results/figures/`: boxplots, podium plots, qualitative slice overlays, jitter curves, and sparse 3D graphs.*

#### Step 4.2: Compile Comprehensive Markdown Report
```bash
python scripts/make_report.py
```
*Compiles all numbers into `results/REPORT.md`, computes Wilcoxon tests, cross-task generalizations, and training tables.*

---

## 4. Master Orchestration Scripts

### 4.1 `scripts/extract_liver_subset.sh`
[scripts/extract_liver_subset.sh](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/scripts/extract_liver_subset.sh) automates extracting the deterministic 40-volume subset from the 27 GB Liver archive and immediately removes the archive to preserve disk space:
```bash
bash scripts/extract_liver_subset.sh
```

### 4.2 `scripts/run_all_remaining.sh`
[scripts/run_all_remaining.sh](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/scripts/run_all_remaining.sh) provides a single automated master chain executing all remaining training and evaluation steps:
```bash
bash scripts/run_all_remaining.sh
```
Progress is tracked with timestamped step markers `=== HH:MM:SS STEP: ...`, and the full stdout is logged to `results/logs/run_all_remaining.log`.

---

## 5. CLI Command Reference

| Script | Key Arguments | Purpose | Main Outputs |
| :--- | :--- | :--- | :--- |
| `data/preprocess_msd.py` | `--task {liver,prostate}` | Ingests NIfTI, normalizes intensities, saves `.npz` slices. | `datasets/npz/<task>/`, `volumes.json`, `dataset_stats.csv` |
| `scripts/cache_embeddings.py`| `--task`, `--split`, `--model {sam,medsam}` | Precomputes ViT-B feature embeddings. | `cache/<model>/<task>/<split>/*.npy` |
| `scripts/train_baseline.py` | `--model {unet,deeplab}`, `--task`, `--target`, `--epochs`, `--bs 16` | Trains 4-channel specialist CNN baselines. | `checkpoints/<model>_<task>_<target>.pth`, `train_<model>_*.json` |
| `scripts/train_lora.py` | `--task`, `--target`, `--rank 4`, `--max_steps 150`, `--bs 1`, `--accum 8` | Trains LoRA adapters + Mask Decoder with activation checkpointing. | `checkpoints/lora_<task>_<target>.pth`, `train_lora_*.json` |
| `scripts/run_benchmark.py` | `--task`, `--target`, `--method`, `--jitter`, `--ckpt`, `--save_preds` | Runs inference, 3D volume reconstruction, and 3D metrics. | `results/tables/benchmark.csv`, `results/preds/` |
| `evaluation/sparse_3d_propagation.py` | `--task`, `--target`, `--ks 1,3,5,10` | Evaluates sparse axial slice bounding box interpolation. | `results/tables/sparse3d.csv` |
| `models/auto_prompt_yolo.py`| `--task`, `--stage {export,train,eval,all}` | Trains YOLOv8n detector and evaluates automated prompting. | `results/yolo/`, `yolo_<task>.json`, `benchmark.csv` |
| `evaluation/visualize_podium_overlays.py` | `--task`, `--target` | Renders publication-quality PNG figures. | `results/figures/*.png` (19 figures) |
| `scripts/make_report.py` | *(no arguments)* | Aggregates tables, computes Wilcoxon tests, writes markdown report. | `results/REPORT.md`, `wilcoxon.csv`, `crosstask.csv` |

---

## 6. Troubleshooting & Gotchas

1. **Windows OpenMP Clash**:
   - Symptom: `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`.
   - Resolution: Set `$env:KMP_DUPLICATE_LIB_OK="TRUE"` before launching Python.

2. **6 GB VRAM Management**:
   - Never run concurrent GPU jobs. Verify with `nvidia-smi` that GPU memory is clear before starting training.
   - If an Out-Of-Memory error occurs during `train_lora.py` in the first 5 steps, the script automatically catches `OOMFallback` and restarts in `DECODER_ONLY` mode using cached embeddings.

3. **DataLoader Multiprocessing on Windows**:
   - Avoid setting `num_workers > 4` on Windows. Windows uses `spawn` instead of `fork`, which can cause CPU memory thrashing if worker counts are too high. In `auto_prompt_yolo.py`, `workers=2` is used for stability.

4. **Missing Embeddings Fallback**:
   - If `cache/` is missing or incomplete during `run_benchmark.py`, the script automatically detects missing `.npy` files and encodes them on-the-fly via `model.encode(img)`, logging the number of cache misses.
