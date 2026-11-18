# MedSAM Domain-Shift Evaluation & Adaptation Architecture

**Group 5, Dhirubhai Ambani University (DAU) — Computer Vision Project**  
*Team Members*: Dhruv Patel (202301095), Kavish Patel (202301074), Prasham Shah (202301252)  
*Reference Paper*: Jun Ma et al., *"Segment anything in medical images"*, Nature Communications 15:654 (2024).  
*Source Code Repository*: [https://github.com/Prasham27/MedSAM](https://github.com/Prasham27/MedSAM) (branch `main`).

---

## 1. Executive Summary & Research Questions

This codebase implements an empirical evaluation and architectural extension of **MedSAM** under clinical domain shift across two distinct radiological modalities from the Medical Segmentation Decathlon (MSD):
1. **Task03_Liver**: Abdominal Computed Tomography (CT), targeting both high-contrast liver parenchyma and small, low-contrast liver lesions/tumors.
2. **Task05_Prostate**: Pelvic T2-weighted Magnetic Resonance Imaging (MRI), targeting the whole prostate gland.

### Core Scientific Questions & Findings
1. **Zero-Shot Transfer vs. Natural Image Foundation**: Does medical pre-training (MedSAM) yield statistically superior segmentation over vanilla Segment Anything (SAM)?
   - *Result*: **Yes**. On pooled Wilcoxon signed-rank tests across 11 test volumes, MedSAM achieves a statistically significant improvement ($p = 0.0010$, median 3D DSC lead of $+0.0455$).
2. **Foundation Models vs. Specialist Supervised Baselines**: How does zero-shot MedSAM compare to fully supervised domain specialists (4-channel U-Net and DeepLabV3+)?
   - *Result*: MedSAM matches supervised specialists in-distribution ($p = 0.8984$ vs U-Net; $p = 0.5195$ vs DeepLabV3+), but **demonstrates complete immunity to cross-domain catastrophic collapse**. When evaluated cross-task, specialist models experience catastrophic failure (prostate-trained U-Net collapses from 0.930 to 0.075 DSC on liver CT), while zero-shot MedSAM maintains unchanged accuracy (0.958 CT / 0.945 MRI).
3. **Parameter-Efficient Adaptation**: Can low-rank adaptation (LoRA, $r=4$) in ViT self-attention improve performance without catastrophic forgetting on consumer hardware?
   - *Result*: **Yes**. LoRA-MedSAM fine-tunes only **4.2M parameters (4.5% of the model)**, fits comfortably within a **6 GB VRAM budget (3.7 GB peak)** via activation checkpointing, and delivers statistically significant gains ($p = 0.0010$ vs MedSAM; liver tumor DSC improves from 0.352 to **0.602**).
4. **Prompt Robustness & Clinical Workflow Feasibility**:
   - *Box Jitter*: MedSAM is remarkably robust to bounding box noise up to 40 pixels, while vanilla SAM degrades.
   - *Point Prompting*: Single-point prompts fail catastrophically in medical imaging (DSC $< 0.08$ across all targets).
   - *Sparse 3D Propagation*: Linear interpolation of bounding boxes across axial slices allows clinicians to annotate only every 10th slice ($k=10$), slashing annotation time by **90%** (19 boxes vs 182) while retaining 0.951 3D DSC on liver CT.
   - *Autonomous Prompting*: Pairing MedSAM with a YOLOv8n detector eliminates human prompting for organs (DSC 0.946 liver, 0.882 prostate), though small lesion detection remains a technical limitation (0.074 tumor DSC).

---

## 2. High-Level System Architecture

The following diagram depicts the end-to-end dataflow, training, inference, and reporting pipeline:

```mermaid
flowchart TD
    %% Subgraph Data Layer
    subgraph Data_Layer ["Data Layer & Ingestion"]
        RAW_CT["MSD Task03 Liver (CT)"] --> PRE["data/preprocess_msd.py"]
        RAW_MRI["MSD Task05 Prostate (MRI)"] --> PRE
        CFG["configs/*.yaml"] --> PRE
        
        PRE --> NPZ["datasets/npz/<task>/<split>/*.npz<br/>(1024x1024 uint8, single-channel)"]
        PRE --> VJSON["datasets/npz/<task>/volumes.json<br/>(3D spacing, original shape, total z)"]
        PRE --> SPLIT["datasets/splits.json<br/>(80/10/10 Patient Volume-Level)"]
        
        NPZ --> CACHE_SCRIPT["scripts/cache_embeddings.py"]
        CACHE_SCRIPT --> CACHE["cache/<model>/<task>/<split>/*.npy<br/>(256x64x64 float16, 2 MB/slice)"]
        
        NPZ --> YOLO_EXP["models/auto_prompt_yolo.py (export)"]
        YOLO_EXP --> YOLO_DATA["datasets/yolo/<task>/<br/>(640x640 JPEG + YOLO txt)"]
    end

    %% Subgraph Model Layer
    subgraph Model_Layer ["Model & Adaptation Layer"]
        CKPT_SAM["checkpoints/sam_vit_b_01ec64.pth"] --> WRAP["models/medsam_wrapper.py<br/>(MedSAM Wrapper)"]
        CKPT_MED["checkpoints/medsam_vit_b.pth"] --> WRAP
        
        WRAP -->|LoRA Injection (r=4, alpha=4)| LORA_MOD["models/lora_medsam.py<br/>(LoRAQKV in ViT Blocks + Checkpointing)"]
        
        SPEC_DEF["models/baselines_unet_deeplab.py"] --> BASE_NETS["4-Channel CNN Baselines<br/>U-Net (ResNet-34) / DeepLabV3+ (ResNet-50)"]
        
        YOLO_DATA --> YOLO_TRAIN["models/auto_prompt_yolo.py (train)"]
        YOLO_TRAIN --> YOLO_CKPT["results/yolo/<task>/weights/best.pt"]
    end

    %% Subgraph Training Layer
    subgraph Training_Layer ["Training & Optimization Layer"]
        NPZ & CACHE --> TRAIN_LORA["scripts/train_lora.py<br/>(bs=1, accum=8, lr=1e-4, AdamW, bf16)"]
        LORA_MOD --> TRAIN_LORA
        TRAIN_LORA --> CKPT_LORA["checkpoints/lora_<task>_<target>.pth<br/>(Only LoRA + Decoder, ~17 MB)"]
        
        NPZ --> TRAIN_BASE["scripts/train_baseline.py<br/>(bs=16, 256x256, fp16, AdamW)"]
        BASE_NETS --> TRAIN_BASE
        TRAIN_BASE --> CKPT_BASE["checkpoints/<model>_<task>_<target>.pth"]
    end

    %% Subgraph Evaluation Layer
    subgraph Eval_Layer ["Evaluation & Inference Layer"]
        NPZ & VJSON & CACHE --> BENCH["scripts/run_benchmark.py"]
        WRAP & CKPT_LORA & CKPT_BASE --> BENCH
        YOLO_CKPT --> AUTO_EVAL["models/auto_prompt_yolo.py (eval)"]
        
        BENCH & AUTO_EVAL --> RECON["evaluation/metrics.py: reconstruct_volume()<br/>(Stack Slices to 3D Volume at Orig Matrix)"]
        RECON --> POST["evaluation/metrics.py: postprocess_3d()<br/>(Box Masking + cc3d Largest Component)"]
        POST --> METRICS["Compute 3D DSC, 3D NSD (tau=2mm)<br/>Slice Mean DSC, Latency"]
        
        NPZ & CACHE --> SPARSE["evaluation/sparse_3d_propagation.py<br/>(k in {1, 3, 5, 10} Linear Box Interpolation)"]
    end

    %% Subgraph Output Layer
    subgraph Reporting_Layer ["Statistical & Reporting Layer"]
        METRICS --> CSV_BENCH["results/tables/benchmark.csv"]
        SPARSE --> CSV_SPARSE["results/tables/sparse3d.csv"]
        
        CSV_BENCH --> STATS["scripts/make_report.py"]
        STATS --> WILCOXON["results/tables/wilcoxon.csv<br/>results/tables/wilcoxon_pooled.csv"]
        STATS --> CROSSTASK["results/tables/crosstask.csv"]
        STATS --> REPORT_MD["results/REPORT.md"]
        
        CSV_BENCH & CSV_SPARSE --> VIS["evaluation/visualize_podium_overlays.py"]
        VIS --> FIGURES["results/figures/*.png (19 Figures)<br/>Boxplots, Podium, Overlays, Jitter, Sparse3D"]
    end
```

---

## 3. Subsystem Breakdown

### 3.1 Data Subsystem
Detailed documentation: [docs/database_and_preprocessing.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/database_and_preprocessing.md).

1. **Modality Ingestion & Intensity Windowing**:
   - *CT (Task03_Liver)*: Linear mapping of Hounsfield Units within window $[-160, +240] \text{ HU}$ (width 400, level 40) into $[0, 255]$ uint8.
   - *MRI (Task05_Prostate)*: Percentile clipping at $[0.5, 99.5]$ over non-zero voxels followed by min-max scaling into $[0, 255]$ uint8. Structural T2 channel extracted from 4D array.
2. **Geometric Normalization**:
   - Transposition of 2D slices from $(x, y)$ to $(y, x)$ to ensure anatomical orientation matches OpenCV/PyTorch conventions.
   - Bicubic interpolation (`cv2.INTER_CUBIC`) to $1024 \times 1024$ for images; nearest-neighbor interpolation (`cv2.INTER_NEAREST`) for target masks.
   - Mask area thresholding ($\sum M \ge 100$ pixels) filters out trivial boundary noise.
3. **Storage Schemas**:
   - `.npz` slices: Individual compressed archives containing `img` (uint8 1024x1024), `gts` (stacked uint8 binary masks), `names`, `spacing`, `z`, and `orig_shape`.
   - `volumes.json`: Registry storing original resolution $[H, W]$, voxel spacing $[s_x, s_y, s_z]$, and total axial slices $n_z\_total$.
   - `datasets/splits.json`: Patient volume-level 80/10/10 split ensuring zero inter-slice patient leakage.
4. **Feature Caching (`cache/`)**:
   - Precomputed ViT-B image embeddings ($256 \times 64 \times 64$ float16, 2 MB per slice), eliminating $95\%+$ of computational cost during prompt sensitivity and ablation experiments.

---

### 3.2 Model & Adaptation Subsystem
Detailed documentation: [docs/models_and_loading.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/models_and_loading.md).

```
               Vision Transformer Encoder (ViT-B/16)
               ┌───────────────────────────────────┐
               │ 12 Transformer Self-Attention Blks│
               │ [Frozen Base Weights: 89.7M]      │
               └─────────────────┬─────────────────┘
                                 │ Injected with
                                 ▼
                     LoRA Adapters (r=4, alpha=4)
                     ┌───────────────────────────┐
                     │ Query (Aq, Bq) & Value    │
                     │ (Av, Bv) Projections      │
                     │ [147K Trainable Params]   │
                     └───────────────────────────┘
                                 │
                                 ▼
                        Image Embedding Tensor
                           [B, 256, 64, 64]
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
         Prompt Encoder                     Mask Decoder
     Fourier Positional Tokens           2-Way Cross-Attention
     Boxes [B, 2, 256] (Frozen)          Transposed Conv Upsampling
                                         [4.06M Trainable Params]
                                                  │
                                                  ▼
                                            Output Logits
                                         [B, 1, 1024, 1024]
```

1. **Foundation Wrappers ([models/medsam_wrapper.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/models/medsam_wrapper.py))**:
   - Loads vanilla SAM or MedSAM checkpoints via `sam_model_registry['vit_b']`.
   - Direct $[0, 1]$ float tensor ingestion into `image_encoder`, bypassing standard SAM ImageNet mean/std normalization.
   - Decodes bounding box prompts into $1024 \times 1024$ continuous logits.
2. **LoRA-MedSAM ([models/lora_medsam.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/models/lora_medsam.py))**:
   - Wraps fused `blk.attn.qkv` projections with `LoRAQKV`.
   - Adds low-rank factor matrices: $A_q, A_v \in \mathbb{R}^{768 \times 4}$ (Kaiming uniform) and $B_q, B_v \in \mathbb{R}^{4 \times 768}$ (zeros).
   - Key projection is unmodified base weights.
   - Trainable parameters: **4,205,796 (4.5% of total)**.
   - Checkpoints save only LoRA factor matrices and mask decoder weights (~17 MB).
3. **Specialist Baselines ([models/baselines_unet_deeplab.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/models/baselines_unet_deeplab.py))**:
   - 4-channel architectures (Channels 0–2: RGB image; Channel 3: rasterized bounding box prompt mask).
   - U-Net with ResNet-34 backbone (24.4M params); DeepLabV3+ with ResNet-50 backbone (26.7M params).
   - Ingests $256 \times 256$ inputs and bilinearly upsamples logits to $1024 \times 1024$ for volumetric evaluation.
4. **Autonomous Detection ([models/auto_prompt_yolo.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/models/auto_prompt_yolo.py))**:
   - Ultralytics YOLOv8 nano model trained on exported 640 px slices.
   - Detects organ/tumor bounding boxes with confidence threshold $\ge 0.25$ and delivers them directly to MedSAM.

---

### 3.3 Evaluation & Benchmark Subsystem
Detailed documentation: [docs/evaluation_and_results.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/evaluation_and_results.md).

1. **3D Reconstruction & Post-Processing**:
   - Slices are resized with nearest neighbor to original dimensions $(W, H)$ and assembled into 3D boolean arrays spanning $n_z\_total$ slices.
   - Post-processing enforces spatial bounding box containment and retains only the single largest 3D connected component (`cc3d.largest_k`).
2. **Volumetric Metrics**:
   - **3D DSC**: Voxel overlap ratio.
   - **3D NSD ($\tau = 2.0 \text{ mm}$)**: Boundary contour accuracy computed via DeepMind's `surface_distance` library using true physical voxel zooms $[s_z, s_y, s_x]$.
   - Mean slice DSC and seconds per slice latency profiling.
3. **Statistical Significance Testing**:
   - Paired two-sided Wilcoxon signed-rank tests against MedSAM.
   - Pooled test across all 11 test volumes confirms statistical superiority of MedSAM over SAM ($p = 0.0010$) and LoRA-MedSAM over MedSAM ($p = 0.0010$).
4. **Experimental Studies**:
   - *Box-Jitter Sweep*: Perturbations of $0, 5, 10, 20, 40$ pixels demonstrate MedSAM stability and SAM degradation.
   - *Sparse 3D Propagation*: Axial bounding box interpolation with strides $k \in \{1, 3, 5, 10\}$.
   - *Cross-Task Domain Generalization*: Proves catastrophic domain collapse in specialist CNNs and zero-shot transfer invariance in MedSAM.

---

### 3.4 Orchestration & Execution Subsystem
Detailed documentation: [docs/scripts_and_execution.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/scripts_and_execution.md).

1. **Master Automation**:
   - [scripts/run_all_remaining.sh](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/scripts/run_all_remaining.sh): Complete end-to-end execution pipeline with timestamped logging.
   - [scripts/extract_liver_subset.sh](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/scripts/extract_liver_subset.sh): Deterministic extraction of 40 sorted volumes from the 27 GB archive and tar cleanup.
2. **Reporting & Figure Generation**:
   - [scripts/make_report.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/scripts/make_report.py): Compiles all CSV tables, computes Wilcoxon tests, and synthesizes [results/REPORT.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/results/REPORT.md).
   - [evaluation/visualize_podium_overlays.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/evaluation/visualize_podium_overlays.py): Renders 19 publication-ready PNG figures in `results/figures/`.

---

## 4. Hardware Optimization Architecture (6 GB VRAM)

Training Vision Transformers on full-resolution $1024 \times 1024$ images typically requires enterprise-grade GPUs (e.g. A100 40/80 GB). To execute this project on a single **NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)**, four coordinated architectural optimizations were implemented:

```
                               6 GB VRAM Optimization
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
Activation          Bfloat16 / Float16             Precomputed           Micro-Batching &
Checkpointing           Autocast                 Embedding Cache         Grad Accumulation
Eliminates storing   Reduces tensor              Pre-encodes ViT-B       Batch size = 1
activations in all   memory footprints           features (2 MB/slice)   Accumulation = 8
12 ViT blocks        by 50%                      Saves 95% compute       Effective batch = 8
(Peak: 3.7 GB)                                   (Bypasses encoder)
```

1. **Activation Checkpointing**: Wrapping `blk.forward` in `torch.utils.checkpoint.checkpoint(orig, x, use_reentrant=False)` during LoRA training recomputes activations on backward instead of holding them in memory, dropping peak VRAM from $>10 \text{ GB}$ to **3.7 GB**.
2. **Mixed Precision (AMP)**: `torch.autocast(dtype=torch.bfloat16)` on CUDA for ViT forward passes; `float16` with `GradScaler` for baseline CNNs.
3. **Precomputed Embedding Cache**: Feature maps $[256, 64, 64]$ are stored in float16 `.npy` format on disk. Decoder-only evaluations and prompt sweeps load embeddings directly, achieving inference speeds of $\sim 0.02 \text{ s/slice}$ and bypassing the ViT encoder completely.
4. **Micro-Batching & Dynamic Fallback**: LoRA uses micro-batch size 1 with gradient accumulation across 8 steps. Furthermore, [scripts/train_lora.py](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/scripts/train_lora.py) implements an automated `OOMFallback` handler: if an Out-Of-Memory error occurs during the first 5 steps, memory is flushed and training automatically falls back to `DECODER_ONLY` mode on cached embeddings.

---

## 5. Complete Codebase Directory & File Catalog

```
MedSAM/
├── configs/                                # Modality and preprocessing specifications
│   ├── msd_liver_ct.yaml                   # Task03 Liver CT configuration (HU windowing, 40 vols)
│   └── msd_prostate_mri.yaml               # Task05 Prostate MRI configuration (percentile clip, 32 vols)
│
├── data/                                   # Data layer and dataset loaders
│   ├── __init__.py                         # Package initialization
│   ├── common.py                           # Root path resolution, config loader, logging handler
│   ├── dataset.py                          # MSDSliceDataset, bounding box jitter, prompt channel encoding
│   └── preprocess_msd.py                   # NIfTI volume ingestion, intensity normalization, .npz saving
│
├── models/                                 # Model architectures and wrappers
│   ├── __init__.py                         # Package initialization
│   ├── auto_prompt_yolo.py                 # YOLOv8n dataset export, training, and automated prompting
│   ├── baselines_unet_deeplab.py           # 4-channel U-Net and DeepLabV3+ baseline architectures
│   ├── lora_medsam.py                      # LoRAQKV self-attention adaptation, checkpointing, parameter accounting
│   └── medsam_wrapper.py                   # Unified MedSAM/SAM wrapper for box and point prompting
│
├── evaluation/                             # Volumetric evaluation and visualization
│   ├── __init__.py                         # Package initialization
│   ├── losses.py                           # Combined BCE + Soft Dice loss, slice-level Dice
│   ├── metrics.py                          # 3D DSC, 3D NSD (tau=2mm), 3D postprocessing, Wilcoxon test
│   ├── sparse_3d_propagation.py            # Axial slice bounding box interpolation (k in {1, 3, 5, 10})
│   └── visualize_podium_overlays.py        # Generation of boxplots, podium plots, overlays, jitter curves
│
├── scripts/                                # Execution entrypoints and automation
│   ├── cache_embeddings.py                 # Precomputes ViT-B image embeddings to disk
│   ├── extract_liver_subset.sh             # Bash script for deterministic 40-volume Liver extraction
│   ├── make_report.py                      # Compiles benchmark.csv into results/REPORT.md
│   ├── run_all_remaining.sh                # Master shell orchestration script
│   ├── run_benchmark.py                    # Inference, volume reconstruction, and metric logging
│   ├── train_baseline.py                   # Supervised training loop for U-Net and DeepLabV3+
│   └── train_lora.py                       # LoRA-MedSAM parameter-efficient fine-tuning
│
├── docs/                                   # In-depth architectural references
│   ├── database_and_preprocessing.md       # Detailed guide to data ingestion, schemas, and preprocessing
│   ├── models_and_loading.md               # Detailed guide to architectures, LoRA, and model loading
│   ├── evaluation_and_results.md           # Detailed guide to 3D evaluation, metrics, and findings
│   └── scripts_and_execution.md            # Detailed operational manual, CLI reference, and troubleshooting
│
├── results/                                # Experimental outputs (committed to repository)
│   ├── figures/                            # 19 publication-ready PNG figures
│   ├── logs/                               # Execution logs for every stage and task
│   ├── preds/                              # 256x256 per-slice prediction PNGs and box JSONs
│   ├── tables/                             # CSV tables (benchmark, wilcoxon, crosstask, sparse3d)
│   └── REPORT.md                           # Master synthesized experimental report
│
├── ARCHITECTURE.md                         # Master architectural blueprint (this file)
├── HANDOFF.md                              # Operational status board, decisions, and run log
├── SPEC.md                                 # Technical specification for all modules
└── .gitignore                              # Excludes large binaries (datasets/, cache/, checkpoints/)
```

---

## 6. Key References & Documentation Index

For deep-dive technical explorations of specific subsystems, consult the corresponding documentation modules:
- **Data Layer & Preprocessing**: [docs/database_and_preprocessing.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/database_and_preprocessing.md)
- **Model Architectures & LoRA Adaptation**: [docs/models_and_loading.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/models_and_loading.md)
- **Evaluation Protocols & Results**: [docs/evaluation_and_results.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/evaluation_and_results.md)
- **Execution Runbook & CLI Reference**: [docs/scripts_and_execution.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/docs/scripts_and_execution.md)
- **Project Handoff & Run Log**: [HANDOFF.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/HANDOFF.md)
- **Technical Specification**: [SPEC.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/SPEC.md)
- **Synthesized Results & Tables**: [results/REPORT.md](file:///c:/Users/Dhruv/VSCodeProjects/MedSAM/results/REPORT.md)
