# SPEC — exact technical specification for every module
Root = this folder. All scripts are run from root: `python -m data.preprocess_msd ...` or `python scripts/x.py`.
Every script: argparse, `--task {liver,prostate}`, prints progress with tqdm, writes to `results/logs/<script>_<task>.log`.
Set `os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")` at the top of every entry script.
Add `sys.path.insert(0, <root>)` at top of scripts so `from data.dataset import ...` works.
All randomness seeded with 42. Device = cuda if available.

## 1. Environment (Phase 0)
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install nibabel SimpleITK segmentation-models-pytorch connected-components-3d gdown scikit-image pyyaml
pip install git+https://github.com/deepmind/surface-distance.git
```
Checkpoints -> `checkpoints/`:
- SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- MedSAM: Google Drive id `1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_` (file `medsam_vit_b.pth`) via `gdown`.
  If gdown fails, try `gdown --fuzzy "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN"` or look up "MedSAM checkpoint medsam_vit_b.pth" (bowang-lab/MedSAM README).
Datasets (AWS public mirror used by MONAI):
- https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar  (~230 MB)
- https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar     (~27 GB)
Extract to `datasets/raw/<TaskName>/{imagesTr,labelsTr}`. For Liver, extract ONLY the first 40
(sorted) `liver_XXX.nii.gz` from imagesTr and matching labelsTr (use `tar -tf` to list, then `tar -xf ... <members>`);
skip `imagesTs` (no labels). Files starting with `._` are macOS junk — ignore.
Use `curl -L -C - -o` (resumable) for downloads; run the 27 GB one in background with a log.

## 2. configs/*.yaml
`configs/msd_liver_ct.yaml`:
```
task: liver
raw_dir: datasets/raw/Task03_Liver
npz_dir: datasets/npz/liver
modality: CT
window_width: 400
window_level: 40
labels: {liver: 1, tumor: 2}      # each label evaluated as its own binary target
max_volumes: 40
split: {train: 0.8, val: 0.1, test: 0.1}
min_mask_pixels: 100              # in 1024x1024 space
image_size: 1024
```
`configs/msd_prostate_mri.yaml`: task prostate, modality MRI, percentile_clip [0.5, 99.5],
labels {prostate: [1,2]} (list => union of labels into one binary mask), max_volumes null, rest same.

## 3. data/preprocess_msd.py  (Phase 1.1)
- Load config. List volumes (sorted), apply max_volumes, split 80/10/10 at volume level with
  `random.Random(42).shuffle`, save `datasets/splits.json` as {task: {train:[...], val:[...], test:[...]}} (merge with existing file).
- For each volume: read image+label with nibabel; `np.asanyarray(img.dataobj)`. Prostate images are 4D (T2, ADC) -> take channel 0 (T2).
  MSD arrays are (x,y,z[,c]); iterate over z (last spatial axis). Transpose each slice to (y,x) before resizing so orientation is consistent.
- CT: clip [-160,240] -> rescale to [0,255] uint8. MRI: per-volume percentile clip over non-zero voxels then min-max to [0,255].
- For each z slice and each label target: binary mask; skip target if mask sum (after resizing to 1024) < min_mask_pixels.
  Resize image `cv2.resize(..., (1024,1024), INTER_CUBIC)`, mask INTER_NEAREST.
- Save ONE npz per (volume, slice) containing ALL present targets: `np.savez_compressed(path, img=uint8[1024,1024], gts=uint8[T,1024,1024], names=np.array(target_names), spacing=float32[3] (from header zooms, x,y,z), z=int, orig_shape=int[2])`. Skip the slice entirely if T==0.
  Path: `datasets/npz/<task>/<split>/<volume_id>_z{z:03d}.npz`. volume_id = filename stem without .nii.gz (e.g. liver_012).
- Also write `datasets/npz/<task>/volumes.json`: {volume_id: {split, spacing, orig_shape, n_z_total}} so 3D reconstruction knows the full z-extent.
- Print per-split slice counts and append to `results/tables/dataset_stats.csv`.

## 4. data/dataset.py
`class MSDSliceDataset(Dataset)`: args (npz_dir, split, target_name, jitter=20, image_size=1024, embed_cache_dir=None, out_size=1024).
`__getitem__` -> dict: `image` float32 [3,out_size,out_size] in [0,1] (channel-repeated; if out_size!=1024 resize), `gt` float32 [1,out_size,out_size],
`box` float32 [4] = tight bbox of gt (in out_size coords) with uniform jitter in [0,jitter] applied as in the plan (xmin-=d1, ymin-=d2, xmax+=d3, ymax+=d4, clipped),
`name` str (npz stem), `volume` str, `z` int, `embedding` (only if embed_cache_dir given: load `<embed_cache_dir>/<name>.npy` float16 [256,64,64] -> float32 tensor).
Helpers `get_bbox(mask, jitter, rng)` and `box_to_mask_channel(box, H, W)` (binary 1 inside box) in the same file.
Index npz files once at __init__ (open each to read `names`; cache this index to `<npz_dir>/<split>_index.json` to avoid re-opening next time).

## 5. models/medsam_wrapper.py
```
class MedSAM(nn.Module):
    def __init__(self, checkpoint, model_type="vit_b", freeze_encoder=True, freeze_prompt=True)
    # uses segment_anything.sam_model_registry
    def encode(self, image[B,3,1024,1024]) -> embedding [B,256,64,64]
    def decode(self, embedding, boxes[B,4] in 1024 coords) -> logits [B,1,1024,1024] (low-res 256 upsampled bilinear)
    # decode: prompt_encoder(points=None, boxes=boxes[:,None,:], masks=None); mask_decoder(image_embeddings=emb, image_pe=prompt_encoder.get_dense_pe(), sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=False)
    def forward(self, image, boxes) -> logits
    def point_decode(self, embedding, points[B,2], labels[B]) -> logits   # for center-point ablation
```
Note: SAM expects pixel-normalised input; the official MedSAM inference feeds images in [0,1] directly to `image_encoder` (bypassing `sam.preprocess`). Do the same: call `sam.image_encoder(image)` on [0,1] floats. Use `torch.autocast("cuda", dtype=torch.bfloat16)` for encode in inference. Return fp32 logits.

## 6. models/lora_medsam.py
`class LoRAQKV(nn.Module)`: wraps `blk.attn.qkv` (Linear(768, 2304), frozen). Adds Aq [768,r], Bq [r,768], Av, Bv (A kaiming-uniform, B zeros), scale alpha/r. Forward: base = qkv(x); lq = (x@Aq@Bq)*scale; lv = (x@Av@Bv)*scale; return torch.cat([base[...,:768]+lq, base[...,768:1536], base[...,1536:]+lv], dim=-1).
`apply_lora(sam, rank=4, alpha=4, use_checkpoint=True)`: replace `blk.attn.qkv` in every `sam.image_encoder.blocks`; freeze all params except LoRA params + `sam.mask_decoder`. If use_checkpoint: monkeypatch each block's forward to `torch.utils.checkpoint.checkpoint(orig_forward, x, use_reentrant=False)` when `self.training`.
Provide `lora_state_dict(sam)` (LoRA + decoder params only), `load_lora(sam, path)`, `count_params(sam)` -> (trainable, total). Print both.

## 7. scripts/train_lora.py  (Phase 3.1)
Args: --task, --target (liver|tumor|prostate), --rank 4, --epochs 8, --lr 1e-4, --wd 0.01, --accum 8, --bs 1, --max_steps (optional), --decoder_only.
Loss = BCEWithLogits + soft Dice (on sigmoid, eps 1e-6). AdamW(betas 0.9,0.999). Cosine schedule with 50 warmup optimizer steps.
bf16 autocast on cuda. Log loss every 20 steps to log file, val DSC (slice-level, jitter=0) at end of each epoch (cap val at 200 slices), save best `checkpoints/lora_<task>_<target>.pth` (LoRA+decoder only), log peak VRAM (`torch.cuda.max_memory_allocated`), steps/sec, trainable params to `results/tables/train_lora_<task>_<target>.json`.
`--decoder_only`: freeze encoder, use cached embeddings from `cache/medsam/<task>/<split>/` (dataset embed_cache_dir) and skip encode; saves decoder state only. If CUDA OOM occurs in the first 5 steps of the LoRA run, catch, free memory, and automatically continue in decoder_only mode; write `"fallback": "DECODER_ONLY"` into the json.
Train one target at a time: liver, tumor, prostate => 3 runs. Default --max_steps 1500 for liver/tumor, 600 for prostate.

## 8. scripts/cache_embeddings.py  (Phase 1.2)
Args: --task, --split, --model {sam,medsam}. For every npz in split, compute embedding once (bf16), save float16 .npy to `cache/<model>/<task>/<split>/<npz_stem>.npy`. Skip existing. Batch size 4. One embedding per npz (shared by all targets on that slice).

## 9. evaluation/metrics.py
- `dsc(pred_bool, gt_bool)` (2D or 3D), returns 1.0 if both empty.
- `nsd(pred_bool3d, gt_bool3d, spacing_mm, tol_mm=2.0)` via `surface_distance.compute_surface_distances` + `compute_surface_dice_at_tolerance`. spacing order must match array axes (z,y,x).
- `wilcoxon_table(df, metric, methods, ref_method)` -> DataFrame of p-values (scipy.stats.wilcoxon, zero_method='wilcox', alternative='two-sided'); handles identical arrays (p=1).
- `postprocess_3d(pred_vol_bool, boxes_per_slice)`: zero everything outside the box on each slice; then keep the largest 3D connected component via cc3d (`cc3d.largest_k(labels, k=1)`).
- `reconstruct_volume(slice_preds: dict z->bool[1024,1024], volumes_json_entry)` -> bool volume [n_z_total, H, W] in orig resolution (INTER_NEAREST resize per slice; slices without prediction = empty). GT volume built the same way from npz gts. Note: metrics are computed only over the z-range covered by npz slices (foreground slices) PLUS slices in between; this is the same for all methods so comparisons are fair. Document this.

## 10. scripts/run_benchmark.py  (Phases 1.3, 2.3, 3.2)
Args: --task, --target, --method {sam,medsam,lora,unet,deeplab,medsam_point}, --split test, --jitter 0, --ckpt (for lora/unet/deeplab), --tag "", --save_preds.
Groups test slices by volume, predicts per slice using cached embeddings when method in {sam,medsam,medsam_point} (fall back to on-the-fly encode), stacks to 3D, computes volume DSC & NSD (tol 2 mm, with and without postprocess), plus mean slice DSC and sec/slice.
Appends rows to `results/tables/benchmark.csv`: task,target,method,tag,jitter,volume,dsc3d,nsd3d,dsc3d_pp,nsd3d_pp,slice_dsc_mean,n_slices,sec_per_slice. Before appending, drop existing rows with same (task,target,method,tag,jitter) so re-runs overwrite.
Jitter ablation (Phase 2.3) = loop `--jitter {0,5,10,20,40}` with method medsam & sam, tag jitter; plus `--method medsam_point` (center of gt bbox, label 1). Jitter RNG seeded 42 for reproducibility.
`--save_preds`: save per-slice predictions as uint8 PNG (256x256) + the used box as json to `results/preds/<task>/<target>/<method>/` for overlays.

## 11. models/baselines_unet_deeplab.py + scripts/train_baseline.py  (Phase 2.1/2.2)
`build_model(name, in_channels=4)`: unet -> `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=4, classes=1)`; deeplab -> `smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", in_channels=4, classes=1)`.
Input: 256x256 (dataset out_size=256), 4th channel = box mask (box jittered 0-20 px in 1024 space then scaled). Loss BCE+Dice, AdamW 1e-4 wd 0.01, bs 16, epochs 15 (prostate 25), fp16 autocast + GradScaler, num_workers 4, save best by val DSC to `checkpoints/<name>_<task>_<target>.pth`, write `results/tables/train_<name>_<task>_<target>.json`. For eval in run_benchmark, upsample logits to 1024 bilinear.

## 12. evaluation/sparse_3d_propagation.py  (Phase 2.4)
For each test volume, for k in {1,3,5,10}: take GT boxes on every k-th foreground slice (always including first and last foreground slice), linearly interpolate 4 coords over z (np.interp), run MedSAM (cached embeddings) on every foreground slice, stack, compute 3D DSC/NSD, record wall time and number of human boxes (= number of sampled slices). k=1 = dense prompting. Write `results/tables/sparse3d.csv`: task,target,volume,k,n_boxes,dsc3d,nsd3d,seconds.

## 13. models/auto_prompt_yolo.py  (Phase 3.3, optional)
Export npz train/val to YOLO format (images 640 jpg, labels txt with normalized box per target class), train `yolov8n.pt` 30 epochs imgsz 640 bs 16, predict on test slices (conf 0.25, highest-conf box per class; none => missed slice, empty prediction), feed to MedSAM -> rows in benchmark.csv with method `yolo_medsam`. Record detection recall in `results/tables/yolo_<task>.json`.

## 14. evaluation/visualize_podium_overlays.py  (Phase 4.2)
Reads benchmark.csv (jitter 0, tag "" rows). Produces in `results/figures/`:
- `boxplot_dsc_<task>_<target>.png`, `boxplot_nsd_...`: seaborn boxplot, method order [sam, unet, deeplab, medsam, lora, yolo_medsam (if present)], showfliers, median line, min/max triangles via scatter.
- `podium_<task>_<target>.png`: upper panel dots per volume per method connected by thin grey lines; lower panel stacked bars of rank frequency.
- `overlay_<task>_<target>_<volume>_z<z>.png`: columns = methods, blue box, magenta GT contour (cv2.findContours), yellow alpha 0.4 pred overlay, slice DSC text top-left. Uses `results/preds/`. 3 scenarios: worst-SAM tumor slice, median prostate slice, a tumor slice with >=2 GT components.
- `jitter_curve_<task>_<target>.png`: median DSC vs jitter px for sam & medsam, dashed horizontal line for point prompt.
- `sparse3d_<task>_<target>.png`: bars of seconds and n_boxes vs k, line of DSC vs k.
`scripts/make_report.py`: aggregates everything into `results/REPORT.md` (median [IQR] tables per task/target/method, wilcoxon tables, trainable params, VRAM, train times, figure list, dataset stats, decisions from HANDOFF §3).
