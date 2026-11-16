"""YOLOv8n auto-prompting for MedSAM (SPEC §13).

Stages: export (npz -> YOLO dataset), train (yolov8n, 640 px), eval (detect on test slices -> MedSAM -> 3D metrics
-> benchmark.csv rows with method 'yolo_medsam'), all.

Protocol notes:
- The detector runs on every test npz of the task. MedSAM and the metrics are computed on the slices that contain the
  target (the same slice set as scripts/run_benchmark.py), so yolo_medsam differs from medsam only in the box source.
  Detector false positives on target-absent slices are not fed to MedSAM; their rate is reported in yolo_<task>.json.
- Missed slices: empty prediction. Their per-slice json has box [0,0,0,0] and "missed": true (the visualizer needs an
  array there).
"""
import os, sys, json, time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, csv
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.common import get_logger, load_config
from evaluation.metrics import dsc, nsd, postprocess_3d, reconstruct_volume
from models.medsam_wrapper import MedSAM
from scripts.run_benchmark import COLS

SEED = 42
YOLO_IMG = 640
CLASSES = {"prostate": ["prostate"], "liver": ["liver", "tumor"]}
TARGETS = {"prostate": ["prostate"], "liver": ["liver", "tumor"]}
EPOCHS = {"prostate": 30, "liver": 12}
CONF = 0.25
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bbox_xyxy(mask):
    """Tight box of a non-empty bool mask in continuous pixel coords [x0, y0, x1, y1] (x1/y1 exclusive)."""
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def to_yolo_image(img_u8):
    """1024 uint8 slice -> 640 px 3-channel BGR image, round-tripped through JPEG like the training images."""
    im = cv2.resize(img_u8, (YOLO_IMG, YOLO_IMG), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- export
def export(task, log):
    out = ROOT / "datasets/yolo" / task
    yaml_path = out / "data.yaml"
    if yaml_path.exists():
        log(f"export: {yaml_path} exists, skipping")
        return
    classes = CLASSES[task]
    npz_root = ROOT / load_config(task)["npz_dir"]
    counts = {}
    for split in ("train", "val"):
        img_dir, lab_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((npz_root / split).glob("*.npz"))
        if task == "liver":
            files = files[::2]  # every 2nd slice for train and val (time budget)
        n_pos = 0
        for f in tqdm(files, desc=f"export {task}/{split}"):
            with np.load(f) as d:
                img, gts = d["img"], d["gts"]
                names = [str(n) for n in d["names"]]
            cv2.imwrite(str(img_dir / f"{f.stem}.jpg"), to_yolo_image(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for j, n in enumerate(names):
                if n not in classes:
                    continue
                m = gts[j] > 0
                if not m.any():
                    continue
                x0, y0, x1, y1 = bbox_xyxy(m)
                lines.append(f"{classes.index(n)} {(x0 + x1) / 2 / 1024:.6f} {(y0 + y1) / 2 / 1024:.6f} "
                             f"{(x1 - x0) / 1024:.6f} {(y1 - y0) / 1024:.6f}")
            n_pos += bool(lines)
            (lab_dir / f"{f.stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        counts[split] = {"images": len(files), "with_target_box": n_pos}
    data = {"path": out.as_posix(), "train": "images/train", "val": "images/val",
            "names": {i: c for i, c in enumerate(classes)}}
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    log(f"export: wrote {yaml_path} counts={counts}")


# ---------------------------------------------------------------- train
def train(task, log):
    best = ROOT / "results/yolo" / task / "weights" / "best.pt"
    if best.exists():
        log(f"train: {best} exists, skipping")
        return
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(workers=2, data=str(ROOT / "datasets/yolo" / task / "data.yaml"), epochs=EPOCHS[task], imgsz=YOLO_IMG,
                batch=16, project=str(ROOT / "results/yolo"), name=task, exist_ok=True, seed=SEED,
                verbose=False, plots=False)
    log(f"train: done, best weights {best} exists={best.exists()}")


# ---------------------------------------------------------------- eval
def detect(ymodel, classes, img_u8):
    """Highest-confidence box per class, in 1024 px continuous coords. {cls_idx: (conf, box)}."""
    r = ymodel.predict(to_yolo_image(img_u8), conf=CONF, imgsz=YOLO_IMG, verbose=False)[0]
    out = {}
    b = r.boxes
    if b is None or len(b) == 0:
        return out
    cls = b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    xyxy = b.xyxy.cpu().numpy() * (1024.0 / YOLO_IMG)
    for c in range(len(classes)):
        idx = np.where(cls == c)[0]
        if idx.size:
            k = idx[np.argmax(conf[idx])]
            box = np.clip(xyxy[k].astype(np.float64), 0, 1024)
            if box[2] > box[0] and box[3] > box[1]:
                out[c] = (float(conf[k]), box)
    return out


def save_pred(task, target, name, z, vid, pred, box, missed):
    d = ROOT / "results/preds" / task / target / "yolo_medsam"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{name}.png"), cv2.resize(pred.astype(np.uint8) * 255, (256, 256),
                                                   interpolation=cv2.INTER_NEAREST))
    with open(d / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"volume": vid, "z": z, "box": [float(v) for v in box], "missed": bool(missed)}, f)


def medsam_predict(medsam, cache_dir, stem, img_u8, box1024):
    """MedSAM decode for one slice/box using the cached test embedding when present (as run_benchmark does)."""
    p = cache_dir / f"{stem}.npy"
    with torch.inference_mode():
        if p.exists():
            emb = torch.from_numpy(np.load(p).astype(np.float32)).to(DEV)[None]
        else:
            x = torch.from_numpy(np.repeat(img_u8[None].astype(np.float32) / 255.0, 3, axis=0)).to(DEV)[None]
            emb = medsam.encode(x)
        box_t = torch.as_tensor(box1024, dtype=torch.float32, device=DEV)[None]
        return (medsam.decode(emb, box_t)[0, 0] > 0).cpu().numpy()


def evaluate(task, log):
    from ultralytics import YOLO
    best = ROOT / "results/yolo" / task / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"{best} missing; run --stage train first")
    classes, targets = CLASSES[task], TARGETS[task]
    cfg = load_config(task)
    npz_root = ROOT / cfg["npz_dir"]
    with open(npz_root / "volumes.json", encoding="utf-8") as f:
        volumes = json.load(f)
    ymodel = YOLO(str(best))
    medsam = MedSAM(str(ROOT / "checkpoints/medsam_vit_b.pth"), model_type="vit_b").to(DEV).eval()
    cache_dir = ROOT / "cache/medsam" / task / "test"

    P = {t: {} for t in targets}
    G = {t: {} for t in targets}
    BX = {t: {} for t in targets}
    SD = {t: {} for t in targets}
    T = {t: {} for t in targets}
    stats = {t: {"n_target": 0, "n_found": 0, "ious": [], "ious_all": []} for t in targets}
    nontarget = {t: {"n": 0, "fp": 0} for t in targets}

    files = sorted((npz_root / "test").glob("*.npz"))
    for f in tqdm(files, desc=f"yolo+medsam {task} test"):
        with np.load(f) as d:
            img, gts, z = d["img"], d["gts"], int(d["z"])
            names = [str(n) for n in d["names"]]
        vid = f.stem.rsplit("_z", 1)[0]
        t0 = time.perf_counter()
        det = detect(ymodel, classes, img)
        t_det = time.perf_counter() - t0
        for t in targets:
            ci = classes.index(t)
            if t not in names:
                nontarget[t]["n"] += 1
                nontarget[t]["fp"] += int(ci in det)
                continue
            gt = gts[names.index(t)] > 0
            gt_box = bbox_xyxy(gt)
            st = stats[t]
            st["n_target"] += 1
            t1 = time.perf_counter()
            if ci in det:
                conf, box = det[ci]
                st["n_found"] += 1
                i = iou(box, gt_box)
                st["ious"].append(i)
                st["ious_all"].append(i)
                pred = medsam_predict(medsam, cache_dir, f.stem, img, box.astype(np.float32))
                BX[t].setdefault(vid, {})[z] = box
            else:
                st["ious_all"].append(0.0)
                box = np.zeros(4)
                pred = np.zeros((1024, 1024), bool)
            T[t].setdefault(vid, []).append(t_det + time.perf_counter() - t1)
            P[t].setdefault(vid, {})[z] = pred
            G[t].setdefault(vid, {})[z] = gt
            SD[t].setdefault(vid, []).append(dsc(pred, gt))
            save_pred(task, t, f.stem, z, vid, pred, box, ci not in det)

    rows_all = {}
    for t in targets:
        rows = []
        for vid in sorted(P[t]):
            vj = volumes[vid]
            H, W = vj["orig_shape"]
            sx, sy = W / 1024, H / 1024
            sp = vj["spacing"]
            zyx = [sp[2], sp[1], sp[0]]
            pred3 = reconstruct_volume(P[t][vid], vj)
            gt3 = reconstruct_volume(G[t][vid], vj)
            boxes = {z: b * np.array([sx, sy, sx, sy]) for z, b in BX[t].get(vid, {}).items()}
            pp = postprocess_3d(pred3, boxes)
            r = {"task": task, "target": t, "method": "yolo_medsam", "tag": "", "jitter": 0, "volume": vid,
                 "dsc3d": dsc(pred3, gt3), "nsd3d": nsd(pred3, gt3, zyx, tol_mm=2.0),
                 "dsc3d_pp": dsc(pp, gt3), "nsd3d_pp": nsd(pp, gt3, zyx, tol_mm=2.0),
                 "slice_dsc_mean": float(np.mean(SD[t][vid])), "n_slices": len(SD[t][vid]),
                 "sec_per_slice": float(np.sum(T[t][vid]) / len(T[t][vid]))}
            rows.append(r)
            log(f"{t} {vid}: dsc3d {r['dsc3d']:.4f} nsd3d {r['nsd3d']:.4f} dsc3d_pp {r['dsc3d_pp']:.4f} "
                f"slices {r['n_slices']}")
        rows_all[t] = rows

    # benchmark.csv: drop existing yolo_medsam rows for these task/targets (tag '', jitter 0), append new rows
    out = ROOT / "results/tables/benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            keep = [r for r in csv.DictReader(fh)
                    if not (r["task"] == task and r["target"] in targets and r["method"] == "yolo_medsam"
                            and r["tag"] == "" and r["jitter"] == "0")]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(keep)
        for t in targets:
            w.writerows(rows_all[t])
    log(f"benchmark.csv: wrote {sum(len(v) for v in rows_all.values())} yolo_medsam rows")

    summary = {"task": task, "detector": "yolov8n", "conf": CONF, "epochs": EPOCHS[task],
               "best_weights": str(best.relative_to(ROOT)).replace("\\", "/"), "targets": {}}
    for t in targets:
        st, nt = stats[t], nontarget[t]
        summary["targets"][t] = {
            "n_test_target_slices": st["n_target"], "n_detected": st["n_found"],
            "detection_recall": st["n_found"] / max(st["n_target"], 1),
            "mean_iou_detected": float(np.mean(st["ious"])) if st["ious"] else None,
            "mean_iou_all_target_slices": float(np.mean(st["ious_all"])) if st["ious_all"] else None,
            "n_nontarget_test_slices": nt["n"], "nontarget_false_positive_slices": nt["fp"],
            "nontarget_false_positive_rate": nt["fp"] / max(nt["n"], 1),
            "median_dsc3d": float(np.median([r["dsc3d"] for r in rows_all[t]])),
            "median_dsc3d_pp": float(np.median([r["dsc3d_pp"] for r in rows_all[t]])),
        }
        log(f"{t}: recall {summary['targets'][t]['detection_recall']:.4f} "
            f"mean IoU(det) {summary['targets'][t]['mean_iou_detected']} "
            f"median dsc3d {summary['targets'][t]['median_dsc3d']:.4f}")
    with open(ROOT / "results/tables" / f"yolo_{task}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["liver", "prostate"])
    ap.add_argument("--stage", default="all", choices=["export", "train", "eval", "all"])
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log = get_logger("yolo", a.task).info
    log(f"args {vars(a)} device={DEV}")
    stages = ["export", "train", "eval"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"== stage {s}")
        if s == "export":
            export(a.task, log)
        elif s == "train":
            train(a.task, log)
        else:
            evaluate(a.task, log)
    log("done")


if __name__ == "__main__":
    main()
