import cc3d
import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances


def dsc(pred, gt):
    """Dice on bool arrays of any shape. Returns 1.0 when both are empty."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / s)


def nsd(pred, gt, spacing_mm, tol_mm=2.0):
    """Normalised surface Dice for 3D bool masks. spacing_mm must be (z, y, x) to match the array axes."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    sd = compute_surface_distances(gt, pred, spacing_mm=tuple(float(s) for s in spacing_mm))
    return float(compute_surface_dice_at_tolerance(sd, tol_mm))


def wilcoxon_table(df, metric, methods, ref_method):
    """df in long format with columns volume, method, <metric>. Two-sided paired Wilcoxon of each method vs ref."""
    wide = df[df["method"].isin(set(methods) | {ref_method})].pivot(index="volume", columns="method", values=metric)
    rows = []
    for m in methods:
        if m not in wide or ref_method not in wide:
            continue
        pair = wide[[ref_method, m]].dropna()
        a, b = pair[ref_method].to_numpy(), pair[m].to_numpy()
        if len(a) == 0 or np.allclose(a, b):
            p = 1.0
        else:
            p = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        rows.append({"method": m, "ref_method": ref_method, "metric": metric, "n_pairs": len(a), "p_value": p})
    return pd.DataFrame(rows, columns=["method", "ref_method", "metric", "n_pairs", "p_value"])


def postprocess_3d(pred_vol, boxes_per_slice):
    """pred_vol bool [Z,H,W]; boxes_per_slice {z: xyxy} in the volume's H,W pixel coords.
    Zeroes everything outside each slice's box (slices without a box are cleared), then keeps the largest 3D component."""
    pred = pred_vol.astype(bool)
    _, H, W = pred.shape
    keep = np.zeros(pred.shape, bool)
    for z, b in boxes_per_slice.items():
        x0, y0 = max(int(np.floor(b[0])), 0), max(int(np.floor(b[1])), 0)
        x1, y1 = min(int(np.ceil(b[2])), W - 1), min(int(np.ceil(b[3])), H - 1)
        keep[z, y0:y1 + 1, x0:x1 + 1] = True
    pred = pred & keep
    labels = cc3d.connected_components(pred.astype(np.uint8))
    if labels.max() == 0:
        return pred
    return cc3d.largest_k(labels, k=1) > 0


def reconstruct_volume(slice_preds, volumes_json_entry):
    """slice_preds {z: bool [1024,1024]} -> bool [n_z_total, H, W] at original resolution (nearest resize per slice).
    Slices without a prediction are empty. The returned volume spans all n_z_total slices; evaluate on the z-range
    covered by npz slices (min z .. max z) for every method so comparisons are fair. GT is built the same way."""
    H, W = volumes_json_entry["orig_shape"]
    vol = np.zeros((volumes_json_entry["n_z_total"], H, W), bool)
    for z, s in slice_preds.items():
        vol[z] = cv2.resize(s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return vol
