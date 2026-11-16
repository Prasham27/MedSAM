import os, sys; from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data.common import get_logger
from evaluation.metrics import dsc

TAB, FIG, PRED, NPZ = ROOT / "results/tables", ROOT / "results/figures", ROOT / "results/preds", ROOT / "datasets/npz"
ORDER = ["sam", "unet", "deeplab", "medsam", "lora", "yolo_medsam"]
# categorical slots from references/palette.md (light values); method order = adjacency order
PAL = {"sam": "#2a78d6", "unet": "#eb6834", "deeplab": "#1baf7a", "medsam": "#eda100", "lora": "#e87ba4",
       "yolo_medsam": "#008300", "medsam_point": "#6250d6"}


def order_methods(ms):
    return [m for m in ORDER if m in ms] + sorted(m for m in ms if m not in ORDER)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ax in fig.get_axes():
        if any(len(t.get_text()) > 8 for t in ax.get_xticklabels()):
            ax.tick_params(axis="x", rotation=30)
            for t in ax.get_xticklabels(): t.set_ha("right")
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def boxplot(d, methods, metric, task, target):
    fig, ax = plt.subplots(figsize=(1.0 + 1.1 * len(methods), 4.2))
    sns.boxplot(data=d, x="method", y=metric, hue="method", order=methods, hue_order=methods, palette=PAL,
                legend=False, showfliers=True, fliersize=3, linewidth=1, width=0.55, ax=ax)
    for i, m in enumerate(methods):
        v = d.loc[d.method == m, metric].to_numpy()
        ax.scatter(i, v.min(), marker="v", s=28, color="0.2", zorder=3)
        ax.scatter(i, v.max(), marker="^", s=28, color="0.2", zorder=3)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, colors="black", lw=2, zorder=4)
    ax.set(xlabel="", ylabel=metric, title=f"{task}/{target}: {metric} per volume (jitter 0, n={d.volume.nunique()})")
    save(fig, f"boxplot_{metric[:3]}_{task}_{target}.png")


def podium(d, methods, task, target, log):
    w = d.pivot_table(index="volume", columns="method", values="dsc3d")[methods].dropna()
    if w.empty:
        return log("WARN podium: no volume has all methods, skipped")
    K = len(methods)
    rk = w.rank(axis=1, ascending=False, method="min").astype(int)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(K)
    for _, row in w.iterrows():
        ax1.plot(x, row.values, color="0.75", lw=0.7, zorder=1)
    for i, m in enumerate(methods):
        ax1.scatter(np.full(len(w), i), w[m], s=26, color=PAL[m], edgecolor="white", linewidth=0.5, zorder=2)
    ax1.set_xticks(x, methods)
    ax1.set(ylabel="3D DSC (per volume)", title=f"Podium {task}/{target} (n={len(w)} volumes)")
    frac = np.array([[(rk[m] == r).mean() for r in range(1, K + 1)] for m in methods])
    left = np.zeros(K)
    for r in range(K):
        ax2.barh(np.arange(K), frac[:, r], left=left, color=plt.cm.Blues(0.9 - 0.6 * r / max(K - 1, 1)),
                 edgecolor="white", linewidth=1, label=f"rank {r + 1}")
        for i in range(K):
            if frac[i, r] >= 0.1:
                ax2.text(left[i] + frac[i, r] / 2, i, f"{frac[i, r]:.0%}", ha="center", va="center", fontsize=8,
                         color="white" if r < 2 else "black")
        left += frac[:, r]
    ax2.set_yticks(np.arange(K), methods)
    ax2.invert_yaxis()
    ax2.set(xlim=(0, 1), xlabel="share of volumes at each rank (1 = best)")
    ax2.legend(ncol=K, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False)
    save(fig, f"podium_{task}_{target}.png")


def load_slice(task, target, method, name):
    p = PRED / task / target / method / f"{name}.png"
    if not p.exists():
        return None, None
    with open(p.with_suffix(".json"), encoding="utf-8") as f:
        box = json.load(f)["box"]  # box saved in 1024 px
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127, np.array(box) / 4.0  # pred is 256 px


def load_npz(task, name, target):
    p = NPZ / task / "test" / f"{name}.npz"
    if not p.exists():
        return None
    with np.load(p) as f:
        names = [str(n) for n in f["names"]]
        if target not in names:
            return None
        img = cv2.resize(f["img"], (256, 256), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(f["gts"][names.index(target)], (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    return img, gt


def render(img, gt, pred, box):
    rgb = np.stack([img] * 3, -1).astype(np.float32)
    if pred is not None:
        rgb[pred] = 0.6 * rgb[pred] + 0.4 * np.array([255, 255, 0])  # yellow, alpha 0.4
    rgb = rgb.astype(np.uint8).copy()
    for c in cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]:
        cv2.drawContours(rgb, [c], -1, (255, 0, 255), 1)  # magenta GT contour
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 0, 255), 1)  # blue prompt box
    return rgb


def overlays(task, target, methods, log):
    methods = [m for m in methods if (PRED / task / target / m).is_dir()]
    names = sorted({p.stem for m in methods for p in (PRED / task / target / m).glob("*.png")})
    if not names:
        return log("WARN overlays: no saved predictions in results/preds, skipped")
    gt_of, dsc_of = {}, {m: {} for m in methods}
    for n in names:
        g = load_npz(task, n, target)
        if g is None:
            continue
        gt_of[n] = g
        for m in methods:
            p, _ = load_slice(task, target, m, n)
            if p is not None:
                dsc_of[m][n] = dsc(p, g[1])
    if not gt_of:
        return log("WARN overlays: no matching npz GT found, skipped")
    scen = []
    if target == "tumor":
        if dsc_of.get("sam"):
            scen.append(("worst-SAM", min(dsc_of["sam"], key=dsc_of["sam"].get)))
        multi = [n for n in sorted(gt_of) if cv2.connectedComponents(gt_of[n][1].astype(np.uint8))[0] - 1 >= 2
                 and any(n in dsc_of[m] for m in methods)]
        if multi:
            scen.append(("multi-component", multi[0]))
    else:
        ref = "medsam" if dsc_of.get("medsam") else methods[0]
        vals = dsc_of[ref]
        if vals:
            med = np.median(list(vals.values()))
            scen.append(("median", min(vals, key=lambda n: abs(vals[n] - med))))
    for tag, n in scen:
        vol, z = n.rsplit("_z", 1)
        img, gt = gt_of[n]
        panels = [(m, render(img, gt, *load_slice(task, target, m, n)), dsc_of[m][n])
                  for m in methods if n in dsc_of[m]]
        fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.1), squeeze=False)
        for ax, (m, im, d_) in zip(axes[0], panels):
            ax.imshow(im)
            ax.set_title(m, fontsize=10)
            ax.axis("off")
            ax.text(4, 16, f"DSC {d_:.2f}", color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
        fig.suptitle(f"{task}/{target} {vol} z{z} ({tag}) - blue box, magenta GT, yellow pred; DSC at 256 px",
                     fontsize=9)
        save(fig, f"overlay_{task}_{target}_{vol}_z{z}.png")


def jitter(base, task, target, log):
    j = base[base.method.isin(["sam", "medsam"]) & base.tag.isin(["", "jitter"])]
    j = j[(j.tag == "jitter") | (j.jitter == 0)]
    if j.jitter.nunique() < 2:
        return log("WARN jitter: no jitter sweep rows, skipped")
    med = j.groupby(["method", "jitter"]).dsc3d.median().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ["sam", "medsam"]:
        s = med[med.method == m].sort_values("jitter")
        ax.plot(s.jitter, s.dsc3d, marker="o", color=PAL[m], label=m, lw=1.5, ms=5)
    p = base[(base.method == "medsam_point") & (base.tag == "")]
    if len(p):
        ax.axhline(p.dsc3d.median(), ls="--", color=PAL["medsam_point"], label="medsam point prompt")
    ax.set(xlabel="box jitter (px, 1024 space)", ylabel="median 3D DSC", title=f"Jitter ablation {task}/{target}")
    ax.legend(frameon=False)
    save(fig, f"jitter_curve_{task}_{target}.png")


def sparse(task, target, log):
    p = TAB / "sparse3d.csv"
    if not p.exists():
        return log("WARN sparse3d: results/tables/sparse3d.csv missing, skipped")
    s = pd.read_csv(p)
    s = s[(s.task == task) & (s.target == target)]
    if s.empty:
        return
    g = s.groupby("k").agg(dsc=("dsc3d", "median"), sec=("seconds", "median"), nb=("n_boxes", "median")).reset_index()
    x = np.arange(len(g))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].bar(x, g.sec, color=PAL["sam"])
    axes[0].set(title="seconds per volume (median)", ylabel="s")
    axes[1].bar(x, g.nb, color=PAL["medsam"])
    axes[1].set(title="human boxes per volume (median)", ylabel="boxes")
    axes[2].plot(x, g.dsc, marker="o", color=PAL["sam"])
    axes[2].set(title="3D DSC (median)", ylabel="DSC", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(x, [str(k) for k in g.k])
        ax.set_xlabel("k (every k-th slice boxed)")
    fig.suptitle(f"Sparse 3D propagation {task}/{target}")
    save(fig, f"sparse3d_{task}_{target}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["liver", "prostate"], default=None)
    ap.add_argument("--target", choices=["liver", "tumor", "prostate"], default=None)
    a = ap.parse_args()
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(TAB / "benchmark.csv")
    df["tag"] = df["tag"].fillna("").astype(str)
    for task, target in df[["task", "target"]].drop_duplicates().itertuples(index=False):
        if (a.task and task != a.task) or (a.target and target != a.target):
            continue
        log = get_logger("visualize_podium_overlays", task).info
        base = df[(df.task == task) & (df.target == target)]
        main_rows = base[(base.tag == "") & (base.jitter == 0)]
        methods = order_methods(list(main_rows.method.unique()))
        log(f"== {task}/{target} methods={methods}")
        if not methods:
            log("WARN no jitter-0 rows, skipped")
            continue
        for metric in ["dsc3d", "nsd3d"]:
            boxplot(main_rows, methods, metric, task, target)
        podium(main_rows, methods, task, target, log)
        overlays(task, target, methods, log)
        jitter(base, task, target, log)
        sparse(task, target, log)


if __name__ == "__main__":
    main()
