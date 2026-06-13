"""Phase C: Segmenteringsfigurer fra 1b6c343-artefakten (supervised tau=0.4 checks).

Genererer (samme filnavne som i Overleaf):
  SegmentationCleaned/selected_model_cleaned_port_avg_dice_test_sorted_v1.png
  SegmentationCleaned/selected_model_cleaned_port_avg_dice_val_vs_test_common_v1.png
  SegmentationCleaned/selected_model_cleaned_val_test_dumbbell_v1.png
  SegmentationCleaned/selected_model_cleaned_threshold_metrics_smallmultiples_v2.png
  selected_model_val_roi_comparison_metrics.png
  roi_10_port_delta_dice_sorted.png
  roi_10_coverage_vs_delta_dice.png
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent / "Bachelor-portfolio"
COMMIT = "1b6c343467e7ea73e13554e65316a2fb3b642694"
ART = "report_artifacts/supervised_tau04_checks_cb031ac_8a44dd0"
OUT_DIR = AUDIT / "segmentation_tau04_bundle"
SOURCE = "Source: 1b6c343 supervised_tau04_checks_cb031ac_8a44dd0 (best.pt epoch 96, percentile 2/98, tau=0.4)."

BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
GRID = "#d6dee6"
AXIS = "#4a5560"
BLUE = "#2b9fd6"
BLUE_DARK = "#0b4775"
ORANGE = "#d9730d"
RED = "#d64541"
GREEN = "#00a878"
GOLD = "#edae13"
PINK = "#d889b8"
PINK_DARK = "#9b3e5d"
NEUTRAL = "#66717c"

PORT_LABELS = {
    "abu_dhabi": "Abu Dhabi", "algeciras": "Algeciras", "antwerpbrugges": "Antwerpbrugges",
    "balboa": "Balboa", "bremen": "Bremen", "busan": "Busan", "cai_mep": "Cai Mep",
    "colombo": "Colombo", "colon": "Colon", "da_lian": "Da Lian", "dongguan": "Dongguan",
    "guangxi_beibu": "Guangxi Beibu", "guangzhou": "Guangzhou", "hai_phong": "Hai Phong",
    "hamborg": "Hamburg", "ho_chi_minh_city": "Ho Chi Minh City", "hong_kong": "Hong Kong",
    "houston": "Houston", "jawaharal_nehru": "Jawaharal Nehru", "jebel_ali": "Jebel Ali",
    "kaohsiung": "Kaohsiung", "laem_chabang": "Laem Chabang", "lianyungang": "Lianyungang",
    "long_beach": "Long Beach & Los Angeles", "manila": "Manila", "mundra": "Mundra",
    "new_york_new_jersey": "New York New Jersey", "ningbozhoushan": "Ningbozhoushan",
    "piraeus": "Piraeus", "port_klang": "Port Klang", "qing_dao": "Qingdao",
    "rizhao": "Rizhao", "rotterdam": "Rotterdam", "santos": "Santos", "savannah": "Savannah",
    "shanghai": "Shanghai", "shenzhen": "Shenzhen", "singapore": "Singapore",
    "suzhou": "Suzhou", "tanger_med": "Tanger Med", "tanjung_pelepas": "Tanjung Pelepas",
    "tanjung_perak": "Tanjung Perak", "tanjung_priok": "Tanjung Priok", "tianjin": "Tianjin",
    "tokyo": "Tokyo", "valencia": "Valencia", "xiamen": "Xiamen", "yantai": "Yantai",
    "yingkou": "Yingkou",
}


def git_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO / rel)


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
    ax.tick_params(length=3.5, color=AXIS, pad=5)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(path)


def label(p: str) -> str:
    return PORT_LABELS.get(p, p.replace("_", " ").title())


def plot_port_test_sorted(te: pd.DataFrame) -> None:
    g = te.groupby("port_id").agg(d=("dice", "mean"), n=("dice", "size")).sort_values("d")
    fig = plt.figure(figsize=(9.6, 0.255 * len(g) + 2.2))
    ax = fig.add_axes([0.22, 0.05, 0.72, 0.88])
    fig.text(0.05, 0.985, "Per-port test Dice", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.05, 0.962, "Test set, selected U-Net at threshold 0.4. Labels show average Dice and number of samples.",
             ha="left", va="top", fontsize=9, color=MUTED)
    style_axis(ax, "x")
    y = np.arange(len(g))
    colors = [RED if v < 0.5 else BLUE_DARK for v in g["d"]]
    ax.hlines(y, 0, g["d"], color=GRID, linewidth=1.4)
    ax.scatter(g["d"], y, s=52, color=colors, zorder=4)
    for yy, (_, row) in zip(y, g.iterrows()):
        ax.text(row["d"] + 0.012, yy, f"{row['d']:.3f} (n={int(row['n'])})", va="center", ha="left", fontsize=7.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([label(p) for p in g.index], fontsize=8.0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average test Dice at threshold 0.4", fontsize=9.4)
    ax.invert_yaxis()
    fig.text(0.05, 0.012, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "SegmentationCleaned/selected_model_cleaned_port_avg_dice_test_sorted_v1.png")


def plot_port_val_vs_test(va: pd.DataFrame, te: pd.DataFrame) -> None:
    pv = va.groupby("port_id").dice.mean()
    pt = te.groupby("port_id").dice.mean()
    common = sorted(pt.index.intersection(pv.index), key=lambda p: pt[p])
    fig = plt.figure(figsize=(9.6, 0.27 * len(common) + 2.4))
    ax = fig.add_axes([0.24, 0.055, 0.70, 0.86])
    fig.text(0.05, 0.985, "Per-port Dice: validation vs test", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.05, 0.962,
             f"Selected U-Net at threshold 0.4. {len(common)} ports appear in both splits; lines connect validation and test averages.",
             ha="left", va="top", fontsize=9, color=MUTED)
    style_axis(ax, "x")
    y = np.arange(len(common))
    for yy, p in zip(y, common):
        ax.plot([pv[p], pt[p]], [yy, yy], color=GRID, linewidth=2.1, zorder=2)
    ax.scatter([pv[p] for p in common], y, s=44, color=BLUE, zorder=4, label="Validation")
    ax.scatter([pt[p] for p in common], y, s=44, color=ORANGE, zorder=5, label="Test")
    ax.set_yticks(y)
    ax.set_yticklabels([label(p) for p in common], fontsize=8.0)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average Dice at threshold 0.4", fontsize=9.4)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8.6)
    lower = sum(pt[p] < pv[p] for p in common)
    fig.text(0.05, 0.030, f"{lower} of {len(common)} ports have lower test Dice than validation Dice; {len(common) - lower} have higher.",
             ha="left", va="bottom", fontsize=8.2, color=MUTED)
    fig.text(0.05, 0.012, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "SegmentationCleaned/selected_model_cleaned_port_avg_dice_val_vs_test_common_v1.png")


def plot_val_test_dumbbell(summary: pd.DataFrame) -> None:
    raw = summary[summary["run_type"] == "raw"].set_index("split")
    metrics = [("Dice", "avg_dice"), ("IoU", "avg_iou"), ("Precision", "avg_precision"), ("Recall", "avg_recall")]
    fig = plt.figure(figsize=(10.4, 6.2))
    ax = fig.add_axes([0.13, 0.18, 0.82, 0.66])
    fig.text(0.05, 0.955, "Validation and test metrics", ha="left", va="top", fontsize=17, weight="bold", color=INK)
    fig.text(0.05, 0.915, "Validation and test sets, selected U-Net at threshold 0.4.", ha="left", va="top", fontsize=9.4, color=MUTED)
    style_axis(ax, "x")
    y = np.arange(len(metrics))[::-1]
    for yy, (_, col) in zip(y, metrics):
        v, t = raw.loc["val", col], raw.loc["test", col]
        ax.plot([t, v], [yy, yy], color="#c3ccd4", linewidth=4.4, zorder=2, solid_capstyle="round")
        ax.scatter([v], [yy], s=120, color=BLUE_DARK, zorder=4)
        ax.scatter([t], [yy], s=120, color=ORANGE, zorder=4)
        ax.annotate(f"{v:.3f}", xy=(v, yy), xytext=(6, 9), textcoords="offset points", fontsize=8.6, color=BLUE_DARK)
        ax.annotate(f"{t:.3f}", xy=(t, yy), xytext=(-6, -15), textcoords="offset points", ha="right", fontsize=8.6, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in metrics], fontsize=10.6)
    ax.set_xlabel("Metric value", fontsize=9.6)
    ax.set_xlim(0.4, 0.85)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="None", color=BLUE_DARK, markersize=8, label="Validation"),
               plt.Line2D([0], [0], marker="o", linestyle="None", color=ORANGE, markersize=8, label="Test")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.16), ncols=2, fontsize=9)
    fig.text(0.05, 0.02, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "SegmentationCleaned/selected_model_cleaned_val_test_dumbbell_v1.png")


def plot_threshold_smallmultiples(sweep: pd.DataFrame) -> None:
    sw = sweep[(sweep["threshold"] >= 0.049) & (sweep["threshold"] <= 0.601)].copy()
    panels = [("Dice", "avg_dice", BLUE), ("IoU", "avg_iou", GREEN),
              ("Precision", "avg_precision", GOLD), ("Recall", "avg_recall", PINK)]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.09, hspace=0.42, wspace=0.22)
    fig.text(0.045, 0.965, "Selected model threshold sensitivity", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.045, 0.925, "Validation set threshold sweep. Each panel shows one metric; the vertical line marks the selected operating threshold 0.4.",
             ha="left", va="top", fontsize=9, color=MUTED)
    for ax, (title, col, color) in zip(axes.flat, panels):
        style_axis(ax)
        ax.plot(sw["threshold"], sw[col], color=color, linewidth=1.7, marker="o", markersize=3.4)
        ax.axvline(0.4, color=INK, linewidth=1.0)
        ax.set_title(title, fontsize=11, loc="left", color=INK, weight="bold")
        ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax.tick_params(labelsize=7.4)
    fig.text(0.045, 0.015, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "SegmentationCleaned/selected_model_cleaned_threshold_metrics_smallmultiples_v2.png")


def plot_roi_comparison(summary: pd.DataFrame) -> None:
    val = summary[summary["split"] == "val"].set_index("run_type")
    metrics = [("Dice", "avg_dice"), ("IoU", "avg_iou"), ("Precision", "avg_precision"), ("Recall", "avg_recall")]
    fig = plt.figure(figsize=(9.6, 5.6))
    ax = fig.add_axes([0.09, 0.13, 0.87, 0.66])
    fig.text(0.05, 0.955, "Validation metrics: raw vs ROI10", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.05, 0.915, "Selected U-Net at threshold 0.4. ROI buffer b=10 px. ROI10 slightly raises Dice, IoU and precision, and slightly lowers recall.",
             ha="left", va="top", fontsize=9, color=MUTED)
    style_axis(ax)
    x = np.arange(len(metrics))
    w = 0.34
    raw_vals = [val.loc["raw", c] for _, c in metrics]
    roi_vals = [val.loc["roi10", c] for _, c in metrics]
    ax.bar(x - w / 2, raw_vals, width=w, color=BLUE, alpha=0.9, label="Raw prediction")
    ax.bar(x + w / 2, roi_vals, width=w, color=GREEN, alpha=0.9, label="ROI10 filtered")
    for xx, (rv, ov) in zip(x, zip(raw_vals, roi_vals)):
        ax.text(xx - w / 2, rv + 0.008, f"{rv:.4f}", ha="center", va="bottom", fontsize=8.0, weight="bold")
        ax.text(xx + w / 2, ov + 0.008, f"{ov:.4f}", ha="center", va="bottom", fontsize=8.0, weight="bold")
        delta = ov - rv
        ax.text(xx + w / 2, ov / 2, f"{'+' if delta >= 0 else ''}{delta:.4f}", ha="center", va="center",
                fontsize=7.6, color="white", weight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in metrics], fontsize=10)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Metric value")
    ax.legend(loc="upper left", fontsize=8.8)
    fig.text(0.05, 0.02, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "selected_model_val_roi_comparison_metrics.png")


def plot_roi_port_delta(roi_port: pd.DataFrame) -> None:
    g = roi_port[(roi_port["scope"] == "port") & (roi_port["delta_dice"].abs() >= 0.0001)].sort_values("delta_dice")
    fig = plt.figure(figsize=(9.0, 0.34 * len(g) + 2.4))
    ax = fig.add_axes([0.26, 0.07, 0.68, 0.84])
    fig.text(0.05, 0.98, "Port-level Dice change from ROI10", ha="left", va="top", fontsize=15, weight="bold", color=INK)
    fig.text(0.05, 0.952, "Validation set at threshold 0.4. Ports with |ΔDice| < 0.0001 are omitted; mean port-level change is +0.0007.",
             ha="left", va="top", fontsize=8.8, color=MUTED)
    style_axis(ax, "x")
    y = np.arange(len(g))
    colors = [PINK_DARK if v < 0 else BLUE_DARK for v in g["delta_dice"]]
    ax.barh(y, g["delta_dice"], color=colors, alpha=0.9, height=0.62)
    for yy, (_, row) in zip(y, g.iterrows()):
        v = row["delta_dice"]
        ax.text(v + (0.0008 if v >= 0 else -0.0008), yy, f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.6, color=INK)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([label(p) for p in g["port_id"]], fontsize=8.2)
    ax.set_xlabel("$\\Delta$ Dice (ROI10 $-$ raw)", fontsize=9.4)
    lim = max(abs(g["delta_dice"].min()), g["delta_dice"].max()) * 1.32
    ax.set_xlim(-lim, lim)
    fig.text(0.05, 0.013, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "roi_10_port_delta_dice_sorted.png")


def plot_roi_coverage_vs_delta(roi_port: pd.DataFrame) -> None:
    g = roi_port[roi_port["scope"] == "port"].copy()
    fig = plt.figure(figsize=(9.4, 6.0))
    ax = fig.add_axes([0.10, 0.12, 0.86, 0.70])
    fig.text(0.05, 0.955, "ROI coverage vs Dice change", ha="left", va="top", fontsize=16, weight="bold", color=INK)
    fig.text(0.05, 0.915, "Validation set at threshold 0.4, ROI buffer b=10 px. Each point is a port; marker size reflects the number of validation cases.",
             ha="left", va="top", fontsize=9, color=MUTED)
    style_axis(ax)
    ax.scatter(g["avg_roi_coverage_ratio"], g["delta_dice"], s=22 + 26 * g["num_cases"],
               color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=AXIS, linewidth=1.0, linestyle=(0, (3, 3)))
    for slug in ("port_klang", "qing_dao", "yantai", "kaohsiung"):
        row = g[g["port_id"] == slug]
        if len(row):
            r = row.iloc[0]
            ax.annotate(label(slug), xy=(r["avg_roi_coverage_ratio"], r["delta_dice"]),
                        xytext=(7, 4), textcoords="offset points", fontsize=8.0, color=INK)
    ax.set_xlabel("Average ROI coverage ratio", fontsize=9.6)
    ax.set_ylabel("$\\Delta$ Dice (ROI10 $-$ raw)", fontsize=9.6)
    fig.text(0.05, 0.02, SOURCE, ha="left", va="bottom", fontsize=7.6, color=MUTED)
    save(fig, "roi_10_coverage_vs_delta_dice.png")


def main() -> None:
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG, "legend.frameon": False,
    })
    te = git_csv(f"{ART}/raw/test/predictions.csv")
    va = git_csv(f"{ART}/raw/val/predictions.csv")
    sweep = git_csv(f"{ART}/raw/val/threshold_sweep.csv")
    summary = git_csv(f"{ART}/summary_metrics.csv")
    roi_port = git_csv(f"{ART}/roi10/val/roi_summary.csv")
    plot_port_test_sorted(te)
    plot_port_val_vs_test(va, te)
    plot_val_test_dumbbell(summary)
    plot_threshold_smallmultiples(sweep)
    plot_roi_comparison(summary)
    plot_roi_port_delta(roi_port)
    plot_roi_coverage_vs_delta(roi_port)


if __name__ == "__main__":
    main()
