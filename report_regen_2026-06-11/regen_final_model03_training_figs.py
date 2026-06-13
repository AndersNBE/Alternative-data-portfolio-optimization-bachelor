from __future__ import annotations

import io
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
COMMIT = "cb031ac4deffdb3a91aabb4b5f61186ca9e3ab34"
METRICS_REL = "final_runs/full_gode_havnebilleder_tau06_final_2026_06_05/model/metrics.csv"
OUT_DIR = ROOT / "FinalFigures"


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(REPO / METRICS_REL)


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color="#d9dee7", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.tick_params(colors="#374151")


def save_loss_plot(metrics: pd.DataFrame, best_epoch: int) -> Path:
    out_path = OUT_DIR / "final_model03_train_loss_vs_val_loss.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
    ax.plot(metrics["epoch"], metrics["train_loss"], color="#2563eb", linewidth=2.2, label="Training loss")
    ax.plot(metrics["epoch"], metrics["val_loss"], color="#dc2626", linewidth=2.2, label="Validation loss")
    ax.axvline(best_epoch, color="#111827", linestyle="--", linewidth=1.2, label=f"Selected checkpoint: epoch {best_epoch}")
    best_row = metrics.loc[metrics["epoch"] == best_epoch].iloc[0]
    ax.scatter([best_epoch], [best_row["val_loss"]], color="#111827", s=34, zorder=5)
    ax.set_title("Final Model 03 training and validation loss", fontsize=13, pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metrics_plot(metrics: pd.DataFrame, best_epoch: int) -> Path:
    out_path = OUT_DIR / "final_model03_validation_metrics_over_epochs.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
    series = [
        ("val_dice_thr_0_40", "Validation Dice at tau = 0.4", "#16a34a"),
        ("val_iou", "Validation IoU", "#7c3aed"),
        ("val_precision", "Validation precision", "#ea580c"),
        ("val_recall", "Validation recall", "#0891b2"),
    ]
    for column, label, color in series:
        ax.plot(metrics["epoch"], metrics[column], linewidth=2.0, label=label, color=color)
    ax.axvline(best_epoch, color="#111827", linestyle="--", linewidth=1.2, label=f"Selected checkpoint: epoch {best_epoch}")
    best_row = metrics.loc[metrics["epoch"] == best_epoch].iloc[0]
    ax.scatter([best_epoch], [best_row["val_dice_thr_0_40"]], color="#111827", s=34, zorder=5)
    ax.set_title("Final Model 03 validation metrics", fontsize=13, pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 0.9)
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    best_epoch = int(metrics.loc[metrics["val_loss"].idxmin(), "epoch"])
    loss_path = save_loss_plot(metrics, best_epoch)
    metric_path = save_metrics_plot(metrics, best_epoch)
    print(loss_path)
    print(metric_path)


if __name__ == "__main__":
    main()
