"""Appendiks-grid: ALLE 115 test-billeder med sandsynligheds-heatmap overlagt.

3 kolonner x 5 raekker per side -> 8 sider. Sorteret efter havn + dato.
Hver celle: satellitbillede med inferno-heatmap (alpha ~ sandsynlighed) + Dice.
Samme checkpoint/kodesti som 1b6c343-tjekkene (valideres per case mod artefakten).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT))
from regen_panels_tau04 import ART, TAU, git_csv, load_model, predict  # noqa: E402

OUT_DIR = AUDIT / "panels_tau04_bundle"
BG = "#fafaf7"
INK = "#24292f"
MUTED = "#5b6b78"
COLS, ROWS = 3, 5
PER_PAGE = COLS * ROWS

INFERNO = mpl.colormaps["inferno"]


def label(port_id: str) -> str:
    special = {"qing_dao": "Qingdao", "new_york_new_jersey": "New York New Jersey",
               "long_beach": "Long Beach & LA", "ningbozhoushan": "Ningbo-Zhoushan",
               "antwerpbrugges": "Antwerp-Bruges", "hamborg": "Hamburg",
               "da_lian": "Da Lian", "ho_chi_minh_city": "Ho Chi Minh City"}
    return special.get(port_id, port_id.replace("_", " ").title())


def heat_overlay(rgb: np.ndarray, prob: np.ndarray) -> np.ndarray:
    heat = (INFERNO(prob)[:, :, :3] * 255.0)
    alpha = np.clip(prob * 0.72, 0.0, 0.72)[..., None]
    return (rgb.astype(np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)


def main() -> None:
    te = git_csv(f"{ART}/raw/test/predictions.csv")
    te = te.sort_values(["port_id", "timestamp"]).reset_index(drop=True)
    model = load_model()
    print(f"Model indlæst; {len(te)} test-cases")

    n_pages = int(np.ceil(len(te) / PER_PAGE))
    max_diff = 0.0
    with torch.no_grad():
        for page in range(n_pages):
            chunk = te.iloc[page * PER_PAGE:(page + 1) * PER_PAGE]
            fig, axes = plt.subplots(ROWS, COLS, figsize=(8.7, 14.6))
            fig.subplots_adjust(left=0.012, right=0.988, top=0.945, bottom=0.018,
                                hspace=0.16, wspace=0.04)
            fig.text(0.025, 0.985,
                     f"Test-set probability overlays — page {page + 1} of {n_pages}",
                     ha="left", va="top", fontsize=13, weight="bold", color=INK)
            fig.text(0.025, 0.966,
                     "Selected U-Net (epoch 96, percentile 2/98); inferno heatmap opacity scales with predicted probability; Dice at tau=0.4.",
                     ha="left", va="top", fontsize=7.6, color=MUTED)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            for ax, (_, row) in zip(axes.flat, chunk.iterrows()):
                rgb, prob, gt = predict(model, row["key"])
                pred = (prob >= TAU).astype(np.float32)
                inter = float((pred * gt).sum())
                denom = float(pred.sum() + gt.sum())
                d = 2 * inter / denom if denom else 1.0
                max_diff = max(max_diff, abs(d - float(row["dice"])))
                ax.imshow(heat_overlay(rgb, prob))
                ts = str(row["timestamp"])[:8]
                date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                ax.set_title(f"{label(row['port_id'])} {row['patch_id']} · {date} · Dice {row['dice']:.2f}",
                             fontsize=6.8, color=INK, pad=2.5)
                ax.axis("off")
            path = OUT_DIR / f"test_set_heatmap_overlays_page_{page + 1}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=0.10)
            plt.close(fig)
            print(path)
    print(f"Validering: max |Dice_genberegnet - Dice_artefakt| = {max_diff:.2e}")


if __name__ == "__main__":
    main()
