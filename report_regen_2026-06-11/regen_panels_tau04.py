"""Phase D: Kvalitative prob-paneler, high/low-performance figurer og ROI-story
ved tau=0.4 med den verificerede best.pt (SHA 876d7bb8..., epoch 96, percentile 2/98).

Kode-sti: cb031ac model.py + roi.py (udtrukket lokalt), percentile-normalisering
kopieret 1:1 fra cb031ac dataset.py. Valideres mod 1b6c343 predictions.csv.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT))
import importlib.util

spec = importlib.util.spec_from_file_location("unet_model", AUDIT / "unet_model_cb031ac.py")
unet_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_model)
spec2 = importlib.util.spec_from_file_location("unet_roi", AUDIT / "unet_roi_cb031ac.py")
unet_roi = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(unet_roi)

REPO = AUDIT.parent / "Bachelor-portfolio"
DATA_ENV = "FINAL_SEGMENTATION_IMAGE_ROOT"
COMMIT = "1b6c343467e7ea73e13554e65316a2fb3b642694"
ART = "report_artifacts/supervised_tau04_checks_cb031ac_8a44dd0"
OUT_DIR = AUDIT / "panels_tau04_bundle"
TAU = 0.4
BG = "#fafaf7"
INK = "#24292f"

PANEL_CASES = [
    "jebel_ali__P2__L1C__20230607T064629Z__eoCC0.000__CRNA__ND0.000__512px",
    "abu_dhabi__P1__L1C__20231217T070309Z__eoCC0.000__CRNA__ND0.000__512px",
    "rotterdam__P2__L1C__20220811T105631Z__eoCC0.000__CRNA__ND0.000__512px",
    "tanjung_perak__P1__L1C__20231104T023851Z__eoCC45.897__CRNA__ND0.000__512px",
    "mundra__P2__L1C__20231017T054829Z__eoCC0.743__CRNA__ND0.000__512px",
]
HIGH_PORTS = ["algeciras", "rotterdam", "tanger_med"]
LOW_PORTS = ["hai_phong", "qing_dao", "singapore"]
GUANGXI = "guangxi_beibu__P2__L1C__20221012T031651Z__eoCC0.000__CRNA__ND0.000__512px"
PORT_TITLE = {"algeciras": "Algeciras", "rotterdam": "Rotterdam", "tanger_med": "Tanger Med",
              "hai_phong": "Hai Phong", "qing_dao": "Qingdao", "singapore": "Singapore"}


def git_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REPO / rel)


def segmentation_image_root() -> Path:
    value = os.environ.get(DATA_ENV)
    if not value:
        raise SystemExit(
            f"Set {DATA_ENV} to the frozen segmentation image root containing images/ and masks/."
        )
    root = Path(value).expanduser()
    missing = [name for name in ("images", "masks") if not (root / name).is_dir()]
    if missing:
        raise SystemExit(f"{DATA_ENV} is missing required subdirectories: {', '.join(missing)}")
    return root


def percentile_stretch(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for channel in range(arr.shape[2]):
        values = arr[:, :, channel].astype(np.float32)
        lo = float(np.percentile(values, low))
        hi = float(np.percentile(values, high))
        if hi <= lo:
            out[:, :, channel] = values
        else:
            out[:, :, channel] = np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def load_model() -> torch.nn.Module:
    ckpt = torch.load(AUDIT / "best.pt", map_location="cpu", weights_only=False)
    model = unet_model.UNet(
        norm_type=ckpt.get("norm_type", "group"),
        group_norm_groups=int(ckpt.get("group_norm_groups", 8)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, data_root: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returnerer (rgb_uint8, prob_map, gt_mask_01)."""
    img = Image.open(data_root / "images" / f"{key}.png").convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    norm = percentile_stretch(rgb)
    tensor = torch.from_numpy(np.transpose(norm.astype(np.float32) / 255.0, (2, 0, 1)))[None]
    prob = torch.sigmoid(model(tensor))[0, 0].numpy()
    mask_path = data_root / "masks" / f"{key}_mask.png"
    gt = (np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127).astype(np.float32)
    return rgb, prob, gt


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2 * inter / denom if denom else 1.0


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(237, 201, 19), alpha=0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    for c in range(3):
        out[:, :, c][m] = (1 - alpha) * out[:, :, c][m] + alpha * color[c]
    return out.astype(np.uint8)


def prob_panel(model, data_root: Path, key: str) -> None:
    rgb, prob, gt = predict(model, data_root, key)
    pred = (prob >= TAU).astype(np.float32)
    fig, axes = plt.subplots(1, 5, figsize=(16.4, 3.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.02, wspace=0.025)
    titles = ["Image", "Ground Truth", f"Prediction ($\\tau$={TAU})", "Prob. heatmap", "Overlay"]
    panels = [rgb, gt, pred, prob, overlay(rgb, pred)]
    cmaps = [None, "gray", "gray", "inferno", None]
    for ax, title, panel, cmap in zip(axes, titles, panels, cmaps):
        ax.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=8.2, color=INK)
        ax.axis("off")
    path = OUT_DIR / f"{key}_prob_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    d = dice(pred, gt)
    print(f"{path.name[:46]}…  Dice@0.4={d:.4f}  pred_px={int(pred.sum())}")


def performance_figure(model, data_root: Path, te: pd.DataFrame, ports: list[str], name: str, label: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.0))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.10)
    for col, port in enumerate(ports):
        sub = te[te["port_id"] == port]
        mean_d = sub["dice"].mean()
        row = sub.iloc[(sub["dice"] - mean_d).abs().argsort()].iloc[0]
        key = row["key"]
        rgb, prob, gt = predict(model, data_root, key)
        pred = (prob >= TAU).astype(np.float32)
        axes[0, col].imshow(overlay(rgb, pred))
        axes[0, col].set_title(f"{PORT_TITLE[port]} — pred overlay (case Dice {row['dice']:.3f})", fontsize=9.2, color=INK)
        axes[1, col].imshow(overlay(rgb, gt, color=(43, 159, 214)))
        axes[1, col].set_title("Ground truth overlay", fontsize=9.2, color=INK)
        for r in range(2):
            axes[r, col].axis("off")
        print(f"  {name}: {port} -> {key[:38]}… (case Dice {row['dice']:.3f}, port avg {mean_d:.3f})")
    fig.suptitle(label, fontsize=13, weight="bold", color=INK, y=0.985)
    path = OUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(path)


def roi_story(model, data_root: Path) -> None:
    resolver = unet_roi.ROIResolver(
        polygons_path=REPO / "data/inputs/Havne_koor.txt",
        patch_bboxes_path=REPO / "data/inputs/patch_bboxes_final_49ports_lalb_20260527.txt",
        port_map_path=None,
        buffer_px=10,
    )
    key = GUANGXI
    rgb, prob, gt = predict(model, data_root, key)
    pred = (prob >= TAU).astype(np.float32)
    result = resolver.resolve(key, "P2", 512)
    roi = result.mask.astype(np.float32)
    post = pred * roi
    removed = pred * (1 - roi)
    effect = rgb.astype(np.float32).copy()
    for c, col in zip(range(3), (0, 168, 120)):
        effect[:, :, c][post.astype(bool)] = 0.45 * effect[:, :, c][post.astype(bool)] + 0.55 * col
    for c, col in zip(range(3), (214, 69, 65)):
        effect[:, :, c][removed.astype(bool)] = 0.45 * effect[:, :, c][removed.astype(bool)] + 0.55 * col
    effect = effect.astype(np.uint8)

    panels = [(rgb, None, "Validation image"), (gt, "gray", "Ground truth"),
              (pred, "gray", f"Prediction before ROI ($\\tau$={TAU})"), (roi, "gray", "ROI mask (b=10 px)"),
              (post, "gray", "Prediction after ROI"), (effect, None, "ROI effect: kept (green) / removed (red)")]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.0))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.01, wspace=0.03, hspace=0.10)
    for ax, (panel, cmap, title) in zip(axes.flat, panels):
        ax.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=9.6, color=INK)
        ax.axis("off")
    path = OUT_DIR / "roi_story_cleaned_guangxi_beibu_p2.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    pct = 100 * (pred.sum() - post.sum()) / max(pred.sum(), 1)
    print(f"ROI story: pred px {int(pred.sum())} -> {int(post.sum())} ({pct:.1f}% fjernet); Dice før {dice(pred, gt):.4f} / efter {dice(post, gt):.4f}")
    print(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = segmentation_image_root()
    model = load_model()
    print("Model indlæst (epoch 96, GroupNorm, percentile 2/98)")
    te = git_csv(f"{ART}/raw/test/predictions.csv")
    for key in PANEL_CASES:
        prob_panel(model, data_root, key)
    performance_figure(model, data_root, te, HIGH_PORTS, "high_performace_test.png",
                       "Representative examples from high-performing test ports ($\\tau$=0.4)")
    performance_figure(model, data_root, te, LOW_PORTS, "low_performance_test.png",
                       "Representative examples from low-performing test ports ($\\tau$=0.4)")
    roi_story(model, data_root)


if __name__ == "__main__":
    main()
