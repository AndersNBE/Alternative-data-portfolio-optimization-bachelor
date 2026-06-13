import shutil
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


def default_onedrive_root() -> Path:
    default_root = (
        Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-SharedLibraries-DanmarksTekniskeUniversitet"
        / "Bjarke Jørn Kristensen - Bachelor"
    )
    return Path(os.path.expandvars(os.environ.get("BACHELOR_ONEDRIVE_ROOT", str(default_root)))).expanduser()


ONEDRIVE_BACHELOR_ROOT = default_onedrive_root()
ONEDRIVE_DATASET_ROOT = ONEDRIVE_BACHELOR_ROOT / "dataset_patches_flat"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sorter billeder i god og dårlig ud fra cloud + haze regler."
    )
    parser.add_argument(
        "--root-dir",
        default=str(ONEDRIVE_DATASET_ROOT),
        help="Root mappe med port-mapperne.",
    )
    parser.add_argument("--thr-good", type=float, default=0.04)
    parser.add_argument("--std-thr", type=float, default=10.0)
    parser.add_argument("--lap-thr", type=float, default=220.0)
    parser.add_argument("--sat-thr", type=float, default=0.24)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--mean-gray-gate", type=float, default=70.0)
    parser.add_argument("--grad-percentile", type=float, default=85.0)
    parser.add_argument("--min-structured-pixels", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-outputs", action="store_true")
    parser.add_argument("--debug-first-n", type=int, default=80)
    return parser.parse_args()


def rgb_cloud_ratio(img_rgb: np.ndarray) -> float:
    arr = img_rgb.astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)

    brightness = cmax
    sat = np.zeros_like(cmax)
    mask = cmax > 1e-6
    sat[mask] = (cmax[mask] - cmin[mask]) / cmax[mask]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    gx = np.abs(gray[:, 2:] - gray[:, :-2])
    gy = np.abs(gray[2:, :] - gray[:-2, :])
    gx = np.pad(gx, ((0, 0), (1, 1)), mode="edge")
    gy = np.pad(gy, ((1, 1), (0, 0)), mode="edge")
    grad = gx + gy

    bright_mask = brightness > 200
    low_sat_mask = sat < 0.18
    low_tex_mask = grad < 18

    cloud = bright_mask & low_sat_mask & low_tex_mask
    return float(cloud.mean())


def haze_metrics_masked(img_rgb: np.ndarray, mask: np.ndarray):
    arr = img_rgb.astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)

    sat = np.zeros_like(cmax)
    m = cmax > 1e-6
    sat[m] = (cmax[m] - cmin[m]) / cmax[m]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    gray_v = gray[mask]
    sat_v = sat[mask]

    mean_gray = float(gray_v.mean())
    std_gray = float(gray_v.std())
    mean_sat = float(sat_v.mean())

    d2x = gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2]
    d2y = gray[2:, :] - 2 * gray[1:-1, :] + gray[:-2, :]
    d2x = np.pad(d2x, ((0, 0), (1, 1)), mode="edge")
    d2y = np.pad(d2y, ((1, 1), (0, 0)), mode="edge")
    lap = d2x + d2y

    lap_var = float(lap[mask].var())

    return mean_gray, std_gray, lap_var, mean_sat


def structured_mask(img_rgb: np.ndarray, grad_percentile: float = 85.0) -> np.ndarray:
    arr = img_rgb.astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    gx = np.abs(gray[:, 2:] - gray[:, :-2])
    gy = np.abs(gray[2:, :] - gray[:-2, :])
    gx = np.pad(gx, ((0, 0), (1, 1)), mode="edge")
    gy = np.pad(gy, ((1, 1), (0, 0)), mode="edge")
    grad = gx + gy

    thr = float(np.percentile(grad, grad_percentile))
    return grad >= thr


def is_haze(
    img_rgb: np.ndarray,
    std_thr: float = 10.0,
    lap_thr: float = 220.0,
    sat_thr: float = 0.24,
    min_hits: int = 2,
    mean_gray_gate: float = 70.0,
    grad_percentile: float = 85.0,
    min_structured_pixels: int = 2000,
):
    mask = structured_mask(img_rgb, grad_percentile=grad_percentile)

    if int(mask.sum()) < min_structured_pixels:
        return False

    mean_gray, std_gray, lap_var, mean_sat = haze_metrics_masked(img_rgb, mask)

    if mean_gray < mean_gray_gate:
        return False

    hit_std = std_gray < std_thr
    hit_lap = lap_var < lap_thr
    hit_sat = mean_sat < sat_thr

    hits = int(hit_std) + int(hit_lap) + int(hit_sat)
    return hits >= min_hits


def crop_for_port_rule(img_rgb: np.ndarray, port_name: str, filename: str) -> np.ndarray:
    """
    Rules:
    Manila: always top third
    Shanghai: if filename contains P1 and does not contain P2, bottom third
    Else: full image
    """
    h = img_rgb.shape[0]
    port_l = port_name.lower()
    fname = filename.lower()

    if port_l == "manila":
        y1 = h // 3
        return img_rgb[0:y1, :, :]

    if port_l == "shanghai":
        has_p1 = "__p1__" in fname or "p1" in fname
        has_p2 = "__p2__" in fname or "p2" in fname
        if has_p1 and (not has_p2):
            y0 = (2 * h) // 3
            return img_rgb[y0:h, :, :]
        return img_rgb

    return img_rgb


def should_skip_path(p: Path) -> bool:
    parts = {part.lower() for part in p.parts}
    if "sorted_good" in parts or "sorted_doubt" in parts or "sorted_bad" in parts or "sorted_haze" in parts:
        return True
    if "god" in parts or "dårlig" in parts:
        return True
    return False


def empty_dir(d: Path):
    if d.exists():
        for p in d.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)


def collect_good_bad_all_ports(
    root_dir: str,
    thr_good: float = 0.04,
    std_thr: float = 10.0,
    lap_thr: float = 220.0,
    sat_thr: float = 0.24,
    min_hits: int = 2,
    mean_gray_gate: float = 70.0,
    grad_percentile: float = 85.0,
    min_structured_pixels: int = 2000,
    dry_run: bool = False,
    clean_outputs: bool = False,
    debug_first_n: int = 60,
):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root folder not found: {root_dir}")

    out_good_root = root_dir / "god"
    out_bad_root = root_dir / "dårlig"

    out_good_root.mkdir(exist_ok=True)
    out_bad_root.mkdir(exist_ok=True)

    if clean_outputs and not dry_run:
        empty_dir(out_good_root)
        empty_dir(out_bad_root)
        out_good_root.mkdir(exist_ok=True)
        out_bad_root.mkdir(exist_ok=True)

    port_dirs = [p for p in root_dir.iterdir() if p.is_dir() and p.name not in {"god", "dårlig"}]

    total_good = 0
    total_bad = 0
    debug_printed = 0

    for port_dir in port_dirs:
        port_name = port_dir.name
        out_port_good = out_good_root / port_name
        out_port_bad = out_bad_root / port_name
        out_port_good.mkdir(parents=True, exist_ok=True)
        out_port_bad.mkdir(parents=True, exist_ok=True)

        pngs = [p for p in port_dir.rglob("*.png") if p.is_file() and not should_skip_path(p)]
        n_good = 0
        n_bad = 0

        for p in pngs:
            img_full = np.array(Image.open(p).convert("RGB"))
            img = crop_for_port_rule(img_full, port_name=port_name, filename=p.name)

            haze_flag = is_haze(
                img,
                std_thr=std_thr,
                lap_thr=lap_thr,
                sat_thr=sat_thr,
                min_hits=min_hits,
                mean_gray_gate=mean_gray_gate,
                grad_percentile=grad_percentile,
                min_structured_pixels=min_structured_pixels,
            )

            cr = rgb_cloud_ratio(img)
            is_good = (not haze_flag) and (cr <= thr_good)

            if debug_printed < debug_first_n and port_name.lower() in {"manila", "shanghai"}:
                used = "full"
                if port_name.lower() == "manila":
                    used = "top_third"
                elif port_name.lower() == "shanghai":
                    fname = p.name.lower()
                    has_p1 = "__p1__" in fname or "p1" in fname
                    has_p2 = "__p2__" in fname or "p2" in fname
                    if has_p1 and (not has_p2):
                        used = "bottom_third"

                print(
                    port_name,
                    used,
                    p.name,
                    "GOOD" if is_good else "BAD",
                    "cloud", round(cr, 3),
                    "haze", haze_flag,
                )
                debug_printed += 1

            target = out_port_good if is_good else out_port_bad
            if is_good:
                n_good += 1
            else:
                n_bad += 1

            if not dry_run:
                shutil.copy2(p, target / p.name)

        total_good += n_good
        total_bad += n_bad
        print(f"{port_name}: good={n_good} bad={n_bad} scanned={len(pngs)}")

    print("\nSUMMARY")
    print("root:", root_dir)
    print("good total:", total_good)
    print("bad total:", total_bad)
    print("thr_good:", thr_good)
    print("haze params std<", std_thr, "lap<", lap_thr, "sat<", sat_thr, "min_hits", min_hits, "mean_gate", mean_gray_gate)
    print("structured mask percentile:", grad_percentile, "min_structured_pixels:", min_structured_pixels)
    print("dry_run:", dry_run)
    print("clean_outputs:", clean_outputs)
    print("outputs:", out_good_root, out_bad_root)


if __name__ == "__main__":
    args = parse_args()
    collect_good_bad_all_ports(
        root_dir=str(Path(args.root_dir).expanduser()),
        thr_good=args.thr_good,
        std_thr=args.std_thr,
        lap_thr=args.lap_thr,
        sat_thr=args.sat_thr,
        min_hits=args.min_hits,
        mean_gray_gate=args.mean_gray_gate,
        grad_percentile=args.grad_percentile,
        min_structured_pixels=args.min_structured_pixels,
        dry_run=args.dry_run,
        clean_outputs=args.clean_outputs,
        debug_first_n=args.debug_first_n,
    )
