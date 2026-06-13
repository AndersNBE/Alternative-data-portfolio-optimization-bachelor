import argparse
import os
import shutil
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
        description="Fjern afskaarede billeder fra god-mappen."
    )
    parser.add_argument(
        "--root-dir",
        default=str(ONEDRIVE_DATASET_ROOT),
        help="Root mappe som indeholder god/ undermappen.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--near-black-thr", type=int, default=8)
    parser.add_argument("--near-white-thr", type=int, default=247)
    parser.add_argument("--frac-thr", type=float, default=0.28)
    parser.add_argument("--edge-band-frac-thr", type=float, default=0.55)
    parser.add_argument("--edge-band-width-frac", type=float, default=0.18)
    parser.add_argument("--debug-first-n", type=int, default=40)
    parser.add_argument(
        "--move-dir",
        default="",
        help="Valgfri mappe til at flytte flaggede billeder til i stedet for at slette dem.",
    )
    return parser.parse_args()


def is_cutoff_image(
    img_rgb: np.ndarray,
    near_black_thr: int = 8,
    near_white_thr: int = 247,
    frac_thr: float = 0.28,
    edge_band_frac_thr: float = 0.55,
    edge_band_width_frac: float = 0.18,
):
    arr = img_rgb.astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    near_black = gray <= float(near_black_thr)
    near_white = gray >= float(near_white_thr)
    bad = near_black | near_white

    h, w = bad.shape
    bad_frac = float(bad.mean())

    if bad_frac >= frac_thr:
        return True, {"reason": "global_bad_frac", "bad_frac": bad_frac}

    bw_h = max(1, int(round(h * edge_band_width_frac)))
    bw_w = max(1, int(round(w * edge_band_width_frac)))

    top = bad[:bw_h, :].mean()
    bottom = bad[h - bw_h :, :].mean()
    left = bad[:, :bw_w].mean()
    right = bad[:, w - bw_w :].mean()

    edge_max = float(max(top, bottom, left, right))

    if edge_max >= edge_band_frac_thr and bad_frac >= 0.12:
        return True, {
            "reason": "edge_band",
            "bad_frac": bad_frac,
            "top": float(top),
            "bottom": float(bottom),
            "left": float(left),
            "right": float(right),
        }

    return False, {"reason": "ok", "bad_frac": bad_frac}


def cleanup_god_folder(
    root_dir: str,
    dry_run: bool = False,
    move_dir: str = "",
    near_black_thr: int = 8,
    near_white_thr: int = 247,
    frac_thr: float = 0.28,
    edge_band_frac_thr: float = 0.55,
    edge_band_width_frac: float = 0.18,
    debug_first_n: int = 30,
):
    root = Path(root_dir)
    god_root = root / "god"
    if not god_root.exists():
        raise FileNotFoundError(f"Kunne ikke finde god mappen: {god_root}")
    move_root = Path(move_dir).expanduser().resolve() if move_dir else None
    if move_root is not None and not dry_run:
        move_root.mkdir(parents=True, exist_ok=True)

    port_dirs = [p for p in god_root.iterdir() if p.is_dir()]

    total_deleted = 0
    debug_printed = 0

    for port_dir in sorted(port_dirs):
        deleted = 0
        scanned = 0

        for img_path in port_dir.rglob("*.png"):
            scanned += 1
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue

            flag, info = is_cutoff_image(
                img,
                near_black_thr=near_black_thr,
                near_white_thr=near_white_thr,
                frac_thr=frac_thr,
                edge_band_frac_thr=edge_band_frac_thr,
                edge_band_width_frac=edge_band_width_frac,
            )

            if flag:
                if debug_printed < debug_first_n:
                    action = "MOVE" if move_root is not None else "DELETE"
                    print(action, port_dir.name, img_path.name, info)
                    debug_printed += 1

                deleted += 1
                total_deleted += 1
                if not dry_run:
                    try:
                        if move_root is not None:
                            dst = move_root / img_path.relative_to(god_root)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(img_path), str(dst))
                        else:
                            os.remove(img_path)
                    except Exception:
                        pass

        action_word = "flagged" if move_root is not None else "deleted"
        print(f"{port_dir.name}: scanned={scanned} {action_word}={deleted}")
    action_word = "moved" if move_root is not None else "deleted"
    print(f"\nTOTAL {action_word}={total_deleted} dry_run={dry_run}")


if __name__ == "__main__":
    args = parse_args()
    cleanup_god_folder(
        root_dir=str(Path(args.root_dir).expanduser()),
        dry_run=args.dry_run,
        move_dir=args.move_dir,
        near_black_thr=args.near_black_thr,
        near_white_thr=args.near_white_thr,
        frac_thr=args.frac_thr,
        edge_band_frac_thr=args.edge_band_frac_thr,
        edge_band_width_frac=args.edge_band_width_frac,
        debug_first_n=args.debug_first_n,
    )
