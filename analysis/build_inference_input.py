"""
Scans a folder of satellite patch images and writes a CSV that infer.py can
consume directly.

Expected filename format (produced by Hent_billederne.py):
  <port_slug>__<patch_id>__L1C__<timestamp>__eoCC<cloud>__CR<cr>__ND<nd>__<size>px.png

Usage:
  python -m analysis.build_inference_input \
    --images-dir "/path/to/ALLE GODE BILLEDER" \
    --out inference_input.csv \
    --max-cloud 30
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<port_slug>.+?)__"
    r"(?P<patch_id>P\d+)__"
    r"L1C__"
    r"(?P<timestamp>\d{8}T\d{6}Z)__"
    r"eoCC(?P<cloud>[\d.]+)"
)


def parse_image(path: Path) -> dict | None:
    m = FILENAME_RE.match(path.stem)
    if m is None:
        return None
    return {
        "image_path": str(path),
        "mask_path":  "",
        "key":        path.stem,
        "basename":   path.stem,
        "port_id":    m.group("port_slug"),
        "patch_id":   m.group("patch_id"),
        "timestamp":  m.group("timestamp"),
        "cloud_cover": float(m.group("cloud")),
    }


def build_input_csv(images_dir: Path, out_path: Path, max_cloud: float) -> int:
    fieldnames = ["image_path", "mask_path", "key", "basename", "port_id", "patch_id", "timestamp"]
    rows: list[dict] = []

    for png in sorted(images_dir.rglob("*.png")):
        # Skip subdirectories that are not port folders (e.g. Kasper/dev)
        relative_parts = png.relative_to(images_dir).parts
        if len(relative_parts) != 2:
            continue

        parsed = parse_image(png)
        if parsed is None:
            print(f"  Skipping (unrecognised filename): {png.name}")
            continue
        if parsed["cloud_cover"] > max_cloud:
            continue

        rows.append({k: parsed[k] for k in fieldnames})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inference input CSV from image folder.")
    parser.add_argument("--images-dir", required=True, help="Root folder containing one subfolder per port.")
    parser.add_argument("--out",         required=True, help="Output CSV path.")
    parser.add_argument("--max-cloud",   type=float, default=30.0,
                        help="Discard images with cloud cover above this percentage (default: 30).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = build_input_csv(
        images_dir=Path(args.images_dir),
        out_path=Path(args.out),
        max_cloud=args.max_cloud,
    )
    print(f"Wrote {n:,} rows → {args.out}")


if __name__ == "__main__":
    main()
