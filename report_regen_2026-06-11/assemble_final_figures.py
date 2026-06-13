from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


BUNDLE_DIRS = [
    "segmentation_tau04_bundle",
    "container_index_tau04_bundle",
    "mad_tau04_bundle",
    "panels_tau04_bundle",
]

DIRECT_FINAL_FIGURES = [
    "final_model03_train_loss_vs_val_loss.png",
    "final_model03_validation_metrics_over_epochs.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble FinalFigures from the locked report regeneration bundles.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-missing-direct", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "FinalFigures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for bundle in BUNDLE_DIRS:
        source_dir = root / bundle
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing bundle: {source_dir}")
        for src in sorted(source_dir.glob("*.png")):
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            rows.append(
                {
                    "figure": dst.name,
                    "source_bundle": bundle,
                    "source_path": str(src.relative_to(root)),
                    "sha256": sha256(dst),
                }
            )

    for name in DIRECT_FINAL_FIGURES:
        direct = out_dir / name
        if not direct.exists():
            if args.allow_missing_direct:
                continue
            raise FileNotFoundError(
                f"Missing direct FinalFigures file: {direct}. Run regen_final_model03_training_figs.py first."
            )
        rows.append(
            {
                "figure": name,
                "source_bundle": "FinalFigures/direct",
                "source_path": str(direct.relative_to(root)),
                "sha256": sha256(direct),
            }
        )

    manifest = out_dir / "MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "source_bundle", "source_path", "sha256"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["figure"]))
    print(manifest)


if __name__ == "__main__":
    main()
