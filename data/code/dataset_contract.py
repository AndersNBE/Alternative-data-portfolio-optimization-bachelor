import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTATION_OUTPUT_ROOT = REPO_ROOT / "data" / "outputs" / "segmentation"

MASK_SUFFIX = "_mask"

PATCH_RE = re.compile(r"(?:^|[_-])(P\d+)(?:[_-]|$)", re.IGNORECASE)
TIMESTAMP_PATTERNS = [
    re.compile(r"(\d{8}T\d{6}Z)", re.IGNORECASE),
    re.compile(r"(\d{8}T\d{6})", re.IGNORECASE),
    re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}[-:]\d{2}[-:]\d{2}Z?)", re.IGNORECASE),
    re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
]


@dataclass
class PairRecord:
    key: str
    image_path: Path
    mask_path: Path
    basename: str
    patch_id: str
    timestamp: str
    port_id: str


@dataclass
class PairError:
    error_type: str
    key: str
    image_path: str
    mask_path: str
    details: str


def _normalize_key(path: Path) -> str:
    return str(path.with_suffix("")).replace("\\", "/")


def _extract_patch_id(name: str) -> str:
    match = PATCH_RE.search(name)
    if not match:
        return ""
    return match.group(1).upper()


def _extract_timestamp(name: str) -> str:
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(name)
        if match:
            return match.group(1)
    return ""


def _guess_port_id(relative_no_ext: Path, basename: str) -> str:
    rel_parts = relative_no_ext.parts
    if len(rel_parts) > 1:
        return rel_parts[0]

    # Preferred rule: filenames are usually structured like
    # "<port_slug>__P1__L1C__20231217T070309Z__...".
    # In that case we keep the full harbour slug instead of only the first token.
    slug_part = basename.split("__", 1)[0].strip()
    if slug_part:
        slug = re.sub(r"[\s\-]+", "_", slug_part).strip("_").lower()
        if slug:
            return slug

    # Fallback for older filename patterns without "__".
    cleaned = basename
    timestamp = _extract_timestamp(cleaned)
    if timestamp:
        cleaned = cleaned.replace(timestamp, "")
    patch = _extract_patch_id(cleaned)
    if patch:
        cleaned = re.sub(rf"(?:^|[_-]){patch}(?:[_-]|$)", "_", cleaned, flags=re.IGNORECASE)

    tokens = [tok for tok in re.split(r"[_\-\s]+", cleaned) if tok]
    if tokens:
        return "_".join(tok.lower() for tok in tokens)

    return "unknown"



def _iter_files(root: Path, ext: str) -> list[Path]:
    ext_l = ext.lower()
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ext_l
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_and_pair(
    images_root: Path,
    masks_root: Path,
    ext: str = ".png",
) -> tuple[list[PairRecord], list[PairError], dict[str, int]]:
    images_root = images_root.resolve()
    masks_root = masks_root.resolve()

    image_files = _iter_files(images_root, ext)
    mask_files = _iter_files(masks_root, ext)

    image_map: dict[str, Path] = {}
    mask_map: dict[str, Path] = {}
    errors: list[PairError] = []

    for image_path in image_files:
        relative = image_path.relative_to(images_root)
        if relative.stem.endswith(MASK_SUFFIX):
            continue
        key = _normalize_key(relative)
        if key in image_map:
            errors.append(
                PairError(
                    error_type="duplicate_image_key",
                    key=key,
                    image_path=str(image_path),
                    mask_path="",
                    details=f"Key already used by {image_map[key]}",
                )
            )
            continue
        image_map[key] = image_path

    for mask_path in mask_files:
        relative = mask_path.relative_to(masks_root)
        if not relative.stem.endswith(MASK_SUFFIX):
            errors.append(
                PairError(
                    error_type="invalid_mask_name",
                    key=_normalize_key(relative),
                    image_path="",
                    mask_path=str(mask_path),
                    details=f"Mask must end with '{MASK_SUFFIX}'",
                )
            )
            continue

        original_stem = relative.stem[: -len(MASK_SUFFIX)]
        key = _normalize_key(relative.with_name(original_stem + relative.suffix))
        if key in mask_map:
            errors.append(
                PairError(
                    error_type="duplicate_mask_key",
                    key=key,
                    image_path="",
                    mask_path=str(mask_path),
                    details=f"Key already used by {mask_map[key]}",
                )
            )
            continue
        mask_map[key] = mask_path

    pairs: list[PairRecord] = []
    all_keys = sorted(set(image_map) | set(mask_map))

    for key in all_keys:
        image_path = image_map.get(key)
        mask_path = mask_map.get(key)

        if image_path is None:
            errors.append(
                PairError(
                    error_type="missing_image",
                    key=key,
                    image_path="",
                    mask_path=str(mask_path),
                    details="Mask exists without matching image",
                )
            )
            continue

        if mask_path is None:
            errors.append(
                PairError(
                    error_type="missing_mask",
                    key=key,
                    image_path=str(image_path),
                    mask_path="",
                    details="Image exists without matching mask",
                )
            )
            continue

        relative_no_ext = Path(key)
        basename = relative_no_ext.name
        patch_id = _extract_patch_id(basename)
        timestamp = _extract_timestamp(basename)
        port_id = _guess_port_id(relative_no_ext, basename)

        pairs.append(
            PairRecord(
                key=key,
                image_path=image_path,
                mask_path=mask_path,
                basename=basename,
                patch_id=patch_id,
                timestamp=timestamp,
                port_id=port_id,
            )
        )

    summary = {
        "images_found": len(image_map),
        "masks_found": len(mask_map),
        "pairs_matched": len(pairs),
        "errors": len(errors),
    }
    return pairs, errors, summary


def run_cli(args: argparse.Namespace) -> int:
    pairs, errors, summary = validate_and_pair(
        images_root=Path(args.images),
        masks_root=Path(args.masks),
        ext=args.ext,
    )

    out_dir = Path(args.out_dir).resolve()
    pair_csv = out_dir / args.report_name
    error_csv = out_dir / args.errors_name
    summary_json = out_dir / args.summary_name

    pair_rows = [
        {
            "key": pair.key,
            "basename": pair.basename,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
            "port_id": pair.port_id,
            "patch_id": pair.patch_id,
            "timestamp": pair.timestamp,
        }
        for pair in pairs
    ]
    error_rows = [
        {
            "error_type": err.error_type,
            "key": err.key,
            "image_path": err.image_path,
            "mask_path": err.mask_path,
            "details": err.details,
        }
        for err in errors
    ]

    _write_csv(
        pair_csv,
        pair_rows,
        ["key", "basename", "image_path", "mask_path", "port_id", "patch_id", "timestamp"],
    )
    _write_csv(
        error_csv,
        error_rows,
        ["error_type", "key", "image_path", "mask_path", "details"],
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pair validation complete")
    print(json.dumps(summary, indent=2))
    print("Pair report:", pair_csv)
    print("Error report:", error_csv)

    if args.fail_on_errors and summary["errors"] > 0:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate strict image/mask contract and create pair report CSV."
    )
    parser.add_argument("--images", required=True, help="Directory with source images.")
    parser.add_argument("--masks", required=True, help="Directory with mask images.")
    parser.add_argument("--ext", default=".png", help="Image extension (default: .png).")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_SEGMENTATION_OUTPUT_ROOT),
        help="Output folder for reports.",
    )
    parser.add_argument("--report-name", default="pair_report.csv")
    parser.add_argument("--errors-name", default="pair_errors.csv")
    parser.add_argument("--summary-name", default="pair_summary.json")
    parser.add_argument(
        "--fail-on-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit with code 1 if pairing errors are found (default: true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_cli(parse_args()))
