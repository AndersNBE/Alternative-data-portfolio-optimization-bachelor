import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.code.dataset_contract import validate_and_pair
from data.code.prepare_splits import create_splits
from models.ml.unet.utils import write_csv

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def default_onedrive_root() -> Path:
    default_root = (
        Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-SharedLibraries-DanmarksTekniskeUniversitet"
        / "Bjarke Jørn Kristensen - Bachelor"
    )
    return Path(os.path.expandvars(os.environ.get("BACHELOR_ONEDRIVE_ROOT", str(default_root)))).expanduser()


def resolve_config_path(raw: str) -> Path:
    return Path(os.path.expandvars(raw)).expanduser().resolve()


ONEDRIVE_BACHELOR_ROOT = default_onedrive_root()
ONEDRIVE_SEGMENTATION_ROOT = ONEDRIVE_BACHELOR_ROOT / "segmentation"


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(raw)
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping/object")
        return data

    raise ValueError("Config must be .json, .yml or .yaml")


def _write_pair_outputs(
    pairs: list[Any],
    errors: list[Any],
    summary: dict[str, int],
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = out_dir / "pair_report.csv"
    err_csv = out_dir / "pair_errors.csv"
    summary_json = out_dir / "pair_summary.json"

    pair_rows = [
        {
            "key": p.key,
            "basename": p.basename,
            "image_path": str(p.image_path),
            "mask_path": str(p.mask_path),
            "port_id": p.port_id,
            "patch_id": p.patch_id,
            "timestamp": p.timestamp,
        }
        for p in pairs
    ]
    error_rows = [
        {
            "error_type": e.error_type,
            "key": e.key,
            "image_path": e.image_path,
            "mask_path": e.mask_path,
            "details": e.details,
        }
        for e in errors
    ]

    write_csv(
        pair_csv,
        pair_rows,
        ["key", "basename", "image_path", "mask_path", "port_id", "patch_id", "timestamp"],
    )
    write_csv(
        err_csv,
        error_rows,
        ["error_type", "key", "image_path", "mask_path", "details"],
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pair_csv, err_csv, summary_json


def aggregate_predictions(predictions_csv: Path, out_csv: Path) -> Path:
    import csv

    with predictions_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        port_id = row.get("port_id", "") or "unknown"
        timestamp = row.get("timestamp", "") or "unknown"
        grouped[(port_id, timestamp)].append(row)

    agg_rows: list[dict[str, Any]] = []
    for (port_id, timestamp), samples in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        pred_pixels = sum(int(float(s.get("pred_container_pixels", 0) or 0)) for s in samples)
        total_pixels = 0
        patch_ids: list[str] = []
        for s in samples:
            h = int(float(s.get("image_height", 0) or 0))
            w = int(float(s.get("image_width", 0) or 0))
            total_pixels += h * w
            patch = (s.get("patch_id", "") or "").strip()
            if patch:
                patch_ids.append(patch)

        agg_rows.append(
            {
                "port_id": port_id,
                "timestamp": timestamp,
                "pred_container_pixels_sum": pred_pixels,
                "total_pixels_sum": total_pixels,
                "pred_container_ratio_sum": float(pred_pixels / max(total_pixels, 1)),
                "patch_count": len(samples),
                "patch_ids": "|".join(sorted(set(patch_ids))),
            }
        )

    write_csv(
        out_csv,
        agg_rows,
        [
            "port_id",
            "timestamp",
            "pred_container_pixels_sum",
            "total_pixels_sum",
            "pred_container_ratio_sum",
            "patch_count",
            "patch_ids",
        ],
    )
    return out_csv


def run_pipeline(config_path: Path) -> dict[str, str]:
    from models.ml.unet.infer import run_inference
    from models.ml.unet.report import build_run_report
    from models.ml.unet.train import train_model

    cfg = _load_config(config_path)

    paths = cfg.get("paths", {})
    split_cfg = cfg.get("split", {})
    train_cfg = cfg.get("train", {})
    infer_cfg = cfg.get("infer", {})
    aggregation_cfg = cfg.get("aggregation", {})

    run_root = resolve_config_path(paths.get("run_root", str(ONEDRIVE_SEGMENTATION_ROOT / "runs")))
    run_id = cfg.get("run_id", "") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    images_dir = resolve_config_path(paths["images_dir"])
    masks_dir = resolve_config_path(paths["masks_dir"])
    ext = paths.get("ext", ".png")

    pairs, errors, summary = validate_and_pair(images_dir, masks_dir, ext=ext)
    pair_csv, err_csv, summary_json = _write_pair_outputs(
        pairs=pairs,
        errors=errors,
        summary=summary,
        out_dir=run_dir / "contract",
    )

    fail_on_errors = bool(cfg.get("fail_on_contract_errors", True))
    if fail_on_errors and errors:
        raise RuntimeError(
            f"Contract validation failed with {len(errors)} errors. See {err_csv}"
        )

    split_summary = create_splits(
        pair_csv=pair_csv,
        out_dir=run_dir / "splits",
        strategy=split_cfg.get("strategy", "hybrid"),
        seed=int(split_cfg.get("seed", 42)),
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        test_start_timestamp=str(split_cfg.get("test_start_timestamp", "")),
        train_end_timestamp=str(split_cfg.get("train_end_timestamp", "")),
    )

    train_args = argparse.Namespace(
        train_csv=split_summary["train_csv"],
        val_csv=split_summary["val_csv"],
        img_size=int(train_cfg.get("img_size", 512)),
        batch_size=int(train_cfg.get("batch_size", 1)),
        lr=float(train_cfg.get("lr", 1e-4)),
        loss=str(train_cfg.get("loss", "bce")),
        epochs=int(train_cfg.get("epochs", 30)),
        threshold=float(train_cfg.get("threshold", 0.5)),
        device=str(train_cfg.get("device", "auto")),
        seed=int(train_cfg.get("seed", 42)),
        augment=bool(train_cfg.get("augment", True)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        out_dir=str(run_root),
        run_id=run_id,
    )
    train_result = train_model(train_args)

    infer_args = argparse.Namespace(
        checkpoint=train_result["best_checkpoint"],
        input_csv=split_summary["test_csv"],
        out=str(run_dir / "inference"),
        img_size=int(infer_cfg.get("img_size", train_args.img_size)),
        threshold=float(infer_cfg.get("threshold", train_args.threshold)),
        batch_size=int(infer_cfg.get("batch_size", 1)),
        device=str(infer_cfg.get("device", train_args.device)),
        num_workers=int(infer_cfg.get("num_workers", 0)),
        require_mask=bool(infer_cfg.get("require_mask", True)),
    )
    infer_result = run_inference(infer_args)

    port_timeseries_csv = aggregate_predictions(
        predictions_csv=Path(infer_result["predictions_csv"]),
        out_csv=run_dir / aggregation_cfg.get("port_timeseries_name", "port_timeseries.csv"),
    )

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "pair_report": str(pair_csv),
        "pair_errors": str(err_csv),
        "pair_summary": str(summary_json),
        "train_csv": split_summary["train_csv"],
        "val_csv": split_summary["val_csv"],
        "test_csv": split_summary["test_csv"],
        "best_checkpoint": train_result["best_checkpoint"],
        "last_checkpoint": train_result["last_checkpoint"],
        "metrics_csv": train_result["metrics_csv"],
        "predictions_csv": infer_result["predictions_csv"],
        "inference_summary": infer_result["summary_json"],
        "port_timeseries_csv": str(port_timeseries_csv),
    }
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_result = build_run_report(
        argparse.Namespace(
            run_dir=str(run_dir),
            out="",
            max_samples=12,
        )
    )
    manifest["run_report_html"] = report_result["report_html"]
    manifest["run_report_summary_json"] = report_result["report_summary_json"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full segmentation pipeline from config.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_pipeline(Path(args.config).resolve())
    print("Segmentation pipeline complete")
    for key, value in manifest.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
