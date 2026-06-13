import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_checkpoint(checkpoint_path: Path, device: Any) -> dict[str, Any]:
    import torch

    return torch.load(checkpoint_path, map_location=device)


def _load_model_state(model: Any, ckpt: dict[str, Any]) -> None:
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def _parse_threshold_list(value: str) -> list[float]:
    thresholds: list[float] = []
    for raw in (value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        threshold = float(raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got: {threshold}")
        thresholds.append(threshold)
    return thresholds


def _save_binary_mask(path: Path, mask_01: Any) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask_01.astype(np.uint8) * 255)
    Image.fromarray(out).save(path)


def _save_probability_map(path: Path, prob_01: Any) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(prob_01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "y"}


def _avg_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(row.get(key, ""), float("nan")) for row in rows if str(row.get(key, "")).strip() != ""]
    values = [value for value in values if value == value]
    return float(sum(values) / max(len(values), 1)) if values else 0.0


def _compute_metrics(pred: Any, target: Any) -> dict[str, float]:
    from models.ml.unet.metrics import dice_score, iou_score, precision_score, recall_score

    return {
        "dice": float(dice_score(pred, target).mean().item()),
        "iou": float(iou_score(pred, target).mean().item()),
        "precision": float(precision_score(pred, target).mean().item()),
        "recall": float(recall_score(pred, target).mean().item()),
    }


def _build_roi_summary_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def summarize(scope_rows: list[dict[str, Any]], *, scope: str, port_slug: str = "", port_id: str = "", polygon_port_name: str = "") -> dict[str, Any]:
        available_rows = [row for row in scope_rows if _to_bool(row.get("roi_available", False))]
        mapped_rows = [row for row in scope_rows if _to_bool(row.get("roi_port_mapped", False))]
        empty_rows = [row for row in scope_rows if _to_bool(row.get("roi_mask_empty", False))]
        without_roi_rows = [row for row in scope_rows if not _to_bool(row.get("roi_available", False))]

        avg_dice_pre = _avg_metric(scope_rows, "dice_pre_roi")
        avg_dice_post = _avg_metric(scope_rows, "dice_post_roi")
        avg_iou_pre = _avg_metric(scope_rows, "iou_pre_roi")
        avg_iou_post = _avg_metric(scope_rows, "iou_post_roi")
        avg_precision_pre = _avg_metric(scope_rows, "precision_pre_roi")
        avg_precision_post = _avg_metric(scope_rows, "precision_post_roi")
        avg_recall_pre = _avg_metric(scope_rows, "recall_pre_roi")
        avg_recall_post = _avg_metric(scope_rows, "recall_post_roi")

        return {
            "scope": scope,
            "port_slug": port_slug,
            "port_id": port_id,
            "polygon_port_name": polygon_port_name,
            "num_cases": len(scope_rows),
            "num_cases_port_mapped": len(mapped_rows),
            "num_cases_with_roi": len(available_rows),
            "num_cases_without_roi": len(without_roi_rows),
            "num_cases_empty_roi": len(empty_rows),
            "avg_roi_coverage_ratio": _avg_metric(available_rows, "roi_coverage_ratio"),
            "avg_dice_pre_roi": avg_dice_pre,
            "avg_dice_post_roi": avg_dice_post,
            "delta_dice": float(avg_dice_post - avg_dice_pre),
            "avg_iou_pre_roi": avg_iou_pre,
            "avg_iou_post_roi": avg_iou_post,
            "delta_iou": float(avg_iou_post - avg_iou_pre),
            "avg_precision_pre_roi": avg_precision_pre,
            "avg_precision_post_roi": avg_precision_post,
            "delta_precision": float(avg_precision_post - avg_precision_pre),
            "avg_recall_pre_roi": avg_recall_pre,
            "avg_recall_post_roi": avg_recall_post,
            "delta_recall": float(avg_recall_post - avg_recall_pre),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("port_slug", "")).strip() or "unknown"].append(row)

    summary_rows = [summarize(rows, scope="overall")]
    for port_slug in sorted(grouped):
        port_rows = grouped[port_slug]
        summary_rows.append(
            summarize(
                port_rows,
                scope="port",
                port_slug=port_slug,
                port_id=str(port_rows[0].get("port_id", "")),
                polygon_port_name=str(port_rows[0].get("roi_polygon_port_name", "")),
            )
        )

    missing_ports = []
    for port_slug in sorted(grouped):
        port_rows = grouped[port_slug]
        if any(not _to_bool(row.get("roi_available", False)) for row in port_rows):
            reasons = sorted({str(row.get("roi_missing_reason", "")) for row in port_rows if not _to_bool(row.get("roi_available", False))})
            missing_ports.append(
                {
                    "port_slug": port_slug,
                    "port_id": str(port_rows[0].get("port_id", "")),
                    "polygon_port_name": str(port_rows[0].get("roi_polygon_port_name", "")),
                    "num_cases": len(port_rows),
                    "num_cases_without_roi": sum(1 for row in port_rows if not _to_bool(row.get("roi_available", False))),
                    "missing_reasons": "|".join(reasons),
                }
            )

    overall = summary_rows[0]
    summary_json = {
        "num_cases": int(overall["num_cases"]),
        "num_cases_port_mapped": int(overall["num_cases_port_mapped"]),
        "num_cases_with_roi": int(overall["num_cases_with_roi"]),
        "num_cases_without_roi": int(overall["num_cases_without_roi"]),
        "num_cases_empty_roi": int(overall["num_cases_empty_roi"]),
        "avg_roi_coverage_ratio": float(overall["avg_roi_coverage_ratio"]),
        "avg_dice_pre_roi": float(overall["avg_dice_pre_roi"]),
        "avg_dice_post_roi": float(overall["avg_dice_post_roi"]),
        "delta_dice": float(overall["delta_dice"]),
        "avg_iou_pre_roi": float(overall["avg_iou_pre_roi"]),
        "avg_iou_post_roi": float(overall["avg_iou_post_roi"]),
        "delta_iou": float(overall["delta_iou"]),
        "avg_precision_pre_roi": float(overall["avg_precision_pre_roi"]),
        "avg_precision_post_roi": float(overall["avg_precision_post_roi"]),
        "delta_precision": float(overall["delta_precision"]),
        "avg_recall_pre_roi": float(overall["avg_recall_pre_roi"]),
        "avg_recall_post_roi": float(overall["avg_recall_post_roi"]),
        "delta_recall": float(overall["delta_recall"]),
        "missing_roi_ports": missing_ports,
    }
    return summary_rows, summary_json


def run_inference(args: argparse.Namespace) -> dict[str, str]:
    import torch
    from torch.utils.data import DataLoader

    from models.ml.unet.dataset import SegmentationDataset
    from models.ml.unet.model import UNet
    from models.ml.unet.roi import ROIResolver, basename_to_port_slug, default_bbox_source, default_polygons_path
    from models.ml.unet.utils import select_device, write_csv

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_images = not args.no_save_images
    pred_mask_dir = out_dir / "pred_masks"
    if save_images:
        pred_mask_dir.mkdir(parents=True, exist_ok=True)

    pred_mask_pre_dir = out_dir / "pred_masks_pre_roi"
    prob_pre_dir = out_dir / "prob_maps_pre_roi"
    prob_post_dir = out_dir / "prob_maps_post_roi"
    roi_mask_dir = out_dir / "roi_masks"

    device = select_device(args.device)
    print(f"Using device: {device}")

    roi_resolver: ROIResolver | None = None
    roi_port_map_used_path: Path | None = None
    if args.apply_roi:
        repo_root = Path(__file__).resolve().parents[3]
        polygons_path = Path(args.roi_polygons_path).resolve() if args.roi_polygons_path else default_polygons_path(repo_root)
        patch_bboxes_path = Path(args.roi_patch_bboxes_path).resolve() if args.roi_patch_bboxes_path else default_bbox_source(repo_root)
        port_map_path = Path(args.roi_port_map_path).resolve() if args.roi_port_map_path else None
        roi_resolver = ROIResolver(
            polygons_path=polygons_path,
            patch_bboxes_path=patch_bboxes_path,
            port_map_path=port_map_path,
            buffer_px=args.roi_buffer_px,
        )
        if save_images:
            pred_mask_pre_dir.mkdir(parents=True, exist_ok=True)
            prob_pre_dir.mkdir(parents=True, exist_ok=True)
            prob_post_dir.mkdir(parents=True, exist_ok=True)
            roi_mask_dir.mkdir(parents=True, exist_ok=True)
        roi_port_map_used_path = out_dir / "roi_port_map_used.json"
        roi_port_map_used_path.write_text(json.dumps(roi_resolver.used_port_map(), indent=2), encoding="utf-8")

    dataset = SegmentationDataset(
        csv_path=Path(args.input_csv),
        img_size=args.img_size,
        augment=False,
        require_mask=args.require_mask,
        strict_binary_masks=args.require_mask,
        strict_shape_check=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = _load_checkpoint(Path(args.checkpoint), device)
    model = UNet(
        in_channels=3,
        out_channels=1,
        norm_type=str(ckpt.get("norm_type", "batch")),
        group_norm_groups=int(ckpt.get("group_norm_groups", 8)),
    ).to(device)
    _load_model_state(model, ckpt)
    model.eval()

    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    sweep_thresholds = _parse_threshold_list(args.threshold_sweep)
    threshold_metrics: dict[float, list[dict[str, float]]] = {thr: [] for thr in sweep_thresholds}

    total_images = len(dataset)
    print(f"Running inference on {total_images:,} images...", flush=True)

    with torch.no_grad():
        for images, masks, metas in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).float()

            for i in range(images.shape[0]):
                if len(rows) % 1000 == 0 and len(rows) > 0:
                    print(f"  {len(rows):,} / {total_images:,} ({100 * len(rows) / total_images:.1f}%)", flush=True)
                meta = {k: metas[k][i] for k in metas}
                basename = str(meta.get("basename", f"sample_{len(rows)}"))
                port_slug = basename_to_port_slug(basename, str(meta.get("image_path", "")))
                key = meta.get("key", "") or basename or f"sample_{len(rows)}"
                key_path = Path(str(key))

                prob_pre_np = probs[i, 0].detach().cpu().numpy().astype("float32")
                pred_pre_np = preds[i, 0].detach().cpu().numpy().astype("uint8")
                prob_post_np = prob_pre_np.copy()
                pred_post_np = pred_pre_np.copy()

                roi_mask_path = ""
                pred_pre_path = ""
                prob_pre_path = ""
                prob_post_path = ""
                roi_result = None

                if roi_resolver is not None:
                    roi_result = roi_resolver.resolve(
                        basename=basename,
                        patch_id=str(meta.get("patch_id", "")),
                        image_size=int(pred_pre_np.shape[0]),
                    )
                    if roi_result.mask is not None:
                        roi_mask_np = roi_result.mask.astype("float32")
                        prob_post_np = prob_pre_np * roi_mask_np
                        pred_post_np = (pred_pre_np.astype("float32") * roi_mask_np).astype("uint8")
                        if save_images:
                            roi_mask_path = str(roi_mask_dir / key_path.parent / f"{key_path.name}_roi_mask.png")
                            _save_binary_mask(Path(roi_mask_path), roi_result.mask)

                    if save_images:
                        pred_pre_path = str(pred_mask_pre_dir / key_path.parent / f"{key_path.name}_pred_pre_roi.png")
                        prob_pre_path = str(prob_pre_dir / key_path.parent / f"{key_path.name}_prob_pre_roi.png")
                        prob_post_path = str(prob_post_dir / key_path.parent / f"{key_path.name}_prob_post_roi.png")
                        _save_binary_mask(Path(pred_pre_path), pred_pre_np)
                        _save_probability_map(Path(prob_pre_path), prob_pre_np)
                        _save_probability_map(Path(prob_post_path), prob_post_np)

                pred_path = pred_mask_dir / key_path.parent / f"{key_path.name}_pred.png"
                if save_images:
                    _save_binary_mask(pred_path, pred_post_np)

                pixels_pre = int(pred_pre_np.sum())
                total = int(pred_pre_np.shape[0] * pred_pre_np.shape[1])
                ratio_pre = float(pixels_pre / max(total, 1))
                pixels_post = int(pred_post_np.sum())
                ratio_post = float(pixels_post / max(total, 1))

                row: dict[str, Any] = {
                    "key": meta.get("key", ""),
                    "basename": basename,
                    "image_path": meta.get("image_path", ""),
                    "mask_path": meta.get("mask_path", ""),
                    "pred_mask_path": str(pred_path),
                    "port_id": meta.get("port_id", "unknown"),
                    "patch_id": meta.get("patch_id", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "pred_container_pixels": pixels_post,
                    "pred_container_ratio": ratio_post,
                    "image_height": int(pred_post_np.shape[0]),
                    "image_width": int(pred_post_np.shape[1]),
                }

                if roi_resolver is not None and roi_result is not None:
                    row.update(
                        {
                            "port_slug": port_slug,
                            "pred_container_pixels_pre_roi": pixels_pre,
                            "pred_container_ratio_pre_roi": ratio_pre,
                            "pred_mask_pre_roi_path": pred_pre_path,
                            "pred_mask_post_roi_path": str(pred_path),
                            "prob_map_pre_roi_path": prob_pre_path,
                            "prob_map_post_roi_path": prob_post_path,
                            "roi_mask_path": roi_mask_path,
                            "roi_port_mapped": bool(roi_result.roi_port_mapped),
                            "roi_available": bool(roi_result.roi_available),
                            "roi_mask_empty": bool(roi_result.roi_mask_empty),
                            "roi_coverage_ratio": float(roi_result.roi_coverage_ratio),
                            "roi_buffer_px": int(roi_result.roi_buffer_px),
                            "roi_missing_reason": roi_result.roi_missing_reason,
                            "roi_polygon_port_name": roi_result.polygon_port_name,
                            "roi_bbox_port_name": roi_result.bbox_port_name,
                            "roi_polygon_count": int(roi_result.roi_polygon_count),
                        }
                    )

                if args.require_mask:
                    target = masks[i : i + 1]
                    pred_pre_tensor = preds[i : i + 1]
                    pred_post_tensor = torch.from_numpy(pred_post_np).to(device=device, dtype=target.dtype)[None, None, ...]

                    metrics_post = _compute_metrics(pred_post_tensor, target)
                    row["dice"] = metrics_post["dice"]
                    row["iou"] = metrics_post["iou"]
                    row["precision"] = metrics_post["precision"]
                    row["recall"] = metrics_post["recall"]
                    metric_rows.append(metrics_post)

                    if roi_resolver is not None:
                        metrics_pre = _compute_metrics(pred_pre_tensor, target)
                        row["dice_pre_roi"] = metrics_pre["dice"]
                        row["iou_pre_roi"] = metrics_pre["iou"]
                        row["precision_pre_roi"] = metrics_pre["precision"]
                        row["recall_pre_roi"] = metrics_pre["recall"]
                        row["dice_post_roi"] = metrics_post["dice"]
                        row["iou_post_roi"] = metrics_post["iou"]
                        row["precision_post_roi"] = metrics_post["precision"]
                        row["recall_post_roi"] = metrics_post["recall"]

                    for thr in sweep_thresholds:
                        thr_pred = (probs[i : i + 1] >= thr).float()
                        if roi_result is not None and roi_result.mask is not None:
                            roi_tensor = torch.from_numpy(roi_result.mask).to(device=device, dtype=thr_pred.dtype)[None, None, ...]
                            thr_pred = thr_pred * roi_tensor
                        threshold_metrics[thr].append(_compute_metrics(thr_pred, target))

                rows.append(row)

    fieldnames = [
        "key",
        "basename",
        "image_path",
        "mask_path",
        "pred_mask_path",
        "port_id",
        "patch_id",
        "timestamp",
        "pred_container_pixels",
        "pred_container_ratio",
        "image_height",
        "image_width",
        "dice",
        "iou",
        "precision",
        "recall",
    ]
    optional_keys = ["dice", "iou", "precision", "recall"]

    if roi_resolver is not None:
        roi_fieldnames = [
            "port_slug",
            "pred_container_pixels_pre_roi",
            "pred_container_ratio_pre_roi",
            "pred_mask_pre_roi_path",
            "pred_mask_post_roi_path",
            "prob_map_pre_roi_path",
            "prob_map_post_roi_path",
            "roi_mask_path",
            "roi_port_mapped",
            "roi_available",
            "roi_mask_empty",
            "roi_coverage_ratio",
            "roi_buffer_px",
            "roi_missing_reason",
            "roi_polygon_port_name",
            "roi_bbox_port_name",
            "roi_polygon_count",
            "dice_pre_roi",
            "iou_pre_roi",
            "precision_pre_roi",
            "recall_pre_roi",
            "dice_post_roi",
            "iou_post_roi",
            "precision_post_roi",
            "recall_post_roi",
        ]
        fieldnames.extend(roi_fieldnames)
        optional_keys.extend(
            [
                "port_slug",
                "pred_container_pixels_pre_roi",
                "pred_container_ratio_pre_roi",
                "pred_mask_pre_roi_path",
                "pred_mask_post_roi_path",
                "prob_map_pre_roi_path",
                "prob_map_post_roi_path",
                "roi_mask_path",
                "roi_port_mapped",
                "roi_available",
                "roi_mask_empty",
                "roi_coverage_ratio",
                "roi_buffer_px",
                "roi_missing_reason",
                "roi_polygon_port_name",
                "roi_bbox_port_name",
                "roi_polygon_count",
                "dice_pre_roi",
                "iou_pre_roi",
                "precision_pre_roi",
                "recall_pre_roi",
                "dice_post_roi",
                "iou_post_roi",
                "precision_post_roi",
                "recall_post_roi",
            ]
        )

    for row in rows:
        for optional_key in optional_keys:
            row.setdefault(optional_key, "")

    predictions_csv = out_dir / "predictions.csv"
    write_csv(predictions_csv, rows, fieldnames)

    summary: dict[str, Any] = {
        "samples": len(rows),
        "predictions_csv": str(predictions_csv),
        "pred_mask_dir": str(pred_mask_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device_used": str(device),
        "img_size": args.img_size,
    }

    if metric_rows:
        summary.update(
            {
                "avg_dice": _avg_metric(metric_rows, "dice"),
                "avg_iou": _avg_metric(metric_rows, "iou"),
                "avg_precision": _avg_metric(metric_rows, "precision"),
                "avg_recall": _avg_metric(metric_rows, "recall"),
            }
        )
        if sweep_thresholds:
            threshold_rows: list[dict[str, float]] = []
            for thr in sweep_thresholds:
                rows_for_threshold = threshold_metrics[thr]
                threshold_rows.append(
                    {
                        "threshold": float(thr),
                        "avg_dice": _avg_metric(rows_for_threshold, "dice"),
                        "avg_iou": _avg_metric(rows_for_threshold, "iou"),
                        "avg_precision": _avg_metric(rows_for_threshold, "precision"),
                        "avg_recall": _avg_metric(rows_for_threshold, "recall"),
                    }
                )
            threshold_sweep_csv = out_dir / "threshold_sweep.csv"
            write_csv(
                threshold_sweep_csv,
                threshold_rows,
                ["threshold", "avg_dice", "avg_iou", "avg_precision", "avg_recall"],
            )
            best_threshold_row = max(threshold_rows, key=lambda row: row["avg_dice"])
            summary["threshold_sweep_csv"] = str(threshold_sweep_csv)
            summary["best_threshold_by_dice"] = float(best_threshold_row["threshold"])
            summary["best_threshold_avg_dice"] = float(best_threshold_row["avg_dice"])

    if roi_resolver is not None:
        roi_summary_rows, roi_summary_json = _build_roi_summary_rows(rows)
        roi_summary_csv = out_dir / "roi_summary.csv"
        write_csv(
            roi_summary_csv,
            roi_summary_rows,
            [
                "scope",
                "port_slug",
                "port_id",
                "polygon_port_name",
                "num_cases",
                "num_cases_port_mapped",
                "num_cases_with_roi",
                "num_cases_without_roi",
                "num_cases_empty_roi",
                "avg_roi_coverage_ratio",
                "avg_dice_pre_roi",
                "avg_dice_post_roi",
                "delta_dice",
                "avg_iou_pre_roi",
                "avg_iou_post_roi",
                "delta_iou",
                "avg_precision_pre_roi",
                "avg_precision_post_roi",
                "delta_precision",
                "avg_recall_pre_roi",
                "avg_recall_post_roi",
                "delta_recall",
            ],
        )
        roi_summary_json_path = out_dir / "roi_summary.json"
        roi_summary_json_path.write_text(json.dumps(roi_summary_json, indent=2), encoding="utf-8")
        summary.update(
            {
                "apply_roi": True,
                "roi_polygons_path": str(roi_resolver.polygons_path),
                "roi_patch_bboxes_path": str(roi_resolver.patch_bboxes_path),
                "roi_port_map_path": str(roi_resolver.port_map_path) if roi_resolver.port_map_path else "",
                "roi_port_map_used_json": str(roi_port_map_used_path) if roi_port_map_used_path else "",
                "roi_summary_csv": str(roi_summary_csv),
                "roi_summary_json": str(roi_summary_json_path),
                "roi_mask_dir": str(roi_mask_dir),
                "pred_mask_pre_roi_dir": str(pred_mask_pre_dir),
            }
        )
        summary.update(roi_summary_json)

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        summary["checkpoint_epoch"] = int(ckpt["epoch"])

    summary_path = out_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    result = {
        "predictions_csv": str(predictions_csv),
        "summary_json": str(summary_path),
        "pred_mask_dir": str(pred_mask_dir),
    }
    if roi_resolver is not None:
        result["roi_summary_csv"] = str(out_dir / "roi_summary.csv")
        result["roi_summary_json"] = str(out_dir / "roi_summary.json")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net inference and export prediction masks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--require-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, compute per-image Dice/IoU/Precision/Recall.",
    )
    parser.add_argument(
        "--apply-roi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, mask predictions outside buffered ROI polygons at inference time.",
    )
    parser.add_argument("--roi-polygons-path", default="", help="Optional path to Havne_koor polygon file.")
    parser.add_argument(
        "--roi-patch-bboxes-path",
        default="",
        help="Optional path to a patch_bboxes_*.txt, a bbox manifest.csv, or a directory containing ROI bbox sources.",
    )
    parser.add_argument("--roi-port-map-path", default="", help="Optional explicit ROI port mapping JSON/CSV.")
    parser.add_argument("--roi-buffer-px", type=int, default=10, help="Hard-mask dilation buffer in pixels.")
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        default=False,
        help="Skip writing prediction/probability/ROI mask PNGs. Only predictions.csv is saved. Much faster for index building.",
    )
    return parser.parse_args()


def main() -> None:
    result = run_inference(parse_args())
    print("Inference complete")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
