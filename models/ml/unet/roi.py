import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


_SMART_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
    }
)

_PORT_START_RE = re.compile(r'^\s*"([^"]+)"\s*:\s*\{\s*$')
_POLYGON_START_RE = re.compile(r'^\s*"([^"]*)"\s*:\s*(.+?)\s*,?\s*$')
_COORD_RE = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")
_PATCH_LINE_RE = re.compile(r"^\s*(P\d+)\s*:\s*\[([^\]]+)\]\s*$", re.IGNORECASE)
_SHARED_PORT_SLUG_GROUPS = (
    ("los_angeles", "long_beach"),
    ("new_york", "new_jersey", "new_york_new_jersey"),
)
FINAL_PATCH_BBOXES_REL = Path("data/inputs/patch_bboxes_final_49ports_lalb_20260527.txt")


@dataclass
class ROIMaskResult:
    port_slug: str
    port_id: str
    polygon_port_name: str
    bbox_port_name: str
    patch_id: str
    roi_port_mapped: bool
    roi_available: bool
    roi_mask_empty: bool
    roi_coverage_ratio: float
    roi_buffer_px: int
    roi_missing_reason: str
    roi_polygon_count: int
    mask: np.ndarray | None


def slugify_port_name(name: str) -> str:
    value = (name or "").strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "", value)
    return value or "port"


def infer_port_id_from_slug(port_slug: str) -> str:
    tokens = [token for token in (port_slug or "").split("_") if token]
    return tokens[0] if tokens else "unknown"


def basename_to_port_slug(basename: str, image_path: str = "") -> str:
    stem = (basename or "").strip()
    if stem and "__" in stem:
        return stem.split("__", 1)[0]
    if stem:
        return stem
    image_stem = Path(image_path).stem
    if "__" in image_stem:
        return image_stem.split("__", 1)[0]
    return image_stem or "unknown"


def default_polygons_path(repo_root: Path) -> Path:
    return repo_root / "data" / "inputs" / "Havne_koor.txt"


def default_bbox_source(repo_root: Path) -> Path:
    final_bboxes = repo_root / FINAL_PATCH_BBOXES_REL
    if final_bboxes.is_file():
        return final_bboxes
    preview_dir = repo_root / "data" / "outputs" / "preview_optimal_patches"
    if preview_dir.exists():
        return preview_dir
    outputs_dir = repo_root / "data" / "outputs"
    if outputs_dir.exists():
        return outputs_dir
    return repo_root


def _normalize_text(text: str) -> str:
    return text.translate(_SMART_QUOTES)


def _join_unique(values: list[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = (value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return " + ".join(ordered)


def _build_shared_port_group_index() -> dict[str, tuple[str, ...]]:
    index: dict[str, tuple[str, ...]] = {}
    for raw_group in _SHARED_PORT_SLUG_GROUPS:
        group = tuple(sorted({slugify_port_name(item) for item in raw_group if str(item).strip()}))
        for slug in group:
            index[slug] = group
    return index


def load_ports_polygons(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    text = _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    ports: dict[str, dict[str, list[tuple[float, float]]]] = {}
    current_port: str | None = None
    current_polygon_name: str | None = None
    polygon_buffer: list[str] = []
    bracket_depth = 0

    def flush_polygon() -> None:
        nonlocal current_polygon_name, polygon_buffer, bracket_depth
        if current_port is None or current_polygon_name is None:
            current_polygon_name = None
            polygon_buffer = []
            bracket_depth = 0
            return

        payload = " ".join(polygon_buffer)
        coords = [(float(lat), float(lon)) for lat, lon in _COORD_RE.findall(payload)]
        if len(coords) >= 3:
            polygon_name = current_polygon_name or f"polygon_{len(ports.get(current_port, {})) + 1}"
            ports.setdefault(current_port, {})[polygon_name] = coords

        current_polygon_name = None
        polygon_buffer = []
        bracket_depth = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if current_polygon_name is not None:
            polygon_buffer.append(line)
            bracket_depth += line.count("[") - line.count("]")
            if bracket_depth <= 0:
                flush_polygon()
            continue

        port_match = _PORT_START_RE.match(line)
        if port_match:
            current_port = port_match.group(1).strip()
            ports.setdefault(current_port, {})
            continue

        if line.startswith("}"):
            current_port = None
            continue

        if current_port is None:
            continue

        polygon_match = _POLYGON_START_RE.match(line)
        if not polygon_match:
            continue

        polygon_name = polygon_match.group(1).strip() or f"polygon_{len(ports[current_port]) + 1}"
        payload = polygon_match.group(2).strip()
        if not payload.startswith("["):
            continue

        current_polygon_name = polygon_name
        polygon_buffer = [payload]
        bracket_depth = payload.count("[") - payload.count("]")
        if bracket_depth <= 0:
            flush_polygon()

    flush_polygon()
    return {port_name: polygons for port_name, polygons in ports.items() if polygons}


def load_patch_bboxes(path: Path) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    ports: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    current_port: str | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("PORT:"):
            current_port = line.split(":", 1)[1].strip()
            ports.setdefault(current_port, {})
            continue
        if current_port is None:
            continue
        match = _PATCH_LINE_RE.match(line)
        if not match:
            continue
        patch_id = match.group(1).upper()
        coords = [float(part.strip()) for part in match.group(2).split(",")]
        if len(coords) != 4:
            continue
        ports[current_port][patch_id] = tuple(coords)  # type: ignore[assignment]

    return {port_name: patches for port_name, patches in ports.items() if patches}


def load_manifest_patch_bboxes(path: Path) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    ports: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            port_name = (row.get("port_name") or row.get("port_slug") or "").strip()
            patch_id = (row.get("patch_id") or "").strip().upper()
            if not port_name or not patch_id:
                continue
            try:
                bbox = (
                    float(row["bbox_minlon"]),
                    float(row["bbox_minlat"]),
                    float(row["bbox_maxlon"]),
                    float(row["bbox_maxlat"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            ports.setdefault(port_name, {})[patch_id] = bbox
    return {port_name: patches for port_name, patches in ports.items() if patches}


def _merge_patch_bbox_maps(
    base: dict[str, dict[str, tuple[float, float, float, float]]],
    incoming: dict[str, dict[str, tuple[float, float, float, float]]],
) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    merged = {port_name: dict(patches) for port_name, patches in base.items()}
    for port_name, patches in incoming.items():
        merged.setdefault(port_name, {}).update(patches)
    return merged


def load_patch_bbox_source(path: Path) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    path = path.resolve()
    if path.is_dir():
        merged: dict[str, dict[str, tuple[float, float, float, float]]] = {}
        txt_sources = sorted(path.rglob("patch_bboxes_*.txt"))
        csv_sources = sorted(p for p in path.rglob("manifest.csv") if "dataset_patches_flat" in str(p.parent))
        for source in txt_sources:
            merged = _merge_patch_bbox_maps(merged, load_patch_bboxes(source))
        for source in csv_sources:
            merged = _merge_patch_bbox_maps(merged, load_manifest_patch_bboxes(source))
        return merged

    if path.suffix.lower() == ".txt":
        return load_patch_bboxes(path)
    if path.suffix.lower() == ".csv":
        return load_manifest_patch_bboxes(path)
    raise ValueError(f"Unsupported ROI bbox source: {path}")


def load_port_map(path: Path | None, polygon_ports: list[str]) -> dict[str, dict[str, str]]:
    base = {
        port_name: {
            "polygon_port_name": port_name,
            "port_slug": slugify_port_name(port_name),
            "port_id": infer_port_id_from_slug(slugify_port_name(port_name)),
        }
        for port_name in polygon_ports
    }
    if path is None:
        return base

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"ROI port map not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in ROI port map: {path}")
        for port_name, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict value for port map entry '{port_name}' in {path}")
            entry = base.setdefault(
                port_name,
                {
                    "polygon_port_name": port_name,
                    "port_slug": slugify_port_name(port_name),
                    "port_id": infer_port_id_from_slug(slugify_port_name(port_name)),
                },
            )
            if value.get("port_slug"):
                entry["port_slug"] = str(value["port_slug"])
            if value.get("port_id"):
                entry["port_id"] = str(value["port_id"])
            if value.get("polygon_port_name"):
                entry["polygon_port_name"] = str(value["polygon_port_name"])
        return base

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            port_name = (row.get("polygon_port_name") or row.get("port_name") or "").strip()
            if not port_name:
                continue
            entry = base.setdefault(
                port_name,
                {
                    "polygon_port_name": port_name,
                    "port_slug": slugify_port_name(port_name),
                    "port_id": infer_port_id_from_slug(slugify_port_name(port_name)),
                },
            )
            port_slug = (row.get("port_slug") or "").strip()
            port_id = (row.get("port_id") or "").strip()
            if port_slug:
                entry["port_slug"] = port_slug
            if port_id:
                entry["port_id"] = port_id
        return base

    raise ValueError(f"Unsupported ROI port map format: {path}")


class ROIResolver:
    def __init__(
        self,
        polygons_path: Path,
        patch_bboxes_path: Path,
        port_map_path: Path | None = None,
        buffer_px: int = 10,
    ) -> None:
        self.polygons_path = polygons_path.resolve()
        self.patch_bboxes_path = patch_bboxes_path.resolve()
        self.port_map_path = port_map_path.resolve() if port_map_path else None
        self.buffer_px = max(int(buffer_px), 0)

        self.polygons_by_port_name = load_ports_polygons(self.polygons_path)
        self.patch_bboxes_by_port_name = load_patch_bbox_source(self.patch_bboxes_path)
        self.port_map = load_port_map(self.port_map_path, sorted(self.polygons_by_port_name))
        self.shared_port_group_by_slug = _build_shared_port_group_index()

        self._mask_cache: dict[tuple[str, str, int], Any] = {}

        self.polygons_by_slug: dict[str, dict[str, Any]] = {}
        for port_name, polygons in self.polygons_by_port_name.items():
            entry = self.port_map.get(
                port_name,
                {
                    "polygon_port_name": port_name,
                    "port_slug": slugify_port_name(port_name),
                    "port_id": infer_port_id_from_slug(slugify_port_name(port_name)),
                },
            )
            port_slug = str(entry["port_slug"])
            self.polygons_by_slug[port_slug] = {
                "polygon_port_name": port_name,
                "port_id": str(entry["port_id"]),
                "polygons": polygons,
            }

        self.bboxes_by_slug: dict[str, dict[str, Any]] = {}
        for port_name, patches in self.patch_bboxes_by_port_name.items():
            port_slug = slugify_port_name(port_name)
            self.bboxes_by_slug[port_slug] = {
                "bbox_port_name": port_name,
                "patches": patches,
            }

    def used_port_map(self) -> dict[str, dict[str, str]]:
        return self.port_map

    def _candidate_slugs(self, port_slug: str) -> list[str]:
        normalized = slugify_port_name(port_slug)
        group = self.shared_port_group_by_slug.get(normalized)
        if group is None:
            return [normalized]
        return [normalized] + [slug for slug in group if slug != normalized]

    def _resolve_polygon_info(self, port_slug: str) -> dict[str, Any] | None:
        polygon_entries = [
            (candidate_slug, self.polygons_by_slug[candidate_slug])
            for candidate_slug in self._candidate_slugs(port_slug)
            if candidate_slug in self.polygons_by_slug
        ]
        if not polygon_entries:
            return None

        merged_polygons: dict[str, list[tuple[float, float]]] = {}
        polygon_port_names: list[str] = []
        for candidate_slug, entry in polygon_entries:
            polygon_port_names.append(str(entry["polygon_port_name"]))
            for polygon_name, coords in dict(entry["polygons"]).items():
                merged_name = polygon_name
                if merged_name in merged_polygons and merged_polygons[merged_name] != coords:
                    merged_name = f"{candidate_slug}:{polygon_name}"
                while merged_name in merged_polygons:
                    merged_name = f"{merged_name}_alt"
                merged_polygons[merged_name] = coords

        return {
            "polygon_port_name": _join_unique(polygon_port_names),
            "port_id": infer_port_id_from_slug(port_slug),
            "polygons": merged_polygons,
        }

    def _bbox_candidates(self, port_slug: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (candidate_slug, self.bboxes_by_slug[candidate_slug])
            for candidate_slug in self._candidate_slugs(port_slug)
            if candidate_slug in self.bboxes_by_slug
        ]

    def resolve(self, basename: str, patch_id: str, image_size: int) -> ROIMaskResult:
        port_slug = basename_to_port_slug(basename)
        patch = (patch_id or "").upper()

        polygon_info = self._resolve_polygon_info(port_slug)
        if polygon_info is None:
            return ROIMaskResult(
                port_slug=port_slug,
                port_id=infer_port_id_from_slug(port_slug),
                polygon_port_name="",
                bbox_port_name="",
                patch_id=patch,
                roi_port_mapped=False,
                roi_available=False,
                roi_mask_empty=False,
                roi_coverage_ratio=0.0,
                roi_buffer_px=self.buffer_px,
                roi_missing_reason="missing_polygon_port",
                roi_polygon_count=0,
                mask=None,
            )

        bbox_candidates = self._bbox_candidates(port_slug)
        if not bbox_candidates:
            return ROIMaskResult(
                port_slug=port_slug,
                port_id=str(polygon_info["port_id"]),
                polygon_port_name=str(polygon_info["polygon_port_name"]),
                bbox_port_name="",
                patch_id=patch,
                roi_port_mapped=True,
                roi_available=False,
                roi_mask_empty=False,
                roi_coverage_ratio=0.0,
                roi_buffer_px=self.buffer_px,
                roi_missing_reason="missing_bbox_port",
                roi_polygon_count=len(polygon_info["polygons"]),
                mask=None,
            )

        bbox_port_name = _join_unique([str(info["bbox_port_name"]) for _, info in bbox_candidates])
        if not patch:
            return ROIMaskResult(
                port_slug=port_slug,
                port_id=str(polygon_info["port_id"]),
                polygon_port_name=str(polygon_info["polygon_port_name"]),
                bbox_port_name=bbox_port_name,
                patch_id=patch,
                roi_port_mapped=True,
                roi_available=False,
                roi_mask_empty=False,
                roi_coverage_ratio=0.0,
                roi_buffer_px=self.buffer_px,
                roi_missing_reason="missing_patch_id",
                roi_polygon_count=len(polygon_info["polygons"]),
                mask=None,
            )

        matched_bbox_info: dict[str, Any] | None = None
        patch_bbox: tuple[float, float, float, float] | None = None
        for _, candidate_info in bbox_candidates:
            candidate_bbox = candidate_info["patches"].get(patch)
            if candidate_bbox is not None:
                matched_bbox_info = candidate_info
                patch_bbox = candidate_bbox
                break

        if patch_bbox is None or matched_bbox_info is None:
            return ROIMaskResult(
                port_slug=port_slug,
                port_id=str(polygon_info["port_id"]),
                polygon_port_name=str(polygon_info["polygon_port_name"]),
                bbox_port_name=bbox_port_name,
                patch_id=patch,
                roi_port_mapped=True,
                roi_available=False,
                roi_mask_empty=False,
                roi_coverage_ratio=0.0,
                roi_buffer_px=self.buffer_px,
                roi_missing_reason="missing_patch_bbox",
                roi_polygon_count=len(polygon_info["polygons"]),
                mask=None,
            )

        cache_key = (port_slug, patch, image_size)
        if cache_key in self._mask_cache:
            mask = self._mask_cache[cache_key]
        else:
            mask = rasterize_port_polygons(
                polygons=list(polygon_info["polygons"].values()),
                bbox=patch_bbox,
                image_size=image_size,
                buffer_px=self.buffer_px,
            )
            self._mask_cache[cache_key] = mask
        mask_empty = not bool(mask.any())
        return ROIMaskResult(
            port_slug=port_slug,
            port_id=str(polygon_info["port_id"]),
            polygon_port_name=str(polygon_info["polygon_port_name"]),
            bbox_port_name=str(matched_bbox_info["bbox_port_name"]),
            patch_id=patch,
            roi_port_mapped=True,
            roi_available=True,
            roi_mask_empty=mask_empty,
            roi_coverage_ratio=float(mask.mean()) if mask.size else 0.0,
            roi_buffer_px=self.buffer_px,
            roi_missing_reason="roi_mask_empty" if mask_empty else "ok",
            roi_polygon_count=len(polygon_info["polygons"]),
            mask=mask,
        )


def _latlon_to_pixel(lat: float, lon: float, bbox: tuple[float, float, float, float], image_size: int) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    width = max(float(max_lon - min_lon), 1e-12)
    height = max(float(max_lat - min_lat), 1e-12)
    x = ((float(lon) - float(min_lon)) / width) * (image_size - 1)
    y = ((float(max_lat) - float(lat)) / height) * (image_size - 1)
    return x, y


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique_points = sorted({(float(x), float(y)) for x, y in points})
    if len(unique_points) <= 3:
        return unique_points

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def rasterize_port_polygons(
    polygons: list[list[tuple[float, float]]],
    bbox: tuple[float, float, float, float],
    image_size: int,
    buffer_px: int,
) -> np.ndarray:
    canvas = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        pixel_points = [_latlon_to_pixel(lat, lon, bbox, image_size) for lat, lon in polygon]
        hull_points = _convex_hull(pixel_points)
        if len(hull_points) >= 3:
            draw.polygon(hull_points, fill=255)

    if buffer_px > 0:
        kernel_size = max(3, buffer_px * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        canvas = canvas.filter(ImageFilter.MaxFilter(size=kernel_size))

    return (np.asarray(canvas, dtype=np.uint8) > 0).astype(np.uint8)


def load_mask_array(path: str) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127).astype(np.uint8)


def overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray | None,
    color: tuple[int, int, int] = (48, 200, 90),
    alpha: float = 0.35,
    empty_label: str = "No ROI",
) -> Image.Image:
    base = image.convert("RGB")
    if mask is None or not np.asarray(mask).any():
        overlay = base.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((10, 10, 120, 34), fill=(180, 0, 0))
        draw.text((16, 16), empty_label, fill=(255, 255, 255))
        return overlay

    mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    color_arr = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]
    color_img = Image.fromarray(color_arr)

    blended = Image.blend(base, color_img, alpha=max(0.0, min(alpha, 1.0)))
    result = base.copy()
    result_np = np.asarray(result, dtype=np.uint8).copy()
    blended_np = np.asarray(blended, dtype=np.uint8)
    result_np[mask_arr.astype(bool)] = blended_np[mask_arr.astype(bool)]
    return Image.fromarray(result_np)
