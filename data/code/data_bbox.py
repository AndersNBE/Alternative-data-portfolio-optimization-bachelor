import os, math, io, time, argparse, re
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO_ROOT, ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(REPO_ROOT, ".cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_PREVIEW_ROOT = PROJECT_ROOT / "data" / "outputs" / "preview_optimal_patches"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Beregn optimale patches for havnepolygoner og generer patch_bboxes output."
    )
    parser.add_argument(
        "--credentials-path",
        default=str(PROJECT_ROOT / "config" / "clientID.txt"),
        help="Path til credentials fil med client_id og client_secret.",
    )
    parser.add_argument(
        "--ports-path",
        default=str(PROJECT_ROOT / "data" / "inputs" / "Havne_koor.txt"),
        help="Path til havnepolygon filen.",
    )
    parser.add_argument(
        "--preview-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Reference dato (YYYY-MM-DD) til STAC scenevalg.",
    )
    parser.add_argument("--img-size-px", type=int, default=512)
    parser.add_argument("--meters-per-pixel", type=float, default=10.0)
    parser.add_argument("--offset-step-m", type=float, default=40.0)
    parser.add_argument("--preview-px", type=int, default=1600)
    parser.add_argument("--window-minutes", type=int, default=720)
    parser.add_argument("--mosaicking", default="leastCC")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_PREVIEW_ROOT),
        help="Output mappe til previews og patch_bboxes fil.",
    )
    parser.add_argument(
        "--bbox-output",
        default="",
        help="Valgfri eksplicit outputfil til patch_bboxes.",
    )
    parser.add_argument(
        "--ports",
        default="",
        help="Komma-separeret liste af portnavne der skal behandles. Tom = alle.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Beregn kun patch_bboxes og skip Copernicus preview/scenes.",
    )
    return parser.parse_args()


def parse_selected_ports(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def filter_ports(ports: dict, selected_ports: set[str]) -> dict:
    if not selected_ports:
        return ports

    filtered = {name: value for name, value in ports.items() if name in selected_ports}
    missing = sorted(selected_ports - set(filtered))
    if missing:
        raise RuntimeError(
            "Ukendte portnavne i --ports: "
            + ", ".join(missing)
        )
    if not filtered:
        raise RuntimeError("Ingen porte matchede --ports filteret.")
    return filtered

def _matching_brace_index(txt, start):
    depth = 0
    for j in range(start, len(txt)):
        c = txt[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
    return None

def _is_terminal_polygon_mapping(value):
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(poly, (list, tuple)) for poly in value.values())

def _scan_port_blocks(txt):
    ports = {}
    for match in re.finditer(r'(?m)^\s*"([^"]+)"\s*:\s*\{', txt):
        port_name = match.group(1).strip()
        if not port_name:
            continue

        start = txt.find("{", match.start())
        end = _matching_brace_index(txt, start)
        if end is None:
            continue

        try:
            value = ast.literal_eval(txt[start:end + 1])
        except Exception:
            continue

        if _is_terminal_polygon_mapping(value):
            ports[port_name] = value
    return ports

def load_ports_polygons(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    txt = (
        txt.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("‟", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )

    key = "ports_polygons"
    i = txt.find(key)
    if i == -1:
        raise RuntimeError(f"Fandt ikke '{key}' i {path}")

    eq = txt.find("=", i)
    if eq == -1:
        raise RuntimeError(f"Fandt ikke '=' efter '{key}' i {path}")

    # find første { efter =
    start = txt.find("{", eq)
    if start == -1:
        raise RuntimeError(f"Fandt ikke '{{' efter '{key} =' i {path}")

    end = _matching_brace_index(txt, start)

    if end is None:
        raise RuntimeError(f"Kunne ikke finde matchende '}}' for dict i {path}")

    dict_str = txt[start:end + 1]

    try:
        data = ast.literal_eval(dict_str)
    except Exception as e:
        raise RuntimeError(f"Kunne ikke parse ports_polygons dict fra {path}: {e}")

    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"ports_polygons i {path} blev ikke et ikke tomt dict")

    data.update(_scan_port_blocks(txt))

    sanitized = {}
    skipped = []
    for port_name, polygons in data.items():
        if not isinstance(polygons, dict):
            continue
        clean_polygons = {}
        for polygon_name, coords in polygons.items():
            if not isinstance(coords, (list, tuple)):
                skipped.append((port_name, polygon_name, "not_list"))
                continue
            clean_coords = []
            for item in coords:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and all(isinstance(v, (int, float)) for v in item)
                ):
                    clean_coords.append((float(item[0]), float(item[1])))
            if len(clean_coords) < 3:
                skipped.append((port_name, polygon_name, f"too_few_valid_coords={len(clean_coords)}"))
                continue
            clean_polygons[polygon_name] = clean_coords
        if clean_polygons:
            sanitized[port_name] = clean_polygons

    if skipped:
        print(f"Advarsel: sprang {len(skipped)} ugyldige polygoner over fra {path}")
        for port_name, polygon_name, reason in skipped[:10]:
            print(f"  - {port_name} | {polygon_name!r} | {reason}")

    if not sanitized:
        raise RuntimeError(f"Ingen gyldige polygoner fundet i {path}")

    return sanitized


# ============================================================
# 0) DU RETTER KUN HER
# ports_polygons: port -> dict(terminal_name -> list of (lat, lon))
# Punkterne kan stå i vilkårlig rækkefølge, vi ordner dem i kode
# ============================================================
def load_client_credentials(path):
    path = Path(path)
    client_id_val = None
    client_secret_val = None

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("client_id"):
                    _, rhs = line.split("=", 1)
                    client_id_val = rhs.strip().strip('"').strip("'")
                elif line.startswith("client_secret"):
                    _, rhs = line.split("=", 1)
                    client_secret_val = rhs.strip().strip('"').strip("'")

    env_client_id = os.getenv("CDSE_CLIENT_ID", "").strip()
    env_client_secret = os.getenv("CDSE_CLIENT_SECRET", "").strip()
    if env_client_id and env_client_secret:
        client_id_val = client_id_val or env_client_id
        client_secret_val = client_secret_val or env_client_secret

    if not client_id_val or not client_secret_val:
        raise RuntimeError(
            f"Mangler credentials. Forventede clientID-fil på {path} "
            "eller miljøvariablerne CDSE_CLIENT_ID/CDSE_CLIENT_SECRET."
        )

    return client_id_val, client_secret_val

ARGS = parse_args()

CRED_PATH = Path(ARGS.credentials_path).expanduser()
client_id, client_secret = load_client_credentials(CRED_PATH)

PORTS_PATH = Path(ARGS.ports_path).expanduser()
ports_polygons = load_ports_polygons(PORTS_PATH)
ports_polygons = filter_ports(ports_polygons, parse_selected_ports(ARGS.ports))

preview_date_iso = ARGS.preview_date

# Patch krav
IMG_SIZE_PX = int(ARGS.img_size_px)
METERS_PER_PIXEL = float(ARGS.meters_per_pixel)
PATCH_L_M = IMG_SIZE_PX * METERS_PER_PIXEL  # 5120 m

# Ingen overlap, optimer kun gitter forskydning
OFFSET_STEP_M = float(ARGS.offset_step_m)  # mindre giver bedre men langsommere

# Satellit baggrund preview
PREVIEW_PX = int(ARGS.preview_px)
WINDOW_MINUTES = int(ARGS.window_minutes)
MOSAICKING = str(ARGS.mosaicking)

OUT_DIR = Path(ARGS.out_dir).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
if ARGS.bbox_output:
    BBOX_TXT_PATH = str(Path(ARGS.bbox_output).expanduser())
else:
    BBOX_TXT_PATH = str(OUT_DIR / f"patch_bboxes_{RUN_ID}.txt")
Path(BBOX_TXT_PATH).parent.mkdir(parents=True, exist_ok=True)

def append_line(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")
        f.flush()

_TOKEN_CACHE = None

def get_token_cached():
    global _TOKEN_CACHE
    if _TOKEN_CACHE is None:
        _TOKEN_CACHE = get_token()
    return _TOKEN_CACHE


# ============================================================
# 1) AUTH og Process API preview
# ============================================================

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"



STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"

def request_with_retry(method, url, *, headers=None, json_body=None, timeout=60, max_tries=10, base_sleep=1.5):
    last = ""
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
        except Exception as e:
            last = repr(e)
            time.sleep(base_sleep * attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = (r.text or "")[:200]
            time.sleep(base_sleep * attempt)
            continue
        return r
    raise RuntimeError(f"Request fejlede efter retries. url={url} last={last}")

def stac_find_best_scene_datetime(
    token,
    bbox_lonlat,
    desired_date_iso,
    *,
    search_days=30,
    limit=200,
    max_cloud=20.0,
    collection="sentinel-2-l1c",
):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    desired = datetime.fromisoformat(desired_date_iso).replace(tzinfo=timezone.utc)
    dt_from = (desired - timedelta(days=search_days)).isoformat().replace("+00:00", "Z")
    dt_to   = (desired + timedelta(days=search_days)).isoformat().replace("+00:00", "Z")

    payload = {
        "collections": [collection],
        "bbox": bbox_lonlat,
        "datetime": f"{dt_from}/{dt_to}",
        "limit": int(limit),
    }

    r = request_with_retry("POST", STAC_SEARCH_URL, headers=headers, json_body=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"STAC fejl {r.status_code}: {r.text[:400]}")

    feats = r.json().get("features", [])
    if not feats:
        return None, None, {"from": dt_from, "to": dt_to, "n": 0}

    candidates = []
    for f in feats:
        props = f.get("properties", {}) or {}
        dt = props.get("datetime")
        cc = props.get("eo:cloud_cover", None)
        if not dt:
            continue
        cc_val = float(cc) if cc is not None else 999.0
        if cc_val > float(max_cloud):
            continue
        dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        dist_days = abs((dt_obj - desired).total_seconds()) / 86400.0
        candidates.append((cc_val, dist_days, dt))

    if not candidates:
        return None, None, {"from": dt_from, "to": dt_to, "n": 0}

    candidates.sort(key=lambda t: (t[0], t[1]))
    best_cc, best_dist, best_dt = candidates[0]
    return best_dt, best_cc, {"from": dt_from, "to": dt_to, "n": len(candidates)}

def stac_list_candidates(token, bbox_lonlat, desired_date_iso, *, search_days=365, limit=500, collection="sentinel-2-l1c"):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    desired = datetime.fromisoformat(desired_date_iso).replace(tzinfo=timezone.utc)
    dt_from = (desired - timedelta(days=search_days)).isoformat().replace("+00:00", "Z")
    dt_to   = (desired + timedelta(days=search_days)).isoformat().replace("+00:00", "Z")

    payload = {
        "collections": [collection],
        "bbox": bbox_lonlat,
        "datetime": f"{dt_from}/{dt_to}",
        "limit": int(limit)
    }

    r = request_with_retry("POST", STAC_SEARCH_URL, headers=headers, json_body=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"STAC fejl {r.status_code}: {r.text[:400]}")

    feats = r.json().get("features", [])
    out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        dt = props.get("datetime")
        if not dt:
            continue
        cc = props.get("eo:cloud_cover", None)
        cc_val = float(cc) if cc is not None else 999.0
        dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        dist_days = abs((dt_obj - desired).total_seconds()) / 86400.0
        out.append({"dt": dt, "cloud": cc_val, "dist_days": dist_days})
    out.sort(key=lambda d: (d["cloud"], d["dist_days"]))
    return out, {"from": dt_from, "to": dt_to, "n": len(out)}

def pick_best_scene_fast(
    token,
    bbox_lonlat,
    desired_date_iso,
    *,
    search_days=365,
    stac_limit=300,
    max_cloud_pref=60.0,
    collection="sentinel-2-l1c",
):
    candidates, meta = stac_list_candidates(
        token,
        bbox_lonlat,
        desired_date_iso,
        search_days=search_days,
        limit=stac_limit,
        collection=collection
    )

    if not candidates:
        return None, None, meta

    filtered = [c for c in candidates if c["cloud"] <= max_cloud_pref]
    if not filtered:
        filtered = candidates

    filtered.sort(key=lambda d: (d["cloud"], d["dist_days"]))
    best = filtered[0]
    return best["dt"], best["cloud"], meta



def find_scene_datetime_adaptive(
    token,
    bbox_lonlat,
    desired_date_iso,
    *,
    cloud_targets=(5.0, 10.0, 20.0, 40.0, 60.0, 80.0),
    day_windows=(7, 14, 30, 60, 120),
    limit=300,
    collection="sentinel-2-l1c",
):
    last_meta = None
    for max_cloud in cloud_targets:
        for days in day_windows:
            dt, cc, meta = stac_find_best_scene_datetime(
                token,
                bbox_lonlat,
                desired_date_iso,
                search_days=days,
                limit=limit,
                max_cloud=max_cloud,
                collection=collection,
            )
            last_meta = meta
            if dt is not None:
                return dt, cc, {"search_days": days, "max_cloud": max_cloud, "meta": meta}
    return None, None, {"search_days": None, "max_cloud": None, "meta": last_meta}



def get_token():
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    r = requests.post(url, data=data, timeout=60)
    r.raise_for_status()
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError("Kunne ikke hente access_token")
    return tok

EVAL_RGB_UINT8 = """
//VERSION=3
function setup() {
  return {
    input: ["B04","B03","B02"],
    output: { bands: 3, sampleType: "UINT8" }
  }
}
function evaluatePixel(s) {
  return [
    2.5 * s.B04 * 255,
    2.5 * s.B03 * 255,
    2.5 * s.B02 * 255
  ]
}
"""

EVAL_RGB_UINT8_WITH_MASK = """
//VERSION=3
function setup() {
  return {
    input: ["B04","B03","B02","dataMask"],
    output: { bands: 4, sampleType: "UINT8" }
  }
}
function evaluatePixel(s) {
  var r = 2.5 * s.B04 * 255;
  var g = 2.5 * s.B03 * 255;
  var b = 2.5 * s.B02 * 255;
  var a = s.dataMask * 255;
  return [r, g, b, a];
}
"""


def fetch_preview_png(token, bbox_lonlat, *, center_dt_iso, out_px=1600, window_minutes=720, mosaicking="leastCC"):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    dt_center = datetime.fromisoformat(center_dt_iso.replace("Z", "+00:00"))
    dt_from = (dt_center - timedelta(minutes=window_minutes / 2)).isoformat().replace("+00:00", "Z")
    dt_to   = (dt_center + timedelta(minutes=window_minutes / 2)).isoformat().replace("+00:00", "Z")

    payload = {
        "input": {
            "bounds": {"bbox": bbox_lonlat, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-2-l1c",
                "dataFilter": {
                    "timeRange": {"from": dt_from, "to": dt_to},
                    "mosaickingOrder": mosaicking,
                    "maxCloudCoverage": 100
                }
            }]
        },
        "output": {
            "width": out_px,
            "height": out_px,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": EVAL_RGB_UINT8_WITH_MASK
    }

    r = request_with_retry("POST", PROCESS_URL, headers=headers, json_body=payload, timeout=300)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    nodata_ratio = float((alpha == 0).mean())

    return img, dt_from, dt_to, nodata_ratio




# ============================================================
# 2) Geo helpers, lat lon til lokal meter og tilbage
# ============================================================

def latlon_to_local_m(lat, lon, lat0, lon0):
    R = 6371000.0
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R
    return x, y

def local_m_to_latlon(x, y, lat0, lon0):
    R = 6371000.0
    lat = lat0 + math.degrees(y / R)
    lon = lon0 + math.degrees(x / (R * math.cos(math.radians(lat0))))
    return lat, lon

def normalize_polygon(poly_latlon):
    return [(float(lat), float(lon)) for (lat, lon) in poly_latlon]

# ============================================================
# 3) Punkt ordning, nearest neighbor cykel
# ============================================================

def order_polygon_nearest_neighbor(poly_latlon, lat0, lon0):
    pts = [(latlon_to_local_m(lat, lon, lat0, lon0), (lat, lon)) for (lat, lon) in poly_latlon]
    local = [p[0] for p in pts]
    orig  = [p[1] for p in pts]

    if len(orig) <= 2:
        return orig

    idx0 = min(range(len(local)), key=lambda i: (local[i][0], local[i][1]))

    used = [False] * len(local)
    order = [idx0]
    used[idx0] = True

    for _ in range(len(local) - 1):
        i = order[-1]
        xi, yi = local[i]

        best_j = None
        best_d = None
        for j in range(len(local)):
            if used[j]:
                continue
            xj, yj = local[j]
            d = (xj - xi) ** 2 + (yj - yi) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_j = j

        order.append(best_j)
        used[best_j] = True

    return [orig[i] for i in order]

# ============================================================
# 4) Polygon og square intersection
# ============================================================

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-18)
            if x_int > x:
                inside = not inside
    return inside

def seg_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def onseg(p, q, r):
        return min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if abs(o1) < 1e-12 and onseg(a, c, b): return True
    if abs(o2) < 1e-12 and onseg(a, d, b): return True
    if abs(o3) < 1e-12 and onseg(c, a, d): return True
    if abs(o4) < 1e-12 and onseg(c, b, d): return True
    return False

def poly_intersects_square(poly, sq):
    x0, y0, x1, y1 = sq
    square = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]

    for x, y in poly:
        if (x0 <= x <= x1) and (y0 <= y <= y1):
            return True

    for x, y in square:
        if point_in_poly(x, y, poly):
            return True

    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        for j in range(4):
            c = square[j]
            d = square[(j+1) % 4]
            if seg_intersect(a, b, c, d):
                return True

    return False

# ============================================================
# 5) Optimal patches uden overlap, med terminal navne
# ============================================================

def optimal_patches_for_named_polygons(named_polys_latlon, img_size_px, meters_per_pixel, offset_step_m):
    L = img_size_px * meters_per_pixel

    all_lats = [p[0] for _, poly in named_polys_latlon for p in poly]
    all_lons = [p[1] for _, poly in named_polys_latlon for p in poly]
    lat0 = float(np.mean(all_lats))
    lon0 = float(np.mean(all_lons))

    named_ordered = []
    polys_m = []
    for name, poly in named_polys_latlon:
        poly_ord = order_polygon_nearest_neighbor(poly, lat0, lon0)
        named_ordered.append((name, poly_ord))
        pm = [latlon_to_local_m(lat, lon, lat0, lon0) for (lat, lon) in poly_ord]
        polys_m.append((name, pm))

    offsets = np.arange(0, L, offset_step_m, dtype=float)

    best = None
    for dx in offsets:
        for dy in offsets:
            used = set()

            for _, poly in polys_m:
                pxs = [x for (x, _) in poly]
                pys = [y for (_, y) in poly]
                pminx, pmaxx = min(pxs), max(pxs)
                pminy, pmaxy = min(pys), max(pys)

                ix0 = math.floor((pminx - dx) / L)
                ix1 = math.floor((pmaxx - dx) / L)
                iy0 = math.floor((pminy - dy) / L)
                iy1 = math.floor((pmaxy - dy) / L)

                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        x0 = dx + ix * L
                        y0 = dy + iy * L
                        x1 = x0 + L
                        y1 = y0 + L
                        if poly_intersects_square(poly, (x0, y0, x1, y1)):
                            used.add((ix, iy))

            n = len(used)
            if best is None or n < best["n"]:
                best = {"dx": float(dx), "dy": float(dy), "n": int(n), "cells": sorted(used)}

    centers_local = []
    centers_latlon = []
    for ix, iy in best["cells"]:
        cx = best["dx"] + ix * L + 0.5 * L
        cy = best["dy"] + iy * L + 0.5 * L
        centers_local.append((cx, cy))
        lat_c, lon_c = local_m_to_latlon(cx, cy, lat0, lon0)
        centers_latlon.append((lat_c, lon_c))

    dbg = {
        "lat0": lat0,
        "lon0": lon0,
        "L": L,
        "dx": best["dx"],
        "dy": best["dy"],
        "cells": best["cells"],
        "centers_local": centers_local,
        "centers_latlon": centers_latlon,
        "named_polys_ordered": named_ordered,
        "named_polys_m": polys_m,
        "count": best["n"],
    }
    return dbg

def patch_bbox_lonlat_from_center_local(cx, cy, L, lat0, lon0):
    x0, y0 = cx - 0.5 * L, cy - 0.5 * L
    x1, y1 = cx + 0.5 * L, cy + 0.5 * L
    lat_a, lon_a = local_m_to_latlon(x0, y0, lat0, lon0)
    lat_b, lon_b = local_m_to_latlon(x1, y1, lat0, lon0)
    min_lat, max_lat = min(lat_a, lat_b), max(lat_a, lat_b)
    min_lon, max_lon = min(lon_a, lon_b), max(lon_a, lon_b)
    return [min_lon, min_lat, max_lon, max_lat]  # [minLon, minLat, maxLon, maxLat]

def apply_manual_patch_expansions(port, dbg):
    if port != "Long Beach" or not dbg["centers_local"]:
        return dbg

    L = dbg["L"]
    lat0, lon0 = dbg["lat0"], dbg["lon0"]
    base_cx, base_cy = dbg["centers_local"][0]

    manual_centers = [
        ("P3_west_of_P1", base_cx - L, base_cy),
        ("P4_south_of_P1", base_cx, base_cy - L),
    ]

    existing = {(round(cx, 6), round(cy, 6)) for cx, cy in dbg["centers_local"]}
    added = []
    for label, cx, cy in manual_centers:
        key = (round(cx, 6), round(cy, 6))
        if key in existing:
            continue

        dbg["centers_local"].append((cx, cy))
        dbg["centers_latlon"].append(local_m_to_latlon(cx, cy, lat0, lon0))
        existing.add(key)
        added.append(label)

    dbg["count"] = len(dbg["centers_local"])
    dbg["manual_patch_expansions"] = added
    return dbg

def bbox_lonlat_for_preview(dbg, margin_patches=0.6):
    L = dbg["L"]
    lat0, lon0 = dbg["lat0"], dbg["lon0"]

    xs = []
    ys = []
    for _, poly in dbg["named_polys_m"]:
        xs += [x for (x, _) in poly]
        ys += [y for (_, y) in poly]
    for (cx, cy) in dbg["centers_local"]:
        xs += [cx - 0.5*L, cx + 0.5*L]
        ys += [cy - 0.5*L, cy + 0.5*L]

    m = margin_patches * L
    xmin, xmax = min(xs) - m, max(xs) + m
    ymin, ymax = min(ys) - m, max(ys) + m

    lat_a, lon_a = local_m_to_latlon(xmin, ymin, lat0, lon0)
    lat_b, lon_b = local_m_to_latlon(xmax, ymax, lat0, lon0)

    min_lat, max_lat = min(lat_a, lat_b), max(lat_a, lat_b)
    min_lon, max_lon = min(lon_a, lon_b), max(lon_a, lon_b)
    return [min_lon, min_lat, max_lon, max_lat]

# ============================================================
# 6) Plot med satellit baggrund, polygon labels, patch labels, og udskrift af bbox
# ============================================================

def latlon_to_pixel(lat, lon, bbox, W, H):
    min_lon, min_lat, max_lon, max_lat = bbox
    x = (lon - min_lon) / (max_lon - min_lon) * W
    y = (max_lat - lat) / (max_lat - min_lat) * H
    return x, y

def polygon_centroid_latlon(poly_latlon):
    # simpelt gennemsnit, stabilt nok til labels
    lats = [p[0] for p in poly_latlon]
    lons = [p[1] for p in poly_latlon]
    return float(np.mean(lats)), float(np.mean(lons))

def plot_port(port, dbg, desired_date_iso):
    token = get_token_cached()
    preview_bbox = bbox_lonlat_for_preview(dbg, margin_patches=0.6)

    best_dt, best_cc, meta = pick_best_scene_fast(
        token,
        preview_bbox,
        desired_date_iso,
        search_days=365,
        stac_limit=300,
        max_cloud_pref=60.0,
        collection="sentinel-2-l1c",
    )

    if best_dt is None:
        raise RuntimeError(f"Ingen STAC kandidater fundet for port={port}")

    img, dt_from, dt_to, nodata_ratio = fetch_preview_png(
        token,
        preview_bbox,
        center_dt_iso=best_dt,
        out_px=PREVIEW_PX,
        window_minutes=WINDOW_MINUTES,
        mosaicking=MOSAICKING
    )

    W, H = img.size
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")

    for term_name, poly in dbg["named_polys_ordered"]:
        pts = [latlon_to_pixel(lat, lon, preview_bbox, W, H) for (lat, lon) in poly]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, linewidth=3)

        c_lat, c_lon = polygon_centroid_latlon(poly)
        tx, ty = latlon_to_pixel(c_lat, c_lon, preview_bbox, W, H)
        ax.text(
            tx, ty, term_name,
            fontsize=12, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.60, edgecolor="none", boxstyle="round,pad=0.25")
        )

    L = dbg["L"]
    lat0, lon0 = dbg["lat0"], dbg["lon0"]

    patch_bboxes = []
    for k, (cx, cy) in enumerate(dbg["centers_local"], start=1):
        bb = patch_bbox_lonlat_from_center_local(cx, cy, L, lat0, lon0)
        patch_bboxes.append(bb)

        min_lon, min_lat, max_lon, max_lat = bb
        x0, y0 = latlon_to_pixel(max_lat, min_lon, preview_bbox, W, H)
        x1, y1 = latlon_to_pixel(min_lat, max_lon, preview_bbox, W, H)

        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2, edgecolor="red")
        ax.add_patch(rect)

        ax.text(
            x0 + 6, y0 + 14, f"P{k}",
            fontsize=11, ha="left", va="top",
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.15")
        )

    ax.set_title(
        f"{port} patches={dbg['count']} L={int(L)}m "
        f"scene_dt={best_dt} eoCC={best_cc:.1f}% nodata={nodata_ratio*100:.1f}% "
        f"STAC_n={meta['n']} window={meta['from']}..{meta['to']}"
    )

    out_path = os.path.join(OUT_DIR, f"{port}__preview_{best_dt.replace(':','').replace('-','')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return patch_bboxes, out_path, best_dt, best_cc, nodata_ratio, meta, (dt_from, dt_to)




# ============================================================
# 7) KØR

# ============================================================
def main():
    print("Output folder:", os.path.abspath(OUT_DIR))
    print("Patch krav check, L meter:", PATCH_L_M, "IMG_SIZE_PX:", IMG_SIZE_PX, "METERS_PER_PIXEL:", METERS_PER_PIXEL)
    if ARGS.skip_preview:
        print("Preview mode: skipped (offline bbox generation)")

    for port, term_dict in ports_polygons.items():
        named_polys = []
        for term_name, poly in term_dict.items():
            named_polys.append((term_name, normalize_polygon(poly)))

        dbg = optimal_patches_for_named_polygons(
            named_polys_latlon=named_polys,
            img_size_px=IMG_SIZE_PX,
            meters_per_pixel=METERS_PER_PIXEL,
            offset_step_m=OFFSET_STEP_M
        )
        dbg = apply_manual_patch_expansions(port, dbg)

        print("\nPORT:", port)
        print("Optimal patches:", dbg["count"], "best dx m:", int(dbg["dx"]), "best dy m:", int(dbg["dy"]))
        if dbg.get("manual_patch_expansions"):
            print("Manual patch expansions:", ", ".join(dbg["manual_patch_expansions"]))
        if ARGS.skip_preview:
            L = dbg["L"]
            lat0, lon0 = dbg["lat0"], dbg["lon0"]
            patch_bboxes = [
                patch_bbox_lonlat_from_center_local(cx, cy, L, lat0, lon0)
                for (cx, cy) in dbg["centers_local"]
            ]
            out_path = ""
            best_dt = "SKIPPED_PREVIEW"
            best_cc = float("nan")
            nodata_ratio = float("nan")
            meta = {"from": "", "to": "", "n": 0}
            dt_from, dt_to = "", ""
        else:
            patch_bboxes, out_path, best_dt, best_cc, nodata_ratio, meta, (dt_from, dt_to) = plot_port(port, dbg, preview_date_iso)

        append_line(BBOX_TXT_PATH, f"PORT: {port}")
        append_line(BBOX_TXT_PATH, f"count: {dbg['count']} dx_m: {int(dbg['dx'])} dy_m: {int(dbg['dy'])}")
        if dbg.get("manual_patch_expansions"):
            append_line(BBOX_TXT_PATH, "manual_patch_expansions: " + ", ".join(dbg["manual_patch_expansions"]))
        append_line(BBOX_TXT_PATH, f"scene_dt: {best_dt} eo_cloud_cover: {best_cc}")
        append_line(BBOX_TXT_PATH, f"preview_nodata_ratio: {nodata_ratio:.6f}")
        append_line(BBOX_TXT_PATH, f"stac_window_from: {meta['from']}")
        append_line(BBOX_TXT_PATH, f"stac_window_to: {meta['to']}")
        append_line(BBOX_TXT_PATH, f"preview_time_from: {dt_from}")
        append_line(BBOX_TXT_PATH, f"preview_time_to: {dt_to}")
        append_line(BBOX_TXT_PATH, "patch_bboxes [minLon, minLat, maxLon, maxLat]:")

        for k, bb in enumerate(patch_bboxes, start=1):
            append_line(BBOX_TXT_PATH, f"  P{k}: {bb}")

        append_line(BBOX_TXT_PATH, "")

    print("\nDone")
    print("BBOX_FILE:", os.path.abspath(BBOX_TXT_PATH))


if __name__ == "__main__":
    main()
