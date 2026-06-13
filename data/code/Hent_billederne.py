import os
import re
import io
import csv
import time
import calendar
import ast
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
from PIL import Image


# =========================
# 0) KONFIGURATION
# =========================

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
ONEDRIVE_PREVIEW_ROOT = ONEDRIVE_BACHELOR_ROOT / "preview_optimal_patches"
FINAL_PATCH_BBOXES_PATH = PROJECT_ROOT / "data" / "inputs" / "patch_bboxes_final_49ports_lalb_20260527.txt"

CREDENTIALS_PATH = PROJECT_ROOT / "config" / "clientID.txt"
PATCH_BBOXES_PATH = FINAL_PATCH_BBOXES_PATH

# Output struktur: dataset_patches_flat/<port>/
OUT_ROOT = ONEDRIVE_DATASET_ROOT

# Periode: vælg så langt tilbage som du realistisk vil hente
START_YEAR = 2017
END_YEAR = datetime.now(timezone.utc).year

# Patch størrelse
IMG_SIZE = 512

# STAC search begrænsning, 500 er rigeligt for månedlige queries
STAC_LIMIT = 500

# Process API parametre
WINDOW_MINUTES = 180
MOSAICKING = "mostRecent"  # matcher din tidligere code

# Cloud metadata
# maxCloudCoverage i process er procent 0..100, hold 100 for ikke at filtrere i APIet
MAX_CLOUD_COVERAGE_PROCESS = 100

# Hvis du vil filtrere hårdt på CLP cloud ratio, sæt fx 0.5
# 1.0 betyder gem alt uanset cloud ratio, men stadig beregn og log hvis muligt
CLOUD_RATIO_MAX = 1.0

# Nodata filter baseret på datamask
# 0.98 betyder skip kun hvis næsten alt er nodata
NODATA_RATIO_MAX = 0.98

# Lidt rate limiting
SLEEP_SECONDS = 0.2

# Endpoints
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hent patch billeder fra CDSE ud fra patch_bboxes fil."
    )
    parser.add_argument(
        "--credentials-path",
        default=str(PROJECT_ROOT / "config" / "clientID.txt"),
        help="Path til credentials fil. Alternativt brug CDSE_CLIENT_ID/CDSE_CLIENT_SECRET env vars.",
    )
    parser.add_argument(
        "--patch-bboxes-path",
        default=str(FINAL_PATCH_BBOXES_PATH),
        help="Path til patch_bboxes_*.txt. Default er den endelige 49-port LA/LB-fil.",
    )
    parser.add_argument(
        "--out-root",
        default=str(ONEDRIVE_DATASET_ROOT),
        help="Output root mappe.",
    )
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--stac-limit", type=int, default=500)
    parser.add_argument("--window-minutes", type=int, default=180)
    parser.add_argument("--mosaicking", default="mostRecent")
    parser.add_argument("--max-cloud-coverage-process", type=int, default=100)
    parser.add_argument("--cloud-ratio-max", type=float, default=1.0)
    parser.add_argument("--nodata-ratio-max", type=float, default=0.98)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--ports",
        default="",
        help="Komma-separeret liste af portnavne der skal downloades. Tom = alle.",
    )
    return parser.parse_args()


# =========================
# 1) HJÆLPERE, PARSING
# =========================

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "port"


def parse_selected_ports(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def filter_ports(ports: dict[str, list[tuple[str, list[float]]]], selected_ports: set[str]) -> dict[str, list[tuple[str, list[float]]]]:
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

def month_range(year: int, month: int) -> tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    end = datetime(year, month, last, 23, 59, 59, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return start, end

def parse_dt(dt_iso: str) -> datetime:
    # dt kan være med Z og mikrosekunder
    return datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))

def stamp_from_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def request_with_retry(method, url, *, headers=None, json_body=None, data_body=None, timeout=60, max_tries=10, base_sleep=1.5):
    last = ""
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.request(method, url, headers=headers, json=json_body, data=data_body, timeout=timeout)
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

def load_client_credentials(path: Path) -> tuple[str, str]:
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

def load_patch_bboxes(path: Path) -> dict[str, list[tuple[str, list[float]]]]:
    """
    Returnerer dict:
      port_name -> [(patch_id, [minLon, minLat, maxLon, maxLat]), ...]
    Parser format som i patch_bboxes_*.txt
    """
    ports: dict[str, list[tuple[str, list[float]]]] = {}
    current_port = None

    re_port = re.compile(r"^\s*PORT:\s*(.+?)\s*$")
    re_patch = re.compile(r"^\s*(P\d+)\s*:\s*(\[[^\]]+\])\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = re_port.match(line)
            if m:
                current_port = m.group(1).strip()
                ports.setdefault(current_port, [])
                continue

            m = re_patch.match(line)
            if m and current_port is not None:
                pid = m.group(1).strip()
                bb_str = m.group(2).strip()
                try:
                    bb = ast.literal_eval(bb_str)
                except Exception as e:
                    raise RuntimeError(f"Kunne ikke parse bbox for {current_port} {pid}: {e}")

                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    raise RuntimeError(f"Ugyldig bbox format for {current_port} {pid}: {bb}")

                bb = [float(x) for x in bb]
                min_lon, min_lat, max_lon, max_lat = bb
                if not (min_lon < max_lon and min_lat < max_lat):
                    raise RuntimeError(f"Ugyldig bbox geometri for {current_port} {pid}: {bb}")

                ports[current_port].append((pid, bb))

    ports = {p: lst for p, lst in ports.items() if lst}
    if not ports:
        raise RuntimeError(f"Fandt ingen patches i {path}")

    return ports


# =========================
# 2) AUTH TOKEN MANAGER
# =========================

@dataclass
class TokenState:
    token: str
    expires_at_epoch: float

class TokenManager:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.state: TokenState | None = None

    def get(self) -> str:
        now = time.time()
        if self.state is None or now > (self.state.expires_at_epoch - 60):
            self.state = self._refresh()
        return self.state.token

    def _refresh(self) -> TokenState:
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        r = request_with_retry("POST", TOKEN_URL, data_body=data, timeout=60)
        r.raise_for_status()
        js = r.json()
        tok = js.get("access_token")
        exp = js.get("expires_in", 3600)
        if not tok:
            raise RuntimeError("Kunne ikke hente access_token")
        return TokenState(token=tok, expires_at_epoch=time.time() + float(exp))


# =========================
# 3) STAC LISTE AF ACQUISITIONS
# =========================

def stac_list_acquisitions(token: str, bbox: list[float], dt_from: str, dt_to: str, limit: int) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "collections": ["sentinel-2-l1c"],
        "bbox": bbox,
        "datetime": f"{dt_from}/{dt_to}",
        "limit": int(limit),
    }
    r = request_with_retry("POST", STAC_SEARCH_URL, headers=headers, json_body=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"STAC fejl {r.status_code}: {r.text[:500]}")

    feats = r.json().get("features", [])
    out = []
    seen_dt = set()

    for f in feats:
        props = f.get("properties", {}) or {}
        dt = props.get("datetime")
        if not dt:
            continue
        if dt in seen_dt:
            continue
        seen_dt.add(dt)

        eo_cc = props.get("eo:cloud_cover")
        try:
            eo_cc = float(eo_cc) if eo_cc is not None else None
        except Exception:
            eo_cc = None

        out.append({"datetime": dt, "eo_cloud_cover": eo_cc})

    out.sort(key=lambda x: x["datetime"])
    return out


# =========================
# 4) PROCESS API PATCH DOWNLOAD
# =========================

EVAL_RGB_MASK = """
//VERSION=3
function setup() {
  return { input: ["B04","B03","B02","dataMask"], output: { bands: 4, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  var r = Math.max(0, Math.min(255, s.B04 * 2.5 * 255));
  var g = Math.max(0, Math.min(255, s.B03 * 2.5 * 255));
  var b = Math.max(0, Math.min(255, s.B02 * 2.5 * 255));
  var a = s.dataMask * 255;
  return [r, g, b, a];
}
"""

EVAL_CLP = """
//VERSION=3
function setup() {
  return { input: ["CLP"], output: { bands: 1, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  // CLP antages 0..1, skaleres til 0..255
  return [Math.max(0, Math.min(255, s.CLP * 255))];
}
"""

def process_request_png(token: str, bbox: list[float], dt_iso: str, evalscript: str, img_size: int, window_minutes: int, mosaicking: str):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    dt_obj = parse_dt(dt_iso)
    dt_from = (dt_obj - timedelta(minutes=window_minutes / 2)).isoformat().replace("+00:00", "Z")
    dt_to = (dt_obj + timedelta(minutes=window_minutes / 2)).isoformat().replace("+00:00", "Z")

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-2-l1c",
                "dataFilter": {
                    "timeRange": {"from": dt_from, "to": dt_to},
                    "mosaickingOrder": mosaicking,
                    "maxCloudCoverage": int(MAX_CLOUD_COVERAGE_PROCESS),
                }
            }]
        },
        "output": {
            "width": int(img_size),
            "height": int(img_size),
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": evalscript
    }

    r = request_with_retry("POST", PROCESS_URL, headers=headers, json_body=payload, timeout=300)
    ctype = (r.headers.get("Content-Type") or "").lower()
    return r.status_code, ctype, r.content

def nodata_ratio_from_rgba_png(png_bytes: bytes) -> float:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im)
    alpha = arr[:, :, 3]
    return float((alpha == 0).mean())

def to_rgb_png_bytes(png_bytes: bytes) -> bytes:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    rgb = im.convert("RGB")
    out = io.BytesIO()
    rgb.save(out, format="PNG", optimize=True)
    return out.getvalue()

def cloud_ratio_from_clp_png(png_bytes: bytes) -> float:
    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    arr = np.array(im).astype(np.float32)
    clp = arr / 255.0
    return float((clp > 0.5).mean())


# =========================
# 5) MAIN LOOP, GEM, MANIFEST
# =========================

def ensure_manifest(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "port_name",
            "port_slug",
            "patch_id",
            "datetime_iso",
            "stamp_utc",
            "eo_cloud_cover",
            "cloud_ratio_clp",
            "nodata_ratio",
            "bbox_minlon",
            "bbox_minlat",
            "bbox_maxlon",
            "bbox_maxlat",
            "img_size_px",
            "file_relpath",
        ])

def append_manifest(path: Path, row: list) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

def find_latest_patch_bboxes_path(project_root: Path) -> Path | None:
    if FINAL_PATCH_BBOXES_PATH.is_file():
        return FINAL_PATCH_BBOXES_PATH

    candidates = []

    preview_dir = ONEDRIVE_PREVIEW_ROOT
    outputs_dir = project_root / "data" / "outputs"
    if preview_dir.exists():
        candidates.extend(preview_dir.glob("patch_bboxes_*.txt"))

    if outputs_dir.exists():
        candidates.extend(outputs_dir.glob("patch_bboxes_*.txt"))
    candidates.extend(project_root.glob("patch_bboxes_*.txt"))

    files = [p for p in candidates if p.is_file()]
    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def apply_config(args):
    global CREDENTIALS_PATH
    global PATCH_BBOXES_PATH
    global OUT_ROOT
    global START_YEAR, END_YEAR
    global IMG_SIZE
    global STAC_LIMIT
    global WINDOW_MINUTES
    global MOSAICKING
    global MAX_CLOUD_COVERAGE_PROCESS
    global CLOUD_RATIO_MAX
    global NODATA_RATIO_MAX
    global SLEEP_SECONDS

    CREDENTIALS_PATH = Path(os.path.expandvars(args.credentials_path)).expanduser()
    OUT_ROOT = Path(os.path.expandvars(args.out_root)).expanduser()

    START_YEAR = int(args.start_year)
    END_YEAR = int(args.end_year)
    IMG_SIZE = int(args.img_size)
    STAC_LIMIT = int(args.stac_limit)
    WINDOW_MINUTES = int(args.window_minutes)
    MOSAICKING = str(args.mosaicking)
    MAX_CLOUD_COVERAGE_PROCESS = int(args.max_cloud_coverage_process)
    CLOUD_RATIO_MAX = float(args.cloud_ratio_max)
    NODATA_RATIO_MAX = float(args.nodata_ratio_max)
    SLEEP_SECONDS = float(args.sleep_seconds)

    if args.patch_bboxes_path:
        PATCH_BBOXES_PATH = Path(os.path.expandvars(args.patch_bboxes_path)).expanduser()
    else:
        latest = find_latest_patch_bboxes_path(PROJECT_ROOT)
        if latest is None:
            raise RuntimeError(
                "Kunne ikke finde patch_bboxes_*.txt automatisk. "
                "Brug --patch-bboxes-path."
            )
        PATCH_BBOXES_PATH = latest


def main(args):
    apply_config(args)

    print("BASE_DIR:", str(BASE_DIR))
    print("CREDENTIALS_PATH:", str(CREDENTIALS_PATH))
    print("PATCH_BBOXES_PATH:", str(PATCH_BBOXES_PATH))
    print("OUT_ROOT:", str(OUT_ROOT))

    client_id, client_secret = load_client_credentials(CREDENTIALS_PATH)
    ports = load_patch_bboxes(PATCH_BBOXES_PATH)
    ports = filter_ports(ports, parse_selected_ports(args.ports))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "manifest.csv"
    ensure_manifest(manifest_path)

    tm = TokenManager(client_id, client_secret)

    total_attempts = 0
    total_saved = 0

    for port_name, patches in ports.items():
        port_slug = slugify(port_name)
        port_dir = OUT_ROOT / port_slug
        port_dir.mkdir(parents=True, exist_ok=True)

        print("\nPORT:", port_name, "patches:", len(patches), "folder:", str(port_dir))

        for patch_id, bbox in patches:
            print("  PATCH:", patch_id, "bbox:", bbox)

            for year in range(int(START_YEAR), int(END_YEAR) + 1):
                for month in range(1, 13):
                    dt_from, dt_to = month_range(year, month)
                    token = tm.get()

                    acqs = stac_list_acquisitions(token, bbox, dt_from, dt_to, limit=STAC_LIMIT)
                    if not acqs:
                        continue

                    print(f"    {year:04d} {month:02d} acquisitions:", len(acqs))

                    for item in acqs:
                        total_attempts += 1
                        dt_iso = item["datetime"]
                        eo_cc = item["eo_cloud_cover"]

                        dt_obj = parse_dt(dt_iso)
                        stamp = stamp_from_dt(dt_obj)

                        eo_str = "NA" if eo_cc is None else f"{eo_cc:.3f}"

                        # Filnavn, inkluderer port, patch, timestamp og cloud metadata
                        # Nodata og CLP cloud ratio tilføjes senere
                        prefix = f"{port_slug}__{patch_id}__L1C__{stamp}__eoCC{eo_str}"

                        # Hvis der allerede findes et matchende billede, skip
                        # Vi bruger glob fordi CR og ND først kendes efter requests
                        existing = list(port_dir.glob(prefix + "__CR*__ND*__" + f"{IMG_SIZE}px.png"))
                        if existing:
                            continue

                        token = tm.get()

                        # 1) RGB med mask til nodata ratio
                        s_rgb, c_rgb, b_rgb = process_request_png(
                            token=token,
                            bbox=bbox,
                            dt_iso=dt_iso,
                            evalscript=EVAL_RGB_MASK,
                            img_size=IMG_SIZE,
                            window_minutes=WINDOW_MINUTES,
                            mosaicking=MOSAICKING,
                        )

                        if s_rgb != 200 or (not c_rgb.startswith("image/png")) or len(b_rgb) < 5000:
                            print(f"      skip RGB {stamp}, status={s_rgb}, ctype={c_rgb}, bytes={len(b_rgb)}")
                            continue

                        try:
                            nd_ratio = nodata_ratio_from_rgba_png(b_rgb)
                        except Exception as e:
                            print(f"      nodata parse fejl {stamp}, {repr(e)}")
                            continue

                        if nd_ratio > NODATA_RATIO_MAX:
                            print(f"      skip nodata {stamp}, ND={nd_ratio:.3f}")
                            continue

                        # 2) CLP cloud ratio, hvis den fejler gemmer vi stadig RGB
                        cloud_ratio = None
                        try:
                            s_clp, c_clp, b_clp = process_request_png(
                                token=token,
                                bbox=bbox,
                                dt_iso=dt_iso,
                                evalscript=EVAL_CLP,
                                img_size=IMG_SIZE,
                                window_minutes=WINDOW_MINUTES,
                                mosaicking=MOSAICKING,
                            )
                            if s_clp == 200 and c_clp.startswith("image/png") and len(b_clp) >= 2000:
                                cloud_ratio = cloud_ratio_from_clp_png(b_clp)
                            else:
                                cloud_ratio = None
                        except Exception:
                            cloud_ratio = None

                        if cloud_ratio is not None and cloud_ratio > CLOUD_RATIO_MAX:
                            print(f"      skip cloud {stamp}, CR={cloud_ratio:.3f}")
                            continue

                        cr_str = "NA" if cloud_ratio is None else f"{cloud_ratio:.3f}"
                        nd_str = f"{nd_ratio:.3f}"

                        # Konverter til RGB før vi gemmer, så dine træningsbilleder er konsistente
                        try:
                            rgb_bytes = to_rgb_png_bytes(b_rgb)
                        except Exception as e:
                            print(f"      RGB convert fejl {stamp}, {repr(e)}")
                            continue

                        fname = f"{prefix}__CR{cr_str}__ND{nd_str}__{IMG_SIZE}px.png"
                        fpath = port_dir / fname

                        with open(fpath, "wb") as f:
                            f.write(rgb_bytes)

                        total_saved += 1

                        append_manifest(manifest_path, [
                            port_name,
                            port_slug,
                            patch_id,
                            dt_iso,
                            stamp,
                            eo_cc if eo_cc is not None else "",
                            cloud_ratio if cloud_ratio is not None else "",
                            nd_ratio,
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            IMG_SIZE,
                            str(fpath.relative_to(OUT_ROOT)),
                        ])

                        print("      saved:", str(fpath))
                        time.sleep(SLEEP_SECONDS)

    print("\nDONE")
    print("Attempts:", total_attempts)
    print("Saved:", total_saved)
    print("Manifest:", str((OUT_ROOT / "manifest.csv").resolve()))

if __name__ == "__main__":
    main(parse_args())
