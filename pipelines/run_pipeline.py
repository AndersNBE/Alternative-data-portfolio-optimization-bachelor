import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


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
FINAL_PATCH_BBOXES_REL = Path("data/inputs/patch_bboxes_final_49ports_lalb_20260527.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kør hele billed-pipelinen: patches -> download -> cloud -> cutoff -> inventory."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "patches", "download", "cloud", "cutoff", "inventory"],
        default="all",
        help="Kør hele flowet eller en enkelt del.",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument("--credentials-path", default="config/clientID.txt")
    parser.add_argument("--ports-path", default="data/inputs/Havne_koor.txt")
    parser.add_argument(
        "--ports",
        default="",
        help="Komma-separeret liste af portnavne der skal behandles. Tom = alle.",
    )
    parser.add_argument(
        "--preview-date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="Reference dato til scenevalg (YYYY-MM-DD).",
    )
    parser.add_argument("--preview-out-dir", default=str(ONEDRIVE_PREVIEW_ROOT))
    parser.add_argument("--bbox-output", default="")
    parser.add_argument(
        "--patch-bboxes-path",
        default=str(FINAL_PATCH_BBOXES_REL),
        help="Path til kanonisk patch_bboxes-fil. Default er den endelige 49-port LA/LB-fil.",
    )

    parser.add_argument("--dataset-root", default=str(ONEDRIVE_DATASET_ROOT))
    parser.add_argument(
        "--seed-from",
        default="",
        help="Valgfri mappe med allerede hentede billeder (fx lokal OneDrive sync).",
    )

    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)

    parser.add_argument("--cloud-thr-good", type=float, default=0.04)
    parser.add_argument("--cloud-clean-outputs", action="store_true")
    parser.add_argument("--cloud-dry-run", action="store_true")

    parser.add_argument("--cutoff-dry-run", action="store_true")
    parser.add_argument(
        "--cutoff-move-dir",
        default="",
        help="Valgfri mappe til at flytte cutoff-flaggede billeder til i stedet for at slette dem.",
    )
    parser.add_argument(
        "--inventory-input-path",
        default="",
        help="Valgfri mappe eller arkiv til coverage-rapport. Default er dataset_root/god.",
    )
    parser.add_argument(
        "--inventory-out-dir",
        default="data/outputs/folder_inventory_report",
        help="Outputmappe til coverage-rapport og CSV-filer.",
    )
    parser.add_argument(
        "--inventory-top-ports",
        type=int,
        default=50,
        help="Antal top-havne i bar chart og top-tabel.",
    )
    return parser.parse_args()


def resolve_path(raw: str, base: Path) -> Path:
    p = Path(os.path.expandvars(raw)).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def run_cmd(cmd: list[str], cwd: Path):
    print("\n$", " ".join(cmd))
    env = os.environ.copy()
    mpl_cache = cwd / ".mplconfig"
    xdg_cache = cwd / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    env.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def find_latest_patch_bboxes(repo_root: Path) -> Path | None:
    final_bboxes = repo_root / FINAL_PATCH_BBOXES_REL
    if final_bboxes.is_file():
        return final_bboxes

    candidates = []
    preview_dir = ONEDRIVE_PREVIEW_ROOT
    outputs_dir = repo_root / "data" / "outputs"
    if preview_dir.exists():
        candidates.extend(preview_dir.glob("patch_bboxes_*.txt"))
    if outputs_dir.exists():
        candidates.extend(outputs_dir.glob("patch_bboxes_*.txt"))
    candidates.extend(repo_root.glob("patch_bboxes_*.txt"))

    files = [p for p in candidates if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def sync_seed_dataset(seed_dir: Path, dataset_root: Path):
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed mappe findes ikke: {seed_dir}")
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"\nSync seed data: {seed_dir} -> {dataset_root}")
    rsync_bin = shutil.which("rsync")
    if rsync_bin:
        run_cmd([rsync_bin, "-a", "--ignore-existing", f"{seed_dir}/", f"{dataset_root}/"], cwd=dataset_root.parent)
        return

    copied = 0
    for src in seed_dir.rglob("*"):
        if not src.is_file():
            continue
        dst = dataset_root / src.relative_to(seed_dir)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Seed sync completed via fallback copy. Copied files: {copied}")


def main():
    args = parse_args()
    repo_root = resolve_path(args.repo_root, Path.cwd())
    python_bin = args.python_bin

    credentials_path = resolve_path(args.credentials_path, repo_root)
    ports_path = resolve_path(args.ports_path, repo_root)
    preview_out_dir = resolve_path(args.preview_out_dir, repo_root)
    dataset_root = resolve_path(args.dataset_root, repo_root)

    bbox_output = resolve_path(args.bbox_output, repo_root) if args.bbox_output else None
    patch_bboxes_path = resolve_path(args.patch_bboxes_path, repo_root) if args.patch_bboxes_path else None
    seed_from = resolve_path(args.seed_from, repo_root) if args.seed_from else None
    cutoff_move_dir = resolve_path(args.cutoff_move_dir, repo_root) if args.cutoff_move_dir else None
    inventory_input_path = (
        resolve_path(args.inventory_input_path, repo_root)
        if args.inventory_input_path
        else (dataset_root / "god").resolve()
    )
    inventory_out_dir = resolve_path(args.inventory_out_dir, repo_root)

    if args.mode in {"all", "patches"}:
        cmd = [
            python_bin,
            str(repo_root / "data" / "code" / "data_bbox.py"),
            "--credentials-path",
            str(credentials_path),
            "--ports-path",
            str(ports_path),
            "--preview-date",
            args.preview_date,
            "--out-dir",
            str(preview_out_dir),
        ]
        if args.ports:
            cmd += ["--ports", args.ports]
        if bbox_output is not None:
            cmd += ["--bbox-output", str(bbox_output)]
        run_cmd(cmd, cwd=repo_root)

    if args.mode in {"all", "download"}:
        if patch_bboxes_path is None:
            patch_bboxes_path = find_latest_patch_bboxes(repo_root)
        if patch_bboxes_path is None:
            raise RuntimeError(
                "Ingen patch_bboxes fil fundet. Kør patches-step først eller angiv --patch-bboxes-path."
            )

        if seed_from is not None:
            sync_seed_dataset(seed_from, dataset_root)

        run_cmd(
            [
                python_bin,
                str(repo_root / "data" / "code" / "Hent_billederne.py"),
                "--credentials-path",
                str(credentials_path),
                "--patch-bboxes-path",
                str(patch_bboxes_path),
                "--out-root",
                str(dataset_root),
                "--start-year",
                str(args.start_year),
                "--end-year",
                str(args.end_year),
            ] + (["--ports", args.ports] if args.ports else []),
            cwd=repo_root,
        )

    if args.mode in {"all", "cloud"}:
        cmd = [
            python_bin,
            str(repo_root / "data" / "code" / "Fjern_skyer_manila_shanghai.py"),
            "--root-dir",
            str(dataset_root),
            "--thr-good",
            str(args.cloud_thr_good),
        ]
        if args.cloud_clean_outputs:
            cmd.append("--clean-outputs")
        if args.cloud_dry_run:
            cmd.append("--dry-run")
        run_cmd(cmd, cwd=repo_root)

    if args.mode in {"all", "cutoff"}:
        cmd = [
            python_bin,
            str(repo_root / "data" / "code" / "Fjern_afskaaret.py"),
            "--root-dir",
            str(dataset_root),
        ]
        if args.cutoff_dry_run:
            cmd.append("--dry-run")
        if cutoff_move_dir is not None:
            cmd += ["--move-dir", str(cutoff_move_dir)]
        run_cmd(cmd, cwd=repo_root)

    if args.mode in {"all", "inventory"}:
        cmd = [
            python_bin,
            str(repo_root / "data" / "code" / "analyze_zip_inventory.py"),
            "--input-path",
            str(inventory_input_path),
            "--out-dir",
            str(inventory_out_dir),
            "--top-ports",
            str(args.inventory_top_ports),
        ]
        run_cmd(cmd, cwd=repo_root)

    print("\nPipeline completed.")
    print("Dataset root:", dataset_root)
    print("Usable output (after cloud+cutoff):", dataset_root / "god")
    if args.mode in {"all", "inventory"}:
        print("Coverage report:", inventory_out_dir / "report.html")


if __name__ == "__main__":
    main()
