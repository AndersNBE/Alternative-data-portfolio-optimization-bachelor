import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_ratios(train: float, val: float, test: float) -> tuple[float, float, float]:
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to > 0")
    return train / total, val / total, test / total


def _group_key_hybrid(row: dict[str, str]) -> str:
    port_id = row.get("port_id", "") or "unknown"
    timestamp = row.get("timestamp", "")
    if timestamp:
        return f"{port_id}::{timestamp}"

    key = row.get("key", "")
    if key:
        return f"{port_id}::{key}"

    basename = row.get("basename", "") or row.get("image_path", "unknown")
    return f"{port_id}::{basename}"


def _parse_timestamp(timestamp: str) -> datetime:
    value = (timestamp or "").strip()
    if not value:
        raise ValueError("Missing timestamp for temporal split")

    formats = [
        "%Y%m%dT%H%M%SZ",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {timestamp}")


def _assign_groups_by_target(
    grouped_rows: dict[str, list[dict[str, str]]],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, list[dict[str, str]]]:
    split_names = ["train", "val", "test"]
    targets = {
        split: ratios[idx] * sum(len(v) for v in grouped_rows.values())
        for idx, split in enumerate(split_names)
    }

    assigned_groups: dict[str, list[list[dict[str, str]]]] = {split: [] for split in split_names}
    counts = {split: 0 for split in split_names}

    group_items = list(grouped_rows.items())
    rng.shuffle(group_items)

    for _, rows in group_items:
        deficits = {split: targets[split] - counts[split] for split in split_names}
        best_split = max(deficits, key=lambda split: deficits[split])
        assigned_groups[best_split].append(rows)
        counts[best_split] += len(rows)

    if len(group_items) >= 3:
        # Ensure val/test are not empty when enough groups exist.
        for split in ["val", "test"]:
            if ratios[split_names.index(split)] <= 0:
                continue
            if assigned_groups[split]:
                continue

            donor = max(
                split_names,
                key=lambda s: (len(assigned_groups[s]), counts[s]),
            )
            if not assigned_groups[donor]:
                continue

            moved = assigned_groups[donor].pop()
            assigned_groups[split].append(moved)
            counts[donor] -= len(moved)
            counts[split] += len(moved)

    assigned: dict[str, list[dict[str, str]]] = {split: [] for split in split_names}
    for split in split_names:
        for group_rows in assigned_groups[split]:
            assigned[split].extend(group_rows)
    return assigned


def _split_random(
    rows: list[dict[str, str]],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, list[dict[str, str]]]:
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    n = len(rows_copy)

    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    n_test = max(0, n - n_train - n_val)

    train_rows = rows_copy[:n_train]
    val_rows = rows_copy[n_train : n_train + n_val]
    test_rows = rows_copy[n_train + n_val : n_train + n_val + n_test]

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def _assign_temporal_grouped(
    grouped_rows: dict[str, list[dict[str, str]]],
    ratios: tuple[float, float, float],
    test_start_timestamp: str,
    train_end_timestamp: str,
) -> dict[str, list[dict[str, str]]]:
    group_items: list[tuple[datetime, str, list[dict[str, str]]]] = []
    for group_key, rows in grouped_rows.items():
        timestamps = sorted({_parse_timestamp(row.get("timestamp", "")) for row in rows})
        group_items.append((timestamps[0], group_key, rows))

    group_items.sort(key=lambda item: (item[0], item[1]))

    test_start_dt = _parse_timestamp(test_start_timestamp) if test_start_timestamp else None
    train_end_dt = _parse_timestamp(train_end_timestamp) if train_end_timestamp else None
    if train_end_dt is not None and test_start_dt is not None and train_end_dt >= test_start_dt:
        raise ValueError("train_end_timestamp must be earlier than test_start_timestamp")

    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    if train_end_dt is not None or test_start_dt is not None:
        pre_test_groups: list[tuple[datetime, str, list[dict[str, str]]]] = []
        for group_dt, _, rows in group_items:
            if test_start_dt is not None and group_dt >= test_start_dt:
                split_rows["test"].extend(rows)
            else:
                pre_test_groups.append((group_dt, "", rows))

        if train_end_dt is not None:
            for group_dt, _, rows in pre_test_groups:
                if group_dt <= train_end_dt:
                    split_rows["train"].extend(rows)
                else:
                    split_rows["val"].extend(rows)
            return split_rows

        pre_test_row_count = sum(len(rows) for _, _, rows in pre_test_groups)
        train_ratio = ratios[0]
        val_ratio = ratios[1]
        denom = train_ratio + val_ratio
        if denom <= 0:
            raise ValueError("Temporal split needs train_ratio + val_ratio > 0 when using cutoffs")

        target_val = int(round(pre_test_row_count * (val_ratio / denom)))
        val_groups: list[list[dict[str, str]]] = []
        val_rows = 0
        for _, _, rows in reversed(pre_test_groups):
            if target_val > 0 and val_rows < target_val:
                val_groups.append(rows)
                val_rows += len(rows)
            else:
                split_rows["train"].extend(rows)

        for rows in reversed(val_groups):
            split_rows["val"].extend(rows)

        if (
            val_ratio > 0
            and not split_rows["val"]
            and len(pre_test_groups) >= 2
        ):
            latest_rows = pre_test_groups[-1][2]
            for row in latest_rows:
                split_rows["train"].remove(row)
            split_rows["val"].extend(latest_rows)
        return split_rows

    total_rows = sum(len(rows) for _, _, rows in group_items)
    train_target = int(round(total_rows * ratios[0]))
    val_target = int(round(total_rows * ratios[1]))

    assigned = 0
    for _, _, rows in group_items:
        if assigned < train_target:
            split_rows["train"].extend(rows)
        elif assigned < train_target + val_target:
            split_rows["val"].extend(rows)
        else:
            split_rows["test"].extend(rows)
        assigned += len(rows)

    return split_rows


def create_splits(
    pair_csv: Path,
    out_dir: Path,
    strategy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    test_start_timestamp: str = "",
    train_end_timestamp: str = "",
) -> dict[str, Any]:
    rows = _read_rows(pair_csv)
    if not rows:
        raise ValueError(f"No rows found in pair CSV: {pair_csv}")

    ratios = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)

    if strategy == "random":
        split_rows = _split_random(rows, ratios, rng)
    elif strategy == "hybrid":
        grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped_rows[_group_key_hybrid(row)].append(row)
        split_rows = _assign_groups_by_target(grouped_rows, ratios, rng)
    elif strategy == "temporal_grouped":
        grouped_rows = defaultdict(list)
        for row in rows:
            grouped_rows[_group_key_hybrid(row)].append(row)
        split_rows = _assign_temporal_grouped(
            grouped_rows=grouped_rows,
            ratios=ratios,
            test_start_timestamp=test_start_timestamp,
            train_end_timestamp=train_end_timestamp,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    fieldnames = list(rows[0].keys())
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"

    _write_rows(train_csv, split_rows["train"], fieldnames)
    _write_rows(val_csv, split_rows["val"], fieldnames)
    _write_rows(test_csv, split_rows["test"], fieldnames)

    summary = {
        "strategy": strategy,
        "seed": seed,
        "total": len(rows),
        "train": len(split_rows["train"]),
        "val": len(split_rows["val"]),
        "test": len(split_rows["test"]),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "train_end_timestamp": train_end_timestamp,
        "test_start_timestamp": test_start_timestamp,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits from pair CSV.")
    parser.add_argument("--pairs", required=True, help="Path to pair_report.csv")
    parser.add_argument("--out", required=True, help="Output directory for split CSV files")
    parser.add_argument("--strategy", choices=["hybrid", "random", "temporal_grouped"], default="hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--test-start-timestamp", default="")
    parser.add_argument("--train-end-timestamp", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_splits(
        pair_csv=Path(args.pairs).resolve(),
        out_dir=Path(args.out).resolve(),
        strategy=args.strategy,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        test_start_timestamp=args.test_start_timestamp,
        train_end_timestamp=args.train_end_timestamp,
    )
    print("Split creation complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
