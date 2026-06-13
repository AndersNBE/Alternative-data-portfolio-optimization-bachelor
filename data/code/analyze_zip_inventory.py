import argparse
import csv
import html
import json
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
from zipfile import BadZipFile


TIMESTAMP_RE = re.compile(r"__(\d{8}T\d{6}Z)__", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze image coverage inside an archive or folder and build an HTML report."
    )
    parser.add_argument("--input-path", required=True, help="Path to the archive or folder.")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV/JSON/HTML outputs.")
    parser.add_argument(
        "--top-ports",
        type=int,
        default=20,
        help="Number of largest ports to include in the bar chart and summary table.",
    )
    return parser.parse_args()


def extract_timestamp(name: str) -> str | None:
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    raw = match.group(1)
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return dt.date().isoformat()


def _parse_entry_name(entry_name: str) -> tuple[str, str, str] | None:
    parts = Path(entry_name).parts
    if len(parts) < 3:
        return None
    if Path(entry_name).suffix.lower() != ".png":
        return None

    port = parts[1]
    date = extract_timestamp(Path(entry_name).name)
    if date is None:
        return None

    return entry_name, port, date


def iter_png_entries(zip_path: Path) -> Iterable[tuple[str, str, str]]:
    try:
        with ZipFile(zip_path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                parsed = _parse_entry_name(entry.filename)
                if parsed is not None:
                    yield parsed
        return
    except BadZipFile:
        pass

    try:
        with tarfile.open(zip_path, mode="r:*") as archive:
            for entry in archive:
                if not entry.isfile():
                    continue
                parsed = _parse_entry_name(entry.name)
                if parsed is not None:
                    yield parsed
            return
    except tarfile.TarError:
        pass

    proc = subprocess.Popen(
        ["tar", "-tf", str(zip_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            parsed = _parse_entry_name(line.strip())
            if parsed is not None:
                yield parsed
    finally:
        proc.stdout.close()
        proc.wait()


def iter_png_files(root_dir: Path) -> Iterable[tuple[str, str, str]]:
    for path in root_dir.rglob("*.png"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        port = parts[0]
        date = extract_timestamp(path.name)
        if date is None:
            continue
        yield str(rel).replace("\\", "/"), port, date


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_chart(title: str, labels: list[str], values: list[int], width: int = 1100, height: int = 420) -> str:
    if not labels or not values:
        return f"<section><h3>{html.escape(title)}</h3><p>No data available.</p></section>"

    left = 70
    right = 20
    top = 40
    bottom = 130
    plot_w = max(100, width - left - right)
    plot_h = max(100, height - top - bottom)
    max_value = max(values) or 1
    bar_w = plot_w / max(len(values), 1)

    parts = [
        f"<section><h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#fbfbf8' stroke='#d7d4ca' />",
    ]

    for i in range(5):
        y = top + (plot_h * i / 4.0)
        value = int(round(max_value * (1 - i / 4.0)))
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#e4e0d6' />")
        parts.append(
            f"<text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='#5d5b55'>{value}</text>"
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * bar_w + max(bar_w * 0.1, 1)
        bar_height = plot_h * (value / max_value)
        y = top + plot_h - bar_height
        parts.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(bar_w * 0.8, 1):.2f}' height='{bar_height:.2f}' "
            "fill='#3d7ea6' opacity='0.9' />"
        )
        text_x = left + idx * bar_w + bar_w / 2
        parts.append(
            f"<text x='{text_x:.2f}' y='{top + plot_h + 16}' transform='rotate(55 {text_x:.2f} {top + plot_h + 16})' "
            f"text-anchor='start' font-size='11' fill='#2b2b2b'>{html.escape(label)}</text>"
        )

    parts.append(f"<text x='{left + plot_w / 2:.1f}' y='{height - 12}' text-anchor='middle' font-size='12'>Label</text>")
    parts.append("</svg></section>")
    return "".join(parts)


def svg_heatmap(title: str, ports: list[str], dates: list[str], counts: dict[tuple[str, str], int]) -> str:
    if not ports or not dates:
        return f"<section><h3>{html.escape(title)}</h3><p>No data available.</p></section>"

    cell = 18
    left = 180
    top = 80
    width = left + len(dates) * cell + 40
    height = top + len(ports) * cell + 40
    max_count = max(counts.values()) if counts else 1

    def color(value: int) -> str:
        if value <= 0:
            return "#f1efe8"
        ratio = value / max_count
        if ratio < 0.2:
            return "#cfe1ec"
        if ratio < 0.4:
            return "#9ec3d8"
        if ratio < 0.6:
            return "#6ea4c3"
        if ratio < 0.8:
            return "#3f86ad"
        return "#1f5d85"

    parts = [
        f"<section><h3>{html.escape(title)}</h3>",
        "<p>Mørkere felter betyder flere billeder. Lyse eller tomme felter kan være huller i dækningen.</p>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
    ]

    for idx, date in enumerate(dates):
        x = left + idx * cell + (cell / 2)
        label = date[0:7]
        parts.append(
            f"<text x='{x:.1f}' y='{top - 10}' transform='rotate(-60 {x:.1f} {top - 10})' "
            f"text-anchor='end' font-size='10' fill='#444'>{html.escape(label)}</text>"
        )

    for row_idx, port in enumerate(ports):
        y = top + row_idx * cell
        parts.append(
            f"<text x='{left - 8}' y='{y + 13}' text-anchor='end' font-size='11' fill='#2b2b2b'>{html.escape(port)}</text>"
        )
        for col_idx, date in enumerate(dates):
            x = left + col_idx * cell
            value = counts.get((port, date), 0)
            parts.append(
                f"<rect x='{x}' y='{y}' width='{cell - 1}' height='{cell - 1}' fill='{color(value)}'>"
                f"<title>{html.escape(port)} | {html.escape(date)} | {value} images</title></rect>"
            )

    parts.append("</svg></section>")
    return "".join(parts)


def build_report(
    out_path: Path,
    summary: dict[str, object],
    top_ports_rows: list[dict[str, object]],
    all_port_rows: list[dict[str, object]],
    month_rows: list[dict[str, object]],
    heatmap_ports: list[str],
    heatmap_dates: list[str],
    heatmap_counts: dict[tuple[str, str], int],
) -> None:
    port_labels = [str(row["port_id"]) for row in top_ports_rows]
    port_values = [int(row["image_count"]) for row in top_ports_rows]
    month_labels = [str(row["month"]) for row in month_rows]
    month_values = [int(row["image_count"]) for row in month_rows]

    html_text = f"""<!doctype html>
<html lang="da">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZIP Inventory Report</title>
  <style>
    :root {{
      --bg: #f6f3ec;
      --card: #fffdf8;
      --border: #d9d4c7;
      --text: #1f1f1f;
      --muted: #666055;
      --accent: #3d7ea6;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; background: radial-gradient(circle at top, #faf8f1 0%, var(--bg) 55%); color: var(--text); }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 2.1rem; }}
    h2 {{ margin: 0 0 20px; color: var(--muted); font-size: 1rem; font-weight: 600; }}
    section {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; margin-bottom: 16px; box-shadow: 0 8px 28px rgba(40, 33, 20, 0.05); overflow-x: auto; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid var(--border); border-radius: 10px; padding: 12px; background: #fffaf1; }}
    .metric .label {{ display: block; color: var(--muted); font-size: 0.9rem; }}
    .metric .value {{ display: block; font-size: 1.45rem; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; font-size: 0.95rem; }}
    th {{ background: #f2eee3; }}
    p {{ line-height: 1.5; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <main>
    <h1>ZIP Coverage Report</h1>
    <h2>{html.escape(str(summary["input_path"]))}</h2>

    <section>
      <div class="summary">
        <div class="metric"><span class="label">Kilde</span><span class="value">{html.escape(str(summary["source_type"]))}</span></div>
        <div class="metric"><span class="label">Billeder</span><span class="value">{summary["total_images"]}</span></div>
        <div class="metric"><span class="label">Havne</span><span class="value">{summary["total_ports"]}</span></div>
        <div class="metric"><span class="label">Første dato</span><span class="value">{summary["min_date"]}</span></div>
        <div class="metric"><span class="label">Sidste dato</span><span class="value">{summary["max_date"]}</span></div>
        <div class="metric"><span class="label">Måneder med data</span><span class="value">{summary["total_months"]}</span></div>
      </div>
    </section>

    {svg_bar_chart("Største havne efter antal billeder", port_labels, port_values)}
    {svg_bar_chart("Billeder pr. måned", month_labels, month_values, width=1500, height=420)}
    {svg_heatmap("Månedlig dækning pr. havn", heatmap_ports, heatmap_dates, heatmap_counts)}

    <section>
      <h3>Tophavne</h3>
      <table>
        <thead><tr><th>port_id</th><th>image_count</th><th>first_date</th><th>last_date</th><th>months_with_data</th><th>missing_months_inside_span</th></tr></thead>
        <tbody>
          {"".join(
              "<tr>"
              f"<td>{html.escape(str(row['port_id']))}</td>"
              f"<td>{row['image_count']}</td>"
              f"<td>{html.escape(str(row['first_date']))}</td>"
              f"<td>{html.escape(str(row['last_date']))}</td>"
              f"<td>{row['months_with_data']}</td>"
              f"<td>{row['missing_months_inside_span']}</td>"
              "</tr>"
              for row in top_ports_rows
          )}
        </tbody>
      </table>
      <p>CSV-filerne i outputmappen kan bruges til videre analyse i Excel eller Python.</p>
    </section>

    <section>
      <h3>Alle Havne Alfabetisk</h3>
      <table>
        <thead><tr><th>port_id</th><th>image_count</th><th>first_date</th><th>last_date</th><th>months_with_data</th><th>missing_months_inside_span</th></tr></thead>
        <tbody>
          {"".join(
              "<tr>"
              f"<td>{html.escape(str(row['port_id']))}</td>"
              f"<td>{row['image_count']}</td>"
              f"<td>{html.escape(str(row['first_date']))}</td>"
              f"<td>{html.escape(str(row['last_date']))}</td>"
              f"<td>{row['months_with_data']}</td>"
              f"<td>{row['missing_months_inside_span']}</td>"
              "</tr>"
              for row in all_port_rows
          )}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def month_range(start_month: str, end_month: str) -> list[str]:
    start = datetime.strptime(start_month + "-01", "%Y-%m-%d")
    end = datetime.strptime(end_month + "-01", "%Y-%m-%d")
    months: list[str] = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        year = current.year + (current.month // 12)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return months


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    port_counter: Counter[str] = Counter()
    date_counter: Counter[str] = Counter()
    month_counter: Counter[str] = Counter()
    port_date_counter: Counter[tuple[str, str]] = Counter()
    port_month_counter: Counter[tuple[str, str]] = Counter()
    port_first_date: dict[str, str] = {}
    port_last_date: dict[str, str] = {}

    total_images = 0

    if input_path.is_dir():
        source_iter = iter_png_files(input_path)
        source_type = "directory"
    else:
        source_iter = iter_png_entries(input_path)
        source_type = "archive"

    for _, port, date in source_iter:
        total_images += 1
        month = date[:7]

        port_counter[port] += 1
        date_counter[date] += 1
        month_counter[month] += 1
        port_date_counter[(port, date)] += 1
        port_month_counter[(port, month)] += 1

        if port not in port_first_date or date < port_first_date[port]:
            port_first_date[port] = date
        if port not in port_last_date or date > port_last_date[port]:
            port_last_date[port] = date

    if total_images == 0:
        raise RuntimeError("No PNG images with parseable timestamps were found in the ZIP archive.")

    all_dates = sorted(date_counter)
    all_months = sorted(month_counter)

    port_rows: list[dict[str, object]] = []
    for port, count in port_counter.most_common():
        first_month = port_first_date[port][:7]
        last_month = port_last_date[port][:7]
        full_span = set(month_range(first_month, last_month))
        observed = {month for (p, month), value in port_month_counter.items() if p == port and value > 0}
        port_rows.append(
            {
                "port_id": port,
                "image_count": count,
                "first_date": port_first_date[port],
                "last_date": port_last_date[port],
                "months_with_data": len(observed),
                "missing_months_inside_span": len(full_span - observed),
            }
        )

    date_rows = [{"date": date, "image_count": date_counter[date]} for date in all_dates]
    month_rows = [{"month": month, "image_count": month_counter[month]} for month in all_months]
    port_date_rows = [
        {"port_id": port, "date": date, "image_count": count}
        for (port, date), count in sorted(port_date_counter.items())
    ]
    port_month_rows = [
        {"port_id": port, "month": month, "image_count": count}
        for (port, month), count in sorted(port_month_counter.items())
    ]

    summary = {
        "input_path": str(input_path),
        "source_type": source_type,
        "total_images": total_images,
        "total_ports": len(port_counter),
        "min_date": all_dates[0],
        "max_date": all_dates[-1],
        "total_months": len(all_months),
    }

    top_ports_rows = port_rows[: args.top_ports]
    all_port_rows = sorted(port_rows, key=lambda row: str(row["port_id"]))
    heatmap_ports = [str(row["port_id"]) for row in all_port_rows]
    heatmap_dates = all_months
    heatmap_counts = {(port, month): count for (port, month), count in port_month_counter.items() if port in heatmap_ports}

    write_csv(out_dir / "counts_by_port.csv", port_rows, ["port_id", "image_count", "first_date", "last_date", "months_with_data", "missing_months_inside_span"])
    write_csv(out_dir / "counts_by_date.csv", date_rows, ["date", "image_count"])
    write_csv(out_dir / "counts_by_month.csv", month_rows, ["month", "image_count"])
    write_csv(out_dir / "counts_by_port_date.csv", port_date_rows, ["port_id", "date", "image_count"])
    write_csv(out_dir / "counts_by_port_month.csv", port_month_rows, ["port_id", "month", "image_count"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_report(
        out_dir / "report.html",
        summary=summary,
        top_ports_rows=top_ports_rows,
        all_port_rows=all_port_rows,
        month_rows=month_rows,
        heatmap_ports=heatmap_ports,
        heatmap_dates=heatmap_dates,
        heatmap_counts=heatmap_counts,
    )

    print(json.dumps({"out_dir": str(out_dir), "total_images": total_images, "total_ports": len(port_counter)}, indent=2))


if __name__ == "__main__":
    main()
