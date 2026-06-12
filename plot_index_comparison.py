from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# Toggle plotting scales directly in code (not via CLI params).
USE_LOG_SCALE_X = False
USE_LOG_SCALE_Y = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot index-comparison results by backend")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Input CSV files or directories. Directories are searched recursively for results.csv",
    )
    parser.add_argument(
        "--metric",
        default="metrics.avg_ms",
        help="CSV metric column for y-axis (default: metrics.avg_ms)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory where plot images are saved",
    )
    return parser.parse_args()


def resolve_csv_files(inputs: list[str]) -> list[Path]:
    if not inputs:
        inputs = ["results"]

    csv_files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".csv":
            csv_files.append(path)
            continue
        if path.is_dir():
            csv_files.extend(sorted(path.rglob("results.csv")))

    # Keep deterministic ordering and remove duplicates.
    unique = sorted({file.resolve() for file in csv_files})
    return [Path(p) for p in unique]


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: str | None) -> int | None:
    text = "" if value is None else str(value).strip().replace(",", "")
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)

    numeric = safe_float(text)
    if numeric is None:
        return None
    return int(numeric)


def load_points(csv_files: list[Path], metric_col: str) -> dict[str, dict[str, list[tuple[int, float]]]]:
    points: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("scenario") != "index_comparison":
                    continue

                backend = (row.get("backend") or "").strip().lower()
                index_type = (row.get("params.index_type") or "").strip()
                size = safe_int(row.get("dataset_size"))
                time_value = safe_float(row.get(metric_col))

                if not backend or not index_type or size is None or time_value is None:
                    continue

                points[backend][index_type].append((size, time_value))

    return points


def aggregate_mean(samples: list[tuple[int, float]]) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for size, value in samples:
        grouped[size].append(value)

    aggregated: list[tuple[int, float]] = []
    for size in sorted(grouped):
        values = grouped[size]
        aggregated.append((size, sum(values) / len(values)))
    return aggregated


def format_time_ms(metric_col: str, value: float) -> str:
    metric_name = metric_col.strip().lower()

    if metric_name.endswith("_ms"):
        ms_value = value
    elif metric_name.endswith("_s"):
        ms_value = value * 1000.0
    else:
        ms_value = value

    return f"{ms_value:.1f} ms"


def plot_backend(
    *,
    backend: str,
    per_index_points: dict[str, list[tuple[int, float]]],
    metric_col: str,
    output_dir: Path,
) -> Path:
    plt.figure(figsize=(8, 5))

    for index_type in sorted(per_index_points):
        series = aggregate_mean(per_index_points[index_type])
        if not series:
            continue
        xs = [size for size, _ in series]
        ys = [value for _, value in series]
        line, = plt.plot(xs, ys, marker="o", linewidth=1.8, label=index_type)

        for x, y in zip(xs, ys):
            plt.annotate(
                format_time_ms(metric_col, y),
                xy=(x, y),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
                color=line.get_color(),
            )

    plt.title(f"{backend.upper()} index comparison")
    plt.xlabel("Dataset size")
    plt.ylabel(metric_col)
    if USE_LOG_SCALE_X:
        plt.xscale("log")
    if USE_LOG_SCALE_Y:
        plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Index type")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_name = metric_col.replace(".", "_")
    out_path = output_dir / f"{backend}_index_comparison_{metric_name}.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path


def main() -> None:
    args = parse_args()
    csv_files = resolve_csv_files(args.inputs)

    if not csv_files:
        raise SystemExit("No CSV files found. Provide --inputs with results CSV paths or directories.")

    points = load_points(csv_files, args.metric)
    if not points:
        raise SystemExit("No index_comparison rows found in the provided CSV files.")

    output_dir = Path(args.output_dir)
    generated: list[Path] = []
    for backend in sorted(points):
        generated.append(
            plot_backend(
                backend=backend,
                per_index_points=points[backend],
                metric_col=args.metric,
                output_dir=output_dir,
            )
        )

    print("Generated plots:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
