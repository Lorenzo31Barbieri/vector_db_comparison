from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = [
    "timestamp_utc",
    "backend",
    "dataset_name",
    "dataset_size",
    "dimension",
    "index_type",
    "distance_metric",
    "top_k",
    "query_count",
    "insert_time_s",
    "insert_throughput_vps",
    "index_time_s",
    "index_throughput_vps",
    "load_time_s",
    "memory_mb",
    "avg_ms",
    "median_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "stddev_ms",
    "qps_serial",
    "batch_time_ms",
    "batch_qps",
    "recall_at_k",
]


def _fmt(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def build_report_row(
    *,
    backend: str,
    dataset_name: str,
    dataset_size: int,
    dimension: int,
    index_type: str,
    distance_metric: str,
    top_k: int,
    query_count: int,
    insert_stats: dict,
    index_stats: dict,
    latency_metrics: dict,
    recall: float,
    memory_mb: float,
    load_time_s: float | None = None,
    batch_stats: dict | None = None,
) -> dict:
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "dimension": dimension,
        "index_type": index_type,
        "distance_metric": distance_metric,
        "top_k": top_k,
        "query_count": query_count,
        "insert_time_s": float(insert_stats["insert_time"]),
        "insert_throughput_vps": float(insert_stats["throughput"]),
        "index_time_s": float(index_stats["index_time"]),
        "index_throughput_vps": float(index_stats["index_throughput"]),
        "load_time_s": None if load_time_s is None else float(load_time_s),
        "memory_mb": float(memory_mb),
        "avg_ms": float(latency_metrics["avg_ms"]),
        "median_ms": float(latency_metrics["median_ms"]),
        "p95_ms": float(latency_metrics["p95_ms"]),
        "p99_ms": float(latency_metrics["p99_ms"]),
        "min_ms": float(latency_metrics["min_ms"]),
        "max_ms": float(latency_metrics["max_ms"]),
        "stddev_ms": float(latency_metrics["stddev_ms"]),
        "qps_serial": float(latency_metrics["qps"]),
        "batch_time_ms": None,
        "batch_qps": None,
        "recall_at_k": float(recall),
    }

    if batch_stats:
        row["batch_time_ms"] = float(batch_stats.get("batch_time_ms"))
        row["batch_qps"] = float(batch_stats.get("batch_qps"))

    return row


def print_standard_report(row: dict) -> None:
    sep = "=" * 60
    thin = "-" * 36

    print(f"\n{sep}")
    print(f"  BENCHMARK RESULTS ({row['backend'].upper()})")
    print(sep)

    print(f"\n  Dataset      : {row['dataset_name']}")
    print(f"  Dataset size : {row['dataset_size']:,}")
    print(f"  Dimension    : {row['dimension']}")
    print(f"  Index type   : {row['index_type']}")
    print(f"  Metric       : {row['distance_metric']}")
    print(f"  TopK         : {row['top_k']}   NQ={row['query_count']}")

    print(f"\n{thin}")
    print("  INSERTION")
    print(thin)
    print(f"  Time          : {_fmt(row['insert_time_s'], 2, ' s')}")
    print(f"  Throughput    : {_fmt(row['insert_throughput_vps'], 0, ' vectors/s')}")

    print(f"\n{thin}")
    print("  INDEXING")
    print(thin)
    print(f"  Build time    : {_fmt(row['index_time_s'], 2, ' s')}")
    print(f"  Throughput    : {_fmt(row['index_throughput_vps'], 0, ' vectors/s')}")

    print(f"\n{thin}")
    print("  MEMORY / LOAD")
    print(thin)
    print(f"  Load time     : {_fmt(row['load_time_s'], 2, ' s')}")
    print(f"  Memory        : {_fmt(row['memory_mb'], 1, ' MB')}")

    print(f"\n{thin}")
    print("  SINGLE-QUERY LATENCY")
    print(thin)
    print(f"  Avg           : {_fmt(row['avg_ms'], 2, ' ms')}")
    print(f"  Median (P50)  : {_fmt(row['median_ms'], 2, ' ms')}")
    print(f"  P95           : {_fmt(row['p95_ms'], 2, ' ms')}")
    print(f"  P99           : {_fmt(row['p99_ms'], 2, ' ms')}")
    print(f"  Min           : {_fmt(row['min_ms'], 2, ' ms')}")
    print(f"  Max           : {_fmt(row['max_ms'], 2, ' ms')}")
    print(f"  Std-dev       : {_fmt(row['stddev_ms'], 2, ' ms')}")
    print(f"  QPS (serial)  : {_fmt(row['qps_serial'], 1)}")

    print(f"\n{thin}")
    print("  BATCH QUERY")
    print(thin)
    print(f"  Total time    : {_fmt(row['batch_time_ms'], 2, ' ms')}")
    print(f"  QPS (batch)   : {_fmt(row['batch_qps'], 1)}")

    print(f"\n{thin}")
    print("  ACCURACY")
    print(thin)
    print(f"  Recall@{row['top_k']:<3}   : {_fmt(row['recall_at_k'] * 100, 2, '%')}")

    print(f"\n{sep}")
    print("  Benchmark complete.")
    print(sep)


def append_report_csv(row: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    needs_header = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CSV_COLUMNS})

    return output
