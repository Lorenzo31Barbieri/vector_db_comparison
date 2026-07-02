from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
INPUT_PATH = "results/final/selectivity.csv"
OUTPUT_DIR = "results/plots"

def resolve_csv_files(input_raw: str) -> list[Path]:
    path = Path(input_raw)
    return [path] if path.is_file() else sorted(path.rglob("*.csv"))

def safe_float(value: str | None) -> float | None:
    try:
        return float(value.replace(",", "")) if value else None
    except (ValueError, AttributeError):
        return None

def load_selectivity_data(csv_files: list[Path]) -> dict[str, list[tuple[float, float, float]]]:
    """
    Ritorna: { backend: [(selectivity, latency, qps)] }
    """
    data_points = defaultdict(list)
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                backend = row.get("backend", "").strip().lower()
                sel = safe_float(row.get("selectivity"))
                lat = safe_float(row.get("latency_p95_ms_mean"))
                qps = safe_float(row.get("qps_mean"))

                if all(v is not None for v in [sel, lat, qps]):
                    data_points[backend].append((sel, lat, qps))
    return data_points

def aggregate_and_sort(samples: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    # Aggrega per valore di selectivity
    grouped = defaultdict(lambda: {"lat": [], "qps": []})
    for sel, lat, qps in samples:
        grouped[sel]["lat"].append(lat)
        grouped[sel]["qps"].append(qps)

    return sorted([
        (sel, sum(d["lat"])/len(d["lat"]), sum(d["qps"])/len(d["qps"]))
        for sel, d in grouped.items()
    ])

def generate_plot(data, metric_type, output_path_dir):
    plt.figure(figsize=(8, 5))
    
    for backend, raw_data in data.items():
        series = aggregate_and_sort(raw_data)
        xs = [s[0] for s in series]
        ys = [s[1] if metric_type == "latency" else s[2] for s in series]
        
        plt.plot(xs, ys, marker="o", label=backend)

    plt.title(f"{metric_type.capitalize()} vs Selectivity")
    plt.xlabel("Selectivity")
    plt.ylabel("P95 Latency (ms)" if metric_type == "latency" else "QPS")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = output_path_dir / f"selectivity_vs_{metric_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    csv_files = resolve_csv_files(INPUT_PATH)
    data = load_selectivity_data(csv_files)
    
    output_path_dir = Path(OUTPUT_DIR)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    generate_plot(data, "latency", output_path_dir)
    generate_plot(data, "qps", output_path_dir)
    print(f"Grafici generati in '{output_path_dir}'")

if __name__ == "__main__":
    main()