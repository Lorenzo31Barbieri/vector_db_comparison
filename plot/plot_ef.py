from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAZIONE (SOSTITUISCE CLI ARGS)
# ==============================================================================
# Inserisci qui il percorso del tuo file CSV o della cartella dei risultati
INPUT_PATH = "results/final/ef.csv"

# Inserisci qui la cartella in cui desideri salvare i grafici generati
OUTPUT_DIR = "results/plots"

# Configurazione scala degli assi
USE_LOG_SCALE_X = False
USE_LOG_SCALE_Y = False
# ==============================================================================


def resolve_csv_files(input_raw: str) -> list[Path]:
    path = Path(input_raw)
    csv_files: list[Path] = []
    
    if path.is_file() and path.suffix.lower() == ".csv":
        csv_files.append(path)
    elif path.is_dir():
        csv_files.extend(sorted(path.rglob("*.csv")))

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


def load_ef_data(csv_files: list[Path]) -> dict[str, list[tuple[int, float, float]]]:
    """
    Carica i dati mappandoli per backend.
    Ritorna un dizionario: { backend: [(hnsw_ef, latency, qps)] }
    """
    data_points: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                backend = (row.get("backend") or "").strip().lower()
                
                # Intercettazione e fix automatico del disallineamento della riga Weaviate ef=64
                if backend == "weaviate" and row.get("precision_at_k_std") == "64.0":
                    ef = 64
                    latency = safe_float(row.get("latency_p95_ms_mean"))
                    qps = safe_float(row.get("qps_mean"))
                else:
                    ef = safe_int(row.get("hnsw_ef"))
                    latency = safe_float(row.get("latency_p95_ms_mean"))
                    qps = safe_float(row.get("qps_mean"))

                if not backend or ef is None or latency is None or qps is None:
                    continue

                data_points[backend].append((ef, latency, qps))

    return data_points


def aggregate_and_sort(samples: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Aggrega le repliche calcolando la media aritmetica per ciascun valore di ef."""
    grouped_lat: dict[int, list[float]] = defaultdict(list)
    grouped_qps: dict[int, list[float]] = defaultdict(list)
    
    for ef, lat, qps in samples:
        grouped_lat[ef].append(lat)
        grouped_qps[ef].append(qps)

    aggregated: list[tuple[int, float, float]] = []
    for ef in sorted(grouped_lat):
        mean_lat = sum(grouped_lat[ef]) / len(grouped_lat[ef])
        mean_qps = sum(grouped_qps[ef]) / len(grouped_qps[ef])
        aggregated.append((ef, mean_lat, mean_qps))
        
    return aggregated


def generate_plot(
    *,
    data: dict[str, list[tuple[int, float, float]]],
    metric_type: str,  # "latency" o "qps"
    output_path_dir: Path,
) -> Path:
    # Dimensione della figura aggiornata a (8, 5) come nel secondo script
    plt.figure(figsize=(8, 5))

    # Mappatura dei soli label puliti per mantenere la coerenza dei nomi nella legenda
    backend_labels = {
        "milvus": "milvus",
        "qdrant": "qdrant",
        "weaviate": "weaviate",
    }

    max_y = 0.0

    for backend in sorted(data):
        series = aggregate_and_sort(data[backend])
        if not series:
            continue

        xs = [str(ef) for ef, _, _ in series]
        
        if metric_type == "latency":
            ys = [lat for _, lat, _ in series]
            y_label = "P95 Latency (ms)"
            title = "Latency vs Dataset Size"  # Stile titolo allineato
            file_name = "ef_vs_latency.png"
        else:
            ys = [qps for _, _, qps in series]
            y_label = "Queries Per Second"      # Stile label allineato
            title = "QPS vs Dataset Size"      # Stile titolo allineato
            file_name = "ef_vs_qps.png"

        if ys:
            max_y = max(max_y, max(ys))

        label = backend_labels.get(backend, backend)

        # Plot modificato: rimosse proprietà custom, aggiunto solo marker="o"
        line, = plt.plot(
            xs, ys,
            marker="o",
            label=label,
        )

        # Annotazioni rimosse o ereditate senza formattazioni pesanti per pulizia visiva
        for x, y in zip(xs, ys):
            label_text = f"{y:.1f}"
            plt.annotate(
                label_text,
                xy=(x, y),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=8.5,
                color=line.get_color(),
                ha="center",
            )

    # Titoli e label senza fontweight="bold"
    plt.title(title)
    plt.xlabel("HNSW $ef$ value")
    plt.ylabel(y_label)
    
    if USE_LOG_SCALE_X:
        plt.xscale("log")
    if USE_LOG_SCALE_Y:
        plt.yscale("log")
        
    plt.ylim(0, max_y * 1.15)
    
    # Stile griglia e legenda allineati al secondo script
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    output_path_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path_dir / file_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main() -> None:
    csv_files = resolve_csv_files(INPUT_PATH)

    if not csv_files:
        raise SystemExit(f"No CSV files found at the path: {INPUT_PATH}")

    data_points = load_ef_data(csv_files)
    if not data_points:
        raise SystemExit("No valid benchmark rows found in the provided CSV files.")

    output_path_dir = Path(OUTPUT_DIR)
    generated: list[Path] = []

    # Genera il grafico della Latenza
    generated.append(generate_plot(data=data_points, metric_type="latency", output_path_dir=output_path_dir))
    
    # Genera il grafico del QPS
    generated.append(generate_plot(data=data_points, metric_type="qps", output_path_dir=output_path_dir))

    print("Generated plots:")
    for path in generated:
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()