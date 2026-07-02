"""
plot_concurrency.py
───────────────────
Generates specific concurrency benchmark plots for the vector-database paper.

Outputs (saved as PNG at 300 dpi):
    fig_p95_heatmap                  – Figure 1: P95 heatmap across all (N, c)
    fig_latency_threads_scale_barh   – Figure 2: Horizontal bar chart showing latency for different threads and dataset sizes
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────
# Allineato per uno stile pulito e coerente con le pubblicazioni (IEEE/ACM)
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.6,
    "lines.markersize":  5,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# Colori dedicati e consistenti per ciascun database backend
STYLE = {
    "milvus":   {"color": "#2166ac", "label": "Milvus"},
    "qdrant":   {"color": "#1a9641", "label": "Qdrant"},
    "weaviate": {"color": "#d73027", "label": "Weaviate"},
}
BACKENDS = ["milvus", "qdrant", "weaviate"]

DATASET_SIZES = [10_000, 50_000, 100_000, 200_000, 500_000]
SIZE_LABELS   = ["10k", "50k", "100k", "200k", "500k"]


# ─────────────────────────────────────────────────────────────────────────────
def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["concurrency_level"] = df["concurrency_level"].astype(int)
    df["dataset_size"] = df["dataset_size"].astype(int)
    return df


def savefig_png(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(path)
    print(f"  saved -> {path}")  # Sostituita freccia unicode per prevenire charmap crash
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — P95 latency heatmap: backend × (N, c)
# ─────────────────────────────────────────────────────────────────────────────
def fig_p95_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Three-panel heatmap (one per backend).
    x-axis: concurrency level; y-axis: dataset size.
    Cell colour: P95 latency (ms).
    """
    vmin = df["latency_p95_ms_mean"].min()
    vmax = df["latency_p95_ms_mean"].max()

    norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1), vmax=vmax)
    cmap = "YlOrRd"

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)

    for ax, backend in zip(axes, BACKENDS):
        bdf = df[df["backend"] == backend].copy()
        pivot = bdf.pivot_table(
            index="dataset_size",
            columns="concurrency_level",
            values="latency_p95_ms_mean",
        )
        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(
            [SIZE_LABELS[DATASET_SIZES.index(n)] for n in pivot.index],
            fontsize=8,
        )

        for ri, row_vals in enumerate(pivot.values):
            for ci, val in enumerate(row_vals):
                text_color = "white" if val > (vmax * 0.5) else "black"
                ax.text(ci, ri, f"{val:.0f}",
                        ha="center", va="center",
                        fontsize=6.5, color=text_color)

        ax.set_title(STYLE[backend]["label"], fontsize=10, pad=18)
        ax.set_xlabel("Concurrency $c$", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Dataset size $N$", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("P95 latency (ms)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        r"P95 latency heatmap across all $(N,\,c)$ conditions",
        fontsize=11, y=1.1
    )

    savefig_png(fig, out_dir, "fig_p95_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate filtered concurrency benchmark figures (Heatmap & Horizontal Bar Chart)."
    )
    parser.add_argument(
        "--csv",
        default="results/final/concurrency_summary.csv",
        help="Path to the concurrency_summary.csv file (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="results/plots",
        help="Output directory for plot (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    df = load(args.csv)
    print(f"  {len(df)} rows | backends: {sorted(df['backend'].unique())}")

    print("\nGenerating Figure 1 — P95 heatmap (PNG only) ...")
    fig_p95_heatmap(df, args.out)

    print("\nDone. Figures saved to:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()