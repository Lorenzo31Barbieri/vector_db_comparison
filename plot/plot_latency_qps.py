import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -------------------------
# Configurazione
# -------------------------
output_dir = "results/plots"
use_equal_spacing = True
use_log_x = False          # (Viene ignorata se use_equal_spacing è True)
use_log_y = False          

# Crea la cartella se non esiste
os.makedirs(output_dir, exist_ok=True)

# Caricamento CSV
df = pd.read_csv("results/final/qps_dataset_size.csv")

# Ordiniamo il dataset per dimensione del dataset per evitare linee intrecciate
df = df.sort_values("dataset_size")

# Estraiamo i valori unici e ordinati delle dimensioni del dataset per usarli come etichette
unique_sizes = sorted(df["dataset_size"].unique())
# Creiamo una mappatura per associare ogni dimensione alla sua posizione (0, 1, 2...)
size_to_pos = {size: i for i, size in enumerate(unique_sizes)}

# Funzione di utilità per applicare le scale, spaziature e formattazione
def apply_plot_settings(xlabel, ylabel, title):
    if use_equal_spacing:
        # Forziamo i "ticks" sui numeri interi (0, 1, 2...) e li rimpiazziamo con i valori reali formattati
        plt.xticks(
            range(len(unique_sizes)), 
            [f"{x:,.0f}" for x in unique_sizes]
        )
    else:
        if use_log_x:
            plt.xscale("log")
        else:
            plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
    if use_log_y:
        plt.yscale("log")
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

# -------------------------
# 1. Latency
# -------------------------
plt.figure(figsize=(8, 5))

for backend, tmp in df.groupby("backend"):
    # Se equidistante, usiamo la posizione (0, 1, 2...) invece del valore numerico reale
    x_val = tmp["dataset_size"].map(size_to_pos) if use_equal_spacing else tmp["dataset_size"]
    plt.plot(x_val, tmp["latency_p95_ms_mean"], marker="o", label=backend)

apply_plot_settings("Dataset Size", "P95 Latency (ms)", "Latency vs Dataset Size")
plt.savefig(os.path.join(output_dir, "latency_vs_size.png"), dpi=150)
plt.close()

# -------------------------
# 2. QPS
# -------------------------
plt.figure(figsize=(8, 5))

for backend, tmp in df.groupby("backend"):
    x_val = tmp["dataset_size"].map(size_to_pos) if use_equal_spacing else tmp["dataset_size"]
    plt.plot(x_val, tmp["qps_mean"], marker="o", label=backend)

apply_plot_settings("Dataset Size", "Queries Per Second", "QPS vs Dataset Size")
plt.savefig(os.path.join(output_dir, "qps_vs_size.png"), dpi=150)
plt.close()


print(f"Done. Grafici generati in '{output_dir}'.")
print(f"Spaziatura Equale: {use_equal_spacing} | Log X: {use_log_x if not use_equal_spacing else 'N/A'} | Log Y: {use_log_y}")