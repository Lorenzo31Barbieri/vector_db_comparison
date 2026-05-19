# Vector DB Comparison

A benchmarking suite that measures and compares the performance of **Milvus** and **Qdrant** on the [SIFT-128-euclidean](https://huggingface.co/datasets/open-vdb/sift-128-euclidean) dataset.

## What it measures

For each database the benchmark records:

| Metric | Description |
|---|---|
| **Insertion throughput** | Vectors/s during bulk ingestion |
| **Index build time** | Time and throughput to build the ANN index |
| **Single-query latency** | Per-query latency statistics: avg, P50, P95, P99, min, max, std-dev, QPS |
| **Batch-query QPS** | Throughput when all query vectors are sent in one call (Milvus only) |
| **Memory usage** | Estimated in-memory footprint (MB) |
| **Recall@K** | Fraction of true nearest neighbours returned by the ANN search, verified against an exact brute-force search |

## Index types

- **Milvus** — IVF_FLAT (`nlist=1024`, `nprobe=5`, L2 distance)
- **Qdrant** — HNSW (`m=16`, `ef_construct=100`, `hnsw_ef=128`, Euclidean distance)

## Project structure

```
.
├── docker-compose.yml      # Milvus stack (etcd + MinIO + standalone)
├── requirements.txt
├── common/
│   ├── dataset.py          # Shared dataset loading helpers
│   └── perf.py             # Shared throughput and latency metric helpers
├── milvus/
│   ├── config.py           # Milvus connection, collection, index & query parameters
│   ├── db.py               # Connect, create collection, insert, build index, load
│   ├── benchmark.py        # Warm-up, single/batch search, recall, latency aggregation
│   └── main.py             # Orchestrator and report printer
└── qdrant/
    ├── config.py           # Qdrant connection, collection, HNSW & query parameters
    ├── db.py               # Connect, create collection, insert, build index, memory estimate
    ├── benchmark.py        # Warm-up, ground-truth, ANN search, recall
    └── main.py             # Orchestrator and report printer
```

## Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Milvus)
- A running Qdrant instance (see below)

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

```bash
docker compose up -d
```

Milvus will be available at `localhost:19530`. Wait for all three containers (`milvus-etcd`, `milvus-minio`, `milvus-standalone`) to report healthy before running the benchmark.

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Qdrant will be available at `localhost:6333`.

## Running the benchmarks

Each benchmark is a self-contained script that downloads the dataset, creates the collection, inserts vectors, builds the index and prints a full report.

```bash
# Milvus
cd milvus
python main.py

# Qdrant
cd qdrant
python main.py

# Results
# - Console output uses the same standardized format for both backends.
# - Each run appends one row to ../benchmark_results.csv
```

## Phase 2 benchmark suite (reproducible runner)

The project now includes a configuration-driven benchmark runner for repeatable
experiments and analysis-friendly outputs.

### What it adds

- Reproducible experiment manifests (`seed`, run metadata, platform info, git commit)
- Structured result tracking (`results.jsonl`, `results.csv`, `run.log`)
- Scenario-based execution matrix (ANN frontier, concurrency, filtering)
- Statistical repeat runs (`repeats`) for confidence interval analysis
- Additional quality metric: **precision@k** (alongside recall@k)
- Storage footprint capture where available (Qdrant)

### Benchmark config files

- `benchmark_configs/suite.default.json` — default Phase 2 suite
- `benchmark_configs/academic.matrix.json` — larger matrix for essay-grade analysis

### Run the suite

```bash
# from repository root
python benchmark_runner.py --config benchmark_configs/suite.default.json
```

Results are written to a timestamped folder under `results/`, including:

- `manifest.json` — full experiment/environment configuration
- `results.jsonl` — one JSON record per measured datapoint
- `results.csv` — flattened table for pandas/R analysis
- `run.log` — detailed execution log

### Initialize a fresh default config

```bash
python benchmark_runner.py --init-config --config benchmark_configs/suite.default.json
```

### Scenario coverage

- `lifecycle`: insertion/index/load + memory/storage metrics
- `ann_frontier`: recall/precision/latency/QPS across search-parameter sweeps
- `concurrency`: throughput and tail-latency under increasing parallel clients
- `filtering`: filtered-search latency and quality across selectivity levels
- `hybrid`: placeholder output entry (`skipped`) until a common native hybrid API
    is implemented for both backends in this repository

## Configuration

All parameters are in the respective `config.py` files. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `DATASET_SIZE` | 100,000 | Number of vectors loaded from the dataset |
| `BATCH_SIZE` | 10,000 | Insertion batch size |
| `NQ` | 100 | Number of query vectors used for the latency benchmark |
| `TOP_K` | 10 | Neighbours to retrieve per query |
| `WARM_UP_QUERIES` | 5 | Queries discarded before latency recording starts |
| **Milvus** `NLIST` / `NPROBE` | 1024 / 5 | IVF_FLAT index and search parameters |
| **Qdrant** `HNSW_M` / `HNSW_EF_CONSTRUCT` / `HNSW_EF` | 16 / 100 / 128 | HNSW build and search parameters |

## Dataset

[open-vdb/sift-128-euclidean](https://huggingface.co/datasets/open-vdb/sift-128-euclidean) — 128-dimensional SIFT descriptors with L2 distance, a standard ANN benchmark dataset. Downloaded automatically via the Hugging Face `datasets` library on first run.

## Dependencies

| Package | Purpose |
|---|---|
| `pymilvus` | Milvus Python client |
| `qdrant_client` | Qdrant Python client |
| `datasets` | Hugging Face dataset loader |
| `numpy` | Latency statistics and recall computation |
