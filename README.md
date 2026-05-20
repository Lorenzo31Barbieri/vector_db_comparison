# Vector DB Comparison

Configuration-driven benchmark suite for **Milvus** and **Qdrant** on the [SIFT-128-euclidean](https://huggingface.co/datasets/open-vdb/sift-128-euclidean) dataset.

## Supported execution modes

The project supports all required modes through one architecture:

1. **Milvus independently**
2. **Qdrant independently**
3. **Milvus vs Qdrant comparison in one run**

## Architecture (refactored)

- `benchmark_runner.py`: single orchestrator for lifecycle + scenarios
- `common/backends.py`: backend registry + validation
- `common/benchmark_config.py`: typed suite config loading
- `milvus/adapter.py`, `qdrant/adapter.py`: backend-specific integration only
- `milvus/main.py`, `qdrant/main.py`: thin wrappers that call the shared runner in single-backend mode

This keeps benchmark flow centralized while preserving backend-specific implementation details.

## Project structure

```text
.
├── benchmark_runner.py
├── benchmark_configs/
│   ├── suite.default.json
│   ├── smoke.quick.json
│   └── academic.matrix.json
├── common/
│   ├── backends.py
│   ├── benchmark_config.py
│   ├── benchmark_results.py
│   ├── benchmark_workloads.py
│   ├── dataset.py
│   ├── perf.py
│   └── runtime_config.py
├── milvus/
│   ├── adapter.py
│   ├── benchmark.py
│   ├── config.py
│   ├── db.py
│   └── main.py
└── qdrant/
    ├── adapter.py
    ├── benchmark.py
    ├── config.py
    ├── db.py
    └── main.py
```

## Prerequisites

- Python 3.9+
- Docker + Docker Compose (Milvus)
- Running Qdrant server

## Setup

```bash
pip install -r requirements.txt
```

Start Milvus:

```bash
docker compose up -d
```

Start Qdrant:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

## Running benchmarks

### A) Compare both backends in one run

```bash
python benchmark_runner.py --config benchmark_configs/suite.default.json
```

### B) Run only Milvus

Option 1 (single entrypoint with backend filter):

```bash
python benchmark_runner.py --config benchmark_configs/suite.default.json --backend milvus
```

Option 2 (backend wrapper):

```bash
python milvus/main.py --config benchmark_configs/suite.default.json
```

### C) Run only Qdrant

Option 1:

```bash
python benchmark_runner.py --config benchmark_configs/suite.default.json --backend qdrant
```

Option 2:

```bash
python qdrant/main.py --config benchmark_configs/suite.default.json
```

## Output

Each run writes a timestamped directory under `results/` with:

- `manifest.json`: run metadata, config, environment, active backends
- `results.jsonl`: full record stream
- `results.csv`: flattened analysis table
- `run.log`: execution logs

## Scenarios

- `lifecycle`: insert/index/load + memory/storage
- `ann_frontier`: recall/precision/latency/QPS sweep over HNSW ef
- `concurrency`: throughput and tail latency under concurrent search
- `filtering`: filtered ANN quality and latency by selectivity
- `hybrid`: explicit placeholder (`skipped`) until common API is implemented

## Config notes

Benchmark configuration is fully JSON-driven via `benchmark_configs/*.json`.

Key fields:

- `backends`: `["milvus", "qdrant"]` (or one of them)
- `dataset.sizes`, `dataset.query_count`, `dataset.top_k`
- `ann.hnsw_ef_values`
- `concurrency.concurrency_levels`
- `filtering.selectivities`
- `experiment.repeats`, `experiment.seed`
