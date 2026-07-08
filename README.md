# Vector DB Comparison

Configuration-driven benchmark suite for **Milvus**, **Qdrant**, and **Weaviate** on the [SIFT-128-euclidean](https://huggingface.co/datasets/open-vdb/sift-128-euclidean) dataset.


## Supported execution modes

The project supports all required modes through one architecture:

1. **Milvus independently**
2. **Qdrant independently**
3. **Weaviate independently**
4. **Cross-backend comparison in one run**

## Architecture

- `benchmark_runner.py`: single orchestrator for lifecycle + scenarios
- `common/backends.py`: backend registry + validation
- `common/benchmark_config.py`: typed suite config loading
- `milvus/adapter.py`, `qdrant/adapter.py`, `weaviate/adapter.py`: backend-specific integration only
- `milvus/main.py`, `qdrant/main.py`, `weaviate/main.py`: thin wrappers that call the shared runner in single-backend mode

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
└── <backend>/
    ├── adapter.py
    ├── benchmark.py
    ├── config.py
    ├── db.py
    └── main.py

```

## Prerequisites

- Python 3.9+
- Docker + Docker Compose

## Setup

```bash
pip install -r requirements.txt
```

Start Docker container:

```bash
cd docker
docker compose up -f docker-compose-<backend>.yml -d
```

## Running benchmarks

### A) Compare all backends in one run

```bash
python benchmark_runner.py --config benchmark_configs/<config>.json
```

### B) Run only one backend

Option 1 (single entrypoint with backend filter):

```bash
python benchmark_runner.py --config benchmark_configs/<config>.json --backend <milvus|qdrant|weaviate>
```

Option 2 (backend wrapper):

```bash
python <milvus|qdrant|weaviate_backend>/main.py --config benchmark_configs/<config>.json
```

## Output

Each run writes a timestamped directory under `results/` with:

- `results.csv`: flattened analysis table
- `run.log`: execution logs

## Scenarios

- `lifecycle`: insert/index/load
- `ann_frontier`: recall/latency/QPS
- `concurrency`: throughput and tail latency under concurrent search
- `filtering`: filtered ANN quality and latency by selectivity
- `index_comparison`: compare important vector index types for each backend independently