from __future__ import annotations

import argparse
import logging
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from common.backends import SUPPORTED_BACKENDS, adapter_for_backend, resolve_backends
from common.benchmark_config import load_suite_config, write_default_suite_config
from common.benchmark_results import ExperimentTracker, flatten_for_csv
from common.benchmark_workloads import compute_recall_and_precision_at_k, run_concurrency_benchmark
from common.dataset import load_sift_vectors


def _git_commit(root_dir: Path) -> str | None:
    head = root_dir / ".git" / "HEAD"
    if not head.exists():
        return None

    head_text = head.read_text(encoding="utf-8").strip()
    if not head_text.startswith("ref:"):
        return head_text

    ref_path = root_dir / ".git" / head_text.split(" ", 1)[1]
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()

    return None


def _build_manifest(config, root_dir: Path, active_backends: Sequence[str]) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": {
            "name": config.experiment.name,
            "seed": config.experiment.seed,
            "repeats": config.experiment.repeats,
            "notes": config.experiment.notes,
        },
        "dataset": {
            "name": config.dataset.name,
            "sizes": config.dataset.sizes,
            "query_count": config.dataset.query_count,
            "top_k": config.dataset.top_k,
        },
        "backends": list(active_backends),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": {
            "commit": _git_commit(root_dir),
        },
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _append_result(
    *,
    tracker: ExperimentTracker,
    all_rows: list[dict],
    result: dict,
) -> None:
    tracker.append_result(result)
    all_rows.append(flatten_for_csv(result))


def _run_ann_scenario(*, adapter, config, backend_name: str, dataset_size: int, dataset_name: str, query_vectors: np.ndarray, tracker: ExperimentTracker, rows: list[dict], logger: logging.Logger) -> None:
    if not config.ann.enabled:
        return

    adapter_kwarg = "ef" if backend_name == "milvus" else "hnsw_ef"

    for sweep_value in config.ann.hnsw_ef_values:
        logger.info(
            "Running ANN scenario backend=%s dataset_size=%s hnsw_ef=%s",
            backend_name,
            dataset_size,
            sweep_value,
        )

        ground_truth = adapter.build_ground_truth(
            query_vectors,
            limit=config.dataset.top_k,
        )

        recall_values: list[float] = []
        precision_values: list[float] = []
        qps_values: list[float] = []
        p95_values: list[float] = []

        for repeat_index in range(config.experiment.repeats):
            ann_result = adapter.run_ann(
                query_vectors,
                **{adapter_kwarg: sweep_value},
                limit=config.dataset.top_k,
            )
            quality = compute_recall_and_precision_at_k(
                ann_result["ids"],
                ground_truth,
                config.dataset.top_k,
            )

            recall_values.append(quality["recall_at_k"])
            precision_values.append(quality["precision_at_k"])
            qps_values.append(ann_result["latency"]["qps"])
            p95_values.append(ann_result["latency"]["p95_ms"])

            ann_row = {
                "scenario": "ann_frontier",
                "backend": backend_name,
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "repeat": repeat_index + 1,
                "params": {"hnsw_ef": sweep_value, "top_k": config.dataset.top_k},
                "metrics": {
                    **ann_result["latency"],
                    **quality,
                },
            }
            _append_result(tracker=tracker, all_rows=rows, result=ann_row)

        summary_row = {
            "scenario": "ann_frontier_summary",
            "backend": backend_name,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "params": {"hnsw_ef": sweep_value, "top_k": config.dataset.top_k},
            "metrics": {
                "recall_mean": statistics.mean(recall_values),
                "recall_std": statistics.pstdev(recall_values) if len(recall_values) > 1 else 0.0,
                "precision_mean": statistics.mean(precision_values),
                "qps_mean": statistics.mean(qps_values),
                "p95_ms_mean": statistics.mean(p95_values),
            },
        }
        _append_result(tracker=tracker, all_rows=rows, result=summary_row)


def _run_concurrency_scenario(*, adapter, config, backend_name: str, dataset_size: int, dataset_name: str, query_vectors: np.ndarray, tracker: ExperimentTracker, rows: list[dict], logger: logging.Logger) -> None:
    if not config.concurrency.enabled:
        return

    for concurrency in config.concurrency.concurrency_levels:
        logger.info(
            "Running concurrency scenario backend=%s dataset_size=%s concurrency=%s",
            backend_name,
            dataset_size,
            concurrency,
        )
        for repeat_index in range(config.experiment.repeats):
            concurrency_result = run_concurrency_benchmark(
                query_vectors=query_vectors,
                search_one_fn=lambda query_vector: adapter.search_one(
                    query_vector,
                    limit=config.dataset.top_k,
                ),
                concurrency=concurrency,
            )
            row = {
                "scenario": "concurrency",
                "backend": backend_name,
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "repeat": repeat_index + 1,
                "params": {"concurrency": concurrency, "top_k": config.dataset.top_k},
                "metrics": concurrency_result,
            }
            _append_result(tracker=tracker, all_rows=rows, result=row)


def _run_filtering_scenario(*, adapter, config, backend_name: str, dataset_size: int, dataset_name: str, query_vectors: np.ndarray, tracker: ExperimentTracker, rows: list[dict], logger: logging.Logger) -> None:
    if not config.filtering.enabled:
        return

    for selectivity in config.filtering.selectivities:
        logger.info(
            "Running filtering scenario backend=%s dataset_size=%s selectivity=%s",
            backend_name,
            dataset_size,
            selectivity,
        )

        filtered_gt = adapter.build_filtered_ground_truth(
            query_vectors,
            limit=config.dataset.top_k,
            selectivity=selectivity,
        )

        recall_values: list[float] = []
        precision_values: list[float] = []

        for repeat_index in range(config.experiment.repeats):
            filtered_ann = adapter.run_filtered_ann(
                query_vectors,
                selectivity=selectivity,
                limit=config.dataset.top_k,
            )

            quality = compute_recall_and_precision_at_k(
                filtered_ann["ids"],
                filtered_gt,
                config.dataset.top_k,
            )
            recall_values.append(quality["recall_at_k"])
            precision_values.append(quality["precision_at_k"])

            row = {
                "scenario": "filtering",
                "backend": backend_name,
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "repeat": repeat_index + 1,
                "params": {
                    "selectivity": selectivity,
                    "top_k": config.dataset.top_k,
                },
                "metrics": {
                    **filtered_ann["latency"],
                    **quality,
                },
            }
            _append_result(tracker=tracker, all_rows=rows, result=row)

        summary_row = {
            "scenario": "filtering_summary",
            "backend": backend_name,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "params": {
                "selectivity": selectivity,
                "top_k": config.dataset.top_k,
            },
            "metrics": {
                "recall_mean": statistics.mean(recall_values),
                "precision_mean": statistics.mean(precision_values),
            },
        }
        _append_result(tracker=tracker, all_rows=rows, result=summary_row)


def _run_hybrid_placeholder(*, config, backend_name: str, dataset_name: str, dataset_size: int, tracker: ExperimentTracker, rows: list[dict], logger: logging.Logger) -> None:
    if not config.hybrid.enabled:
        return

    logger.info(
        "Hybrid scenario enabled but backend=%s currently marked not-supported",
        backend_name,
    )
    _append_result(
        tracker=tracker,
        all_rows=rows,
        result={
            "scenario": "hybrid",
            "backend": backend_name,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "status": "skipped",
            "reason": "Current dataset and backend adapters do not expose a common native hybrid API yet.",
            "params": {"mode": config.hybrid.mode},
        },
    )


def _run_index_comparison_scenario(
    *,
    adapter,
    config,
    backend_name: str,
    dataset_size: int,
    dataset_name: str,
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    tracker: ExperimentTracker,
    rows: list[dict],
    logger: logging.Logger,
) -> None:
    if not config.index_comparison.enabled:
        return

    supported = getattr(adapter, "supported_index_types", None)
    set_index_type = getattr(adapter, "set_index_type", None)
    if not callable(supported) or not callable(set_index_type):
        logger.info("Skipping index comparison for backend=%s (adapter does not expose index switching)", backend_name)
        return

    requested = config.index_comparison.index_types_by_backend.get(backend_name, [])
    if not requested:
        logger.info("Skipping index comparison for backend=%s (no index types configured)", backend_name)
        return

    available = set(supported())
    index_types = [index_type for index_type in requested if index_type in available]
    if not index_types:
        logger.info("Skipping index comparison for backend=%s (no configured index types are supported)", backend_name)
        return

    adapter_kwarg = "ef" if backend_name == "milvus" else "hnsw_ef"
    sweep_ef = config.index_comparison.ann_hnsw_ef
    teardown = getattr(adapter, "teardown", None)

    # Close the initial runtime before rebuilding for each index variant.
    if callable(teardown):
        teardown()

    for index_index, index_type in enumerate(index_types):
        if index_index > 0 and callable(teardown):
            teardown()

        logger.info(
            "Running index comparison backend=%s dataset_size=%s index_type=%s",
            backend_name,
            dataset_size,
            index_type,
        )

        set_index_type(index_type)
        lifecycle = adapter.prepare(vectors)
        adapter.warm_up(query_vectors)

        ground_truth = adapter.build_ground_truth(
            query_vectors,
            limit=config.dataset.top_k,
        )
        ann_result = adapter.run_ann(
            query_vectors,
            **{adapter_kwarg: sweep_ef},
            limit=config.dataset.top_k,
        )
        quality = compute_recall_and_precision_at_k(
            ann_result["ids"],
            ground_truth,
            config.dataset.top_k,
        )

        row = {
            "scenario": "index_comparison",
            "backend": backend_name,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "params": {
                "index_type": index_type,
                "hnsw_ef": sweep_ef,
                "top_k": config.dataset.top_k,
            },
            "metrics": {
                "insert_time_s": lifecycle["insert"].get("insert_time"),
                "insert_throughput_vps": lifecycle["insert"].get("throughput"),
                "index_time_s": lifecycle["index"].get("index_time"),
                "index_throughput_vps": lifecycle["index"].get("index_throughput"),
                "load_time_s": lifecycle["load"].get("load_time"),
                **ann_result["latency"],
                **quality,
            },
        }
        _append_result(tracker=tracker, all_rows=rows, result=row)


def _run_hnsw_m_scenario(
    *,
    adapter,
    config,
    backend_name: str,
    dataset_size: int,
    dataset_name: str,
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    tracker: ExperimentTracker,
    rows: list[dict],
    logger: logging.Logger,
) -> None:
    if not config.hnsw_m.enabled:
        return

    set_hnsw_m = getattr(adapter, "set_hnsw_m", None)
    set_index_type = getattr(adapter, "set_index_type", None)
    if not callable(set_hnsw_m) or not callable(set_index_type):
        logger.info("Skipping hnsw_m scenario for backend=%s (adapter does not expose hnsw M/index controls)", backend_name)
        return

    hnsw_index_type = "hnsw" if backend_name == "weaviate" else "HNSW"
    adapter_kwarg = "ef" if backend_name == "milvus" else "hnsw_ef"
    sweep_ef = config.hnsw_m.ann_hnsw_ef
    teardown = getattr(adapter, "teardown", None)

    # Close the initial runtime before rebuilding for each HNSW M variant.
    if callable(teardown):
        teardown()

    for m_index, hnsw_m in enumerate(config.hnsw_m.m_values):
        if m_index > 0 and callable(teardown):
            teardown()

        logger.info(
            "Running hnsw_m scenario backend=%s dataset_size=%s hnsw_m=%s",
            backend_name,
            dataset_size,
            hnsw_m,
        )

        set_index_type(hnsw_index_type)
        set_hnsw_m(int(hnsw_m))
        lifecycle = adapter.prepare(vectors)
        adapter.warm_up(query_vectors)

        ground_truth = adapter.build_ground_truth(
            query_vectors,
            limit=config.dataset.top_k,
        )

        recall_values: list[float] = []
        precision_values: list[float] = []
        qps_values: list[float] = []
        p95_values: list[float] = []

        for repeat_index in range(config.experiment.repeats):
            ann_result = adapter.run_ann(
                query_vectors,
                **{adapter_kwarg: sweep_ef},
                limit=config.dataset.top_k,
            )
            quality = compute_recall_and_precision_at_k(
                ann_result["ids"],
                ground_truth,
                config.dataset.top_k,
            )

            recall_values.append(quality["recall_at_k"])
            precision_values.append(quality["precision_at_k"])
            qps_values.append(ann_result["latency"].get("qps", 0.0))
            p95_values.append(ann_result["latency"].get("p95_ms", 0.0))

            row = {
                "scenario": "hnsw_m",
                "backend": backend_name,
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "repeat": repeat_index + 1,
                "hnsw_m": int(hnsw_m),
                "hnsw_ef": sweep_ef,
                "top_k": config.dataset.top_k,
                "index_time_s": lifecycle["index"].get("index_time"),
                "index_throughput_vps": lifecycle["index"].get("index_throughput"),
                "load_time_s": lifecycle["load"].get("load_time"),
                "qps": ann_result["latency"].get("qps"),
                "mean_ms": ann_result["latency"].get("mean_ms"),
                "p95_ms": ann_result["latency"].get("p95_ms"),
                "p99_ms": ann_result["latency"].get("p99_ms"),
                "recall_at_k": quality["recall_at_k"],
                "precision_at_k": quality["precision_at_k"],
            }
            _append_result(tracker=tracker, all_rows=rows, result=row)

        summary_row = {
            "scenario": "hnsw_m_summary",
            "backend": backend_name,
            "dataset_name": dataset_name,
            "dataset_size": dataset_size,
            "hnsw_m": int(hnsw_m),
            "hnsw_ef": sweep_ef,
            "top_k": config.dataset.top_k,
            "index_time_s": lifecycle["index"].get("index_time"),
            "index_throughput_vps": lifecycle["index"].get("index_throughput"),
            "load_time_s": lifecycle["load"].get("load_time"),
            "qps_mean": statistics.mean(qps_values),
            "p95_ms_mean": statistics.mean(p95_values),
            "recall_mean": statistics.mean(recall_values),
            "recall_std": statistics.pstdev(recall_values) if len(recall_values) > 1 else 0.0,
            "precision_mean": statistics.mean(precision_values),
        }
        _append_result(tracker=tracker, all_rows=rows, result=summary_row)


def run_suite(config_path: str, backends: Sequence[str] | None = None) -> Path:
    root_dir = Path(__file__).resolve().parent
    config = load_suite_config(config_path)
    selected_backends = resolve_backends(backends if backends else config.backends)

    _seed_everything(config.experiment.seed)

    tracker = ExperimentTracker(Path(config.experiment.output_dir), config.experiment.name)
    tracker.write_manifest(_build_manifest(config, root_dir, selected_backends))

    logger = logging.getLogger("benchmark_suite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(tracker.run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    rows: list[dict] = []

    for dataset_size in config.dataset.sizes:
        vectors, _ = load_sift_vectors(
            dataset_name=config.dataset.name,
            dataset_size=dataset_size,
            as_numpy=True,
        )
        vectors = vectors.astype(np.float32)
        query_vectors = vectors[: config.dataset.query_count]

        for backend_name in selected_backends:
            logger.info("Starting backend=%s dataset_size=%s", backend_name, dataset_size)
            adapter = adapter_for_backend(backend_name)

            try:
                adapter.configure(
                    dataset_name=config.dataset.name,
                    dataset_size=dataset_size,
                    top_k=config.dataset.top_k,
                    query_count=config.dataset.query_count,
                    batch_size=10_000,
                    warm_up_queries=min(10, config.dataset.query_count),
                )

                lifecycle = adapter.prepare(vectors)
                adapter.warm_up(query_vectors)
                logger.info("Prepared backend=%s dataset_size=%s", backend_name, dataset_size)

                lifecycle_row = {
                    "scenario": "lifecycle",
                    "backend": backend_name,
                    "dataset_name": config.dataset.name,
                    "dataset_size": dataset_size,
                    "top_k": config.dataset.top_k,
                    "metrics": {
                        "insert_time_s": lifecycle["insert"].get("insert_time"),
                        "insert_throughput_vps": lifecycle["insert"].get("throughput"),
                        "index_time_s": lifecycle["index"].get("index_time"),
                        "index_throughput_vps": lifecycle["index"].get("index_throughput"),
                        "load_time_s": lifecycle["load"].get("load_time"),
                    },
                }
                _append_result(tracker=tracker, all_rows=rows, result=lifecycle_row)

                _run_ann_scenario(
                    adapter=adapter,
                    config=config,
                    backend_name=backend_name,
                    dataset_size=dataset_size,
                    dataset_name=config.dataset.name,
                    query_vectors=query_vectors,
                    tracker=tracker,
                    rows=rows,
                    logger=logger,
                )
                _run_concurrency_scenario(
                    adapter=adapter,
                    config=config,
                    backend_name=backend_name,
                    dataset_size=dataset_size,
                    dataset_name=config.dataset.name,
                    query_vectors=query_vectors,
                    tracker=tracker,
                    rows=rows,
                    logger=logger,
                )
                _run_filtering_scenario(
                    adapter=adapter,
                    config=config,
                    backend_name=backend_name,
                    dataset_size=dataset_size,
                    dataset_name=config.dataset.name,
                    query_vectors=query_vectors,
                    tracker=tracker,
                    rows=rows,
                    logger=logger,
                )
                _run_hybrid_placeholder(
                    config=config,
                    backend_name=backend_name,
                    dataset_name=config.dataset.name,
                    dataset_size=dataset_size,
                    tracker=tracker,
                    rows=rows,
                    logger=logger,
                )
                _run_index_comparison_scenario(
                    adapter=adapter,
                    config=config,
                    backend_name=backend_name,
                    dataset_size=dataset_size,
                    dataset_name=config.dataset.name,
                    vectors=vectors,
                    query_vectors=query_vectors,
                    tracker=tracker,
                    rows=rows,
                    logger=logger,
                )
                _run_hnsw_m_scenario(
                    adapter=adapter,
                    config=config,
                    backend_name=backend_name,
                    dataset_size=dataset_size,
                    dataset_name=config.dataset.name,
                    vectors=vectors,
                    query_vectors=query_vectors,
                    tracker=tracker,
                    rows=rows,
                    logger=logger,
                )
            finally:
                teardown = getattr(adapter, "teardown", None)
                if callable(teardown):
                    teardown()

    tracker.write_csv(rows)
    logger.info("Benchmark suite complete: %s", tracker.run_dir)
    return tracker.run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible vector DB benchmark suite")
    parser.add_argument(
        "--config",
        default="benchmark_configs/suite.default.json",
        help="Path to benchmark suite configuration JSON",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Write a default benchmark config at --config and exit",
    )
    parser.add_argument(
        "--backend",
        dest="backends",
        action="append",
        choices=list(SUPPORTED_BACKENDS),
        help="Backend to run. Repeat to run multiple backends. If omitted, uses config backends.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.init_config:
        path = write_default_suite_config(args.config)
        print(f"Default suite config written to: {path}")
        return

    run_dir = run_suite(args.config, backends=args.backends)
    print(f"\nBenchmark suite complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()
