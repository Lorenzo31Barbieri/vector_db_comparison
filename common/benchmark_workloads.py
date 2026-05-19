from __future__ import annotations

import concurrent.futures
import statistics
import time
from typing import Callable

import numpy as np

from common.perf import aggregate_latency_metrics


def compute_recall_and_precision_at_k(
    ann_results: list[list[int]],
    ground_truth: list[list[int]],
    k: int,
) -> dict[str, float]:
    recalls: list[float] = []
    precisions: list[float] = []

    for ann_hits, truth_hits in zip(ann_results, ground_truth):
        ann_set = set(ann_hits[:k])
        truth_set = set(truth_hits[:k])
        overlap = len(ann_set.intersection(truth_set))
        recalls.append(overlap / k)
        precisions.append(overlap / k)

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
    }


def run_concurrency_benchmark(
    query_vectors: list,
    search_one_fn: Callable[[object], list[int]],
    concurrency: int,
) -> dict:
    latencies_ms: list[float] = []

    def timed_search(query_vector):
        t0 = time.perf_counter()
        _ = search_one_fn(query_vector)
        return (time.perf_counter() - t0) * 1000

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(timed_search, query_vector) for query_vector in query_vectors]
        for future in concurrent.futures.as_completed(futures):
            latencies_ms.append(float(future.result()))

    elapsed = time.perf_counter() - start
    total_queries = len(query_vectors)

    latency_profile = aggregate_latency_metrics(latencies_ms)
    latency_profile["p999_ms"] = float(np.percentile(np.asarray(latencies_ms), 99.9)) if latencies_ms else 0.0
    latency_profile["qps"] = total_queries / elapsed if elapsed > 0 else float("inf")

    return {
        "concurrency": concurrency,
        "query_count": total_queries,
        "total_time_s": elapsed,
        "qps": latency_profile["qps"],
        "latency": latency_profile,
        "latency_cv": (statistics.pstdev(latencies_ms) / statistics.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def selectivity_to_filter_bucket(selectivity: float, buckets: int = 100) -> int:
    bounded = min(max(selectivity, 0.0), 1.0)
    threshold = max(1, int(round(buckets * bounded)))
    return threshold
