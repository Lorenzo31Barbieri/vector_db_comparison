import time
import statistics

import numpy as np
from pymilvus import Collection

import config as config


# ── Search helpers ────────────────────────────────────────────────────────────

def _search(collection: Collection, vectors: list) -> list:
    """Run a single search call (one or many query vectors)."""
    return collection.search(
        data=vectors,
        anns_field="vector",
        param={"metric_type": config.METRIC_TYPE, "params": {"nprobe": config.NPROBE}},
        limit=config.TOP_K,
    )


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warm_up(collection: Collection, query_vectors: list) -> None:
    """Send a few queries to warm caches before recording latencies."""
    n = min(config.WARM_UP_QUERIES, len(query_vectors))
    if n == 0:
        return
    print(f"\nWarm-up: running {n} queries (results discarded)...")
    for i in range(n):
        _search(collection, [query_vectors[i]])
    print("Warm-up done.")


# ── Single-query benchmark ────────────────────────────────────────────────────

def run_single_query_benchmark(collection: Collection, query_vectors: list) -> dict:
    """
    Execute NQ queries one at a time and record per-query latency.

    Returns
    -------
    dict with:
        latencies          list[float]   per-query latency in ms
        results            list          raw Milvus result objects (for recall)
    """
    print(f"\nRunning single-query benchmark ({len(query_vectors)} queries)...")
    latencies = []
    results = []

    for i, qv in enumerate(query_vectors):
        t0 = time.perf_counter()
        res = _search(collection, [qv])
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        results.append(res[0])  # one result set per query
        print(f"  Query {i + 1:>3}/{len(query_vectors)}  {latency_ms:6.2f} ms")

    return {"latencies": latencies, "results": results}


# ── Batch-query benchmark ─────────────────────────────────────────────────────

def run_batch_query_benchmark(collection: Collection, query_vectors: list) -> dict:
    """
    Send all NQ queries in a single batch search call.

    Returns
    -------
    dict with:
        batch_time_ms      float   total wall-clock time in ms
        batch_qps          float   queries-per-second for the batch
        results            list    raw Milvus result objects
    """
    print(f"\nRunning batch-query benchmark ({len(query_vectors)} queries in one call)...")
    t0 = time.perf_counter()
    results = _search(collection, query_vectors)
    batch_time_ms = (time.perf_counter() - t0) * 1000
    batch_qps = len(query_vectors) / (batch_time_ms / 1000)
    print(f"  Batch completed in {batch_time_ms:.2f} ms  ({batch_qps:.0f} QPS)")
    return {"batch_time_ms": batch_time_ms, "batch_qps": batch_qps, "results": results}


# ── Recall@K ──────────────────────────────────────────────────────────────────

def compute_recall(
    ann_results: list,
    brute_force_results: list,
    k: int,
) -> float:
    """
    Recall@K = fraction of true nearest neighbours returned by ANN.

    Parameters
    ----------
    ann_results          list of Milvus Hits objects (one per query)
    brute_force_results  same structure, from an exhaustive search (nprobe=nlist)
    k                    the K in Recall@K
    """
    recalls = []
    for ann_hits, bf_hits in zip(ann_results, brute_force_results):
        ann_ids = {hit.id for hit in ann_hits[:k]}
        bf_ids  = {hit.id for hit in bf_hits[:k]}
        recalls.append(len(ann_ids & bf_ids) / k)
    return float(np.mean(recalls))


def build_ground_truth(collection: Collection, query_vectors: list) -> list:
    """
    Brute-force ground truth: search with nprobe == nlist (exhaustive IVF).
    """
    print("\nBuilding brute-force ground truth for recall computation...")
    results = collection.search(
        data=query_vectors,
        anns_field="vector",
        param={
            "metric_type": config.METRIC_TYPE,
            "params": {"nprobe": config.NLIST},  # exhaustive
        },
        limit=config.TOP_K,
    )
    return list(results)


# ── Metric aggregation ────────────────────────────────────────────────────────

def aggregate_latency_metrics(latencies: list) -> dict:
    """
    Compute a full latency profile from a list of per-query latencies (ms).

    Returns
    -------
    dict with: avg, median, p95, p99, min, max, stddev, qps
    """
    arr = np.array(latencies)
    avg = float(np.mean(arr))
    return {
        "avg_ms":    avg,
        "min_ms":    float(np.min(arr)),
        "max_ms":    float(np.max(arr)),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "stddev_ms": float(np.std(arr)),
        "qps":       1000.0 / avg,
    }