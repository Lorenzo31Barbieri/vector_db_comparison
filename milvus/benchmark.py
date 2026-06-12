import time

import numpy as np
from pymilvus import Collection

try:
    from . import config as config
except ImportError:
    import config as config
from common.perf import aggregate_latency_metrics as shared_aggregate_latency_metrics


def _build_search_param(effective_ef: int) -> dict:
    index_type = str(config.INDEX_TYPE).upper()

    if index_type == "HNSW":
        return {
            "metric_type": config.METRIC_TYPE,
            "params": {"ef": effective_ef},
        }

    if index_type == "IVF_FLAT":
        nprobe = max(1, min(int(config.IVF_NPROBE), int(config.IVF_NLIST)))
        return {
            "metric_type": config.METRIC_TYPE,
            "params": {"nprobe": nprobe},
        }

    if index_type == "FLAT":
        return {
            "metric_type": config.METRIC_TYPE,
            "params": {},
        }

    raise ValueError(f"Unsupported Milvus index type '{config.INDEX_TYPE}'")


def search(
    collection: Collection,
    vectors: list,
    *,
    expr: str | None = None,
    ef: int | None = None,
    limit: int | None = None,
) -> list:
    effective_ef = config.HNSW_EF if ef is None else int(ef)
    effective_limit = config.TOP_K if limit is None else int(limit)
    return collection.search(
        data=vectors,
        anns_field="vector",
        param=_build_search_param(effective_ef),
        limit=effective_limit,
        expr=expr,
    )


# ── Warm-up ───────────────────────────────────────────────────────────────────

def warm_up(collection: Collection, query_vectors: list) -> None:
    """Send a few queries to warm caches before recording latencies."""
    n = min(config.WARM_UP_QUERIES, len(query_vectors))
    if n == 0:
        return
    print(f"\nWarm-up: running {n} queries (results discarded)...")
    for i in range(n):
        search(collection, [query_vectors[i]])
    print("Warm-up done.")


# ── Single-query benchmark ────────────────────────────────────────────────────

def run_single_query_benchmark(
    collection: Collection,
    query_vectors: list,
    *,
    expr: str | None = None,
    ef: int | None = None,
    limit: int | None = None,
) -> dict:
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
        res = search(collection, [qv], expr=expr, ef=ef, limit=limit)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        results.append(res[0])
        print(f"  Query {i + 1:>3}/{len(query_vectors)}  {latency_ms:6.2f} ms")

    return {"latencies": latencies, "results": results}


# ── Batch-query benchmark ─────────────────────────────────────────────────────

def run_batch_query_benchmark(
    collection: Collection,
    query_vectors: list,
    *,
    expr: str | None = None,
    ef: int | None = None,
    limit: int | None = None,
) -> dict:
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
    results = search(collection, query_vectors, expr=expr, ef=ef, limit=limit)
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
    Near-exhaustive ground truth: search with a large ef to maximise recall.
    """
    print("\nBuilding brute-force ground truth for recall computation...")
    results = search(collection, query_vectors, ef=config.HNSW_EF * 4)
    return list(results)


def build_ground_truth_filtered(
    collection: Collection,
    query_vectors: list,
    expr: str,
    *,
    limit: int,
) -> list:
    return list(search(collection, query_vectors, expr=expr, ef=config.HNSW_EF * 4, limit=limit))


def extract_hit_ids(results: list, *, k: int) -> list[list[int]]:
    return [[hit.id for hit in hits[:k]] for hits in results]


# ── Metric aggregation ────────────────────────────────────────────────────────

def aggregate_latency_metrics(latencies: list) -> dict:
    """
    Compute a full latency profile from a list of per-query latencies (ms).

    Returns
    -------
    dict with: avg, median, p95, p99, min, max, stddev, qps
    """
    return shared_aggregate_latency_metrics(latencies)