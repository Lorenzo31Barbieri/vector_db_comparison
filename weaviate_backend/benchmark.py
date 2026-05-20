from __future__ import annotations

import time

from weaviate.classes import query as wq

try:
    from . import config as config
except ImportError:
    import config as config
from common.perf import aggregate_latency_metrics as shared_aggregate_latency_metrics


def search(
    collection,
    query_vector,
    *,
    query_filter=None,
    hnsw_ef: int | None = None,
    limit: int | None = None,
):
    _ = hnsw_ef
    return collection.query.near_vector(
        near_vector=query_vector.tolist(),
        filters=query_filter,
        limit=config.TOP_K if limit is None else int(limit),
    )


def warm_up(collection, query_vectors) -> None:
    n = min(config.WARM_UP_QUERIES, len(query_vectors))
    if n == 0:
        return

    print(f"\nRunning {n} warm-up queries...")

    for query_vector in query_vectors[:n]:
        search(collection, query_vector)

    print("Warm-up complete.")


def build_ground_truth(
    collection,
    query_vectors,
    *,
    query_filter=None,
    limit: int | None = None,
):
    print("\nBuilding high-recall ground truth...")

    ground_truth = []

    for i, query_vector in enumerate(query_vectors):
        response = search(
            collection,
            query_vector,
            query_filter=query_filter,
            hnsw_ef=config.HNSW_EF * 4,
            limit=limit,
        )

        ids = [str(obj.uuid) for obj in response.objects]
        ground_truth.append(ids)

        if i > 0 and i % 25 == 0:
            print(f"  Processed {i}/{len(query_vectors)} queries")

    print("Ground truth complete.")
    return ground_truth


def run_single_query_benchmark(
    collection,
    query_vectors,
    *,
    query_filter=None,
    hnsw_ef: int | None = None,
    limit: int | None = None,
):
    print(f"\nRunning single-query benchmark ({len(query_vectors)} queries)...")

    latencies = []
    results = []

    for i, query_vector in enumerate(query_vectors):
        start = time.perf_counter()
        response = search(
            collection,
            query_vector,
            query_filter=query_filter,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        ids = [str(obj.uuid) for obj in response.objects]
        results.append(ids)

        print(f"  Query {i + 1:>3}/{len(query_vectors)}  {elapsed_ms:6.2f} ms")

    return {
        "latencies": latencies,
        "results": results,
    }


def aggregate_latency_metrics(latencies: list) -> dict:
    return shared_aggregate_latency_metrics(latencies)


def build_bucket_filter(max_bucket_exclusive: int):
    return wq.Filter.by_property("bucket").less_than(max_bucket_exclusive)