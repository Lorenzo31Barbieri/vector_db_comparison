import time
import numpy as np
from qdrant_client import models

try:
    from . import config as config
except ImportError:
    import config as config
from common.perf import aggregate_latency_metrics as shared_aggregate_latency_metrics


def _search(client, query_vector, *, exact: bool):
    return client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=config.TOP_K,
        search_params={
            "hnsw_ef": config.HNSW_EF,
            "exact": exact,
        },
    )


def search(
    client,
    query_vector,
    *,
    exact: bool,
    query_filter: models.Filter | None = None,
    hnsw_ef: int | None = None,
    limit: int | None = None,
):
    return client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=config.TOP_K if limit is None else int(limit),
        query_filter=query_filter,
        search_params={
            "hnsw_ef": config.HNSW_EF if hnsw_ef is None else int(hnsw_ef),
            "exact": exact,
        },
    )


# ─────────────────────────────────────────────────────────────
# Warm-up
# ─────────────────────────────────────────────────────────────

def warm_up(
    client,
    query_vectors,
):
    n = min(config.WARM_UP_QUERIES, len(query_vectors))
    if n == 0:
        return

    print(f"\nRunning {n} warm-up queries...")

    for query_vector in query_vectors[:n]:
        search(client, query_vector, exact=False)

    print("Warm-up complete.")


# ─────────────────────────────────────────────────────────────
# Ground truth (exact search)
# ─────────────────────────────────────────────────────────────

def build_ground_truth(
    client,
    query_vectors,
    *,
    query_filter: models.Filter | None = None,
    limit: int | None = None,
):

    print("\nBuilding brute-force ground truth...")

    ground_truth = []

    for i, query_vector in enumerate(query_vectors):

        response = search(
            client,
            query_vector,
            exact=True,
            query_filter=query_filter,
            limit=limit,
        )

        hits = response.points

        ids = [hit.id for hit in hits]

        ground_truth.append(ids)

        if i > 0 and i % 25 == 0:
            print(f"  Processed {i}/{len(query_vectors)} queries")

    print("Ground truth complete.")

    return ground_truth


# ─────────────────────────────────────────────────────────────
# Single-query benchmark
# ─────────────────────────────────────────────────────────────

def run_single_query_benchmark(
    client,
    query_vectors,
    *,
    query_filter: models.Filter | None = None,
    hnsw_ef: int | None = None,
    limit: int | None = None,
):

    print(f"\nRunning single-query benchmark ({len(query_vectors)} queries)...")

    latencies = []
    results = []

    for i, query_vector in enumerate(query_vectors):

        start = time.perf_counter()
        response = search(
            client,
            query_vector,
            exact=False,
            query_filter=query_filter,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)
        results.append([hit.id for hit in response.points])
        print(f"  Query {i + 1:>3}/{len(query_vectors)}  {elapsed_ms:6.2f} ms")

    return {"latencies": latencies, "results": results}


# ─────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────

def compute_recall(
    ann_results,
    ground_truth,
    k=config.TOP_K,
):

    print(f"\nComputing Recall@{k}...")

    recalls = []

    for i, ann_hits in enumerate(ann_results):

        ann_ids = set(ann_hits[:k])
        gt_ids = set(ground_truth[i][:k])

        recall = (
            len(ann_ids.intersection(gt_ids))
            / k
        )

        recalls.append(recall)

    mean_recall = float(np.mean(recalls))

    print(f"Recall@{k}: {mean_recall:.4f}")

    return mean_recall


def aggregate_latency_metrics(latencies: list) -> dict:
    return shared_aggregate_latency_metrics(latencies)


def build_bucket_filter(max_bucket_exclusive: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="bucket",
                range=models.Range(gte=0, lt=max_bucket_exclusive),
            )
        ]
    )