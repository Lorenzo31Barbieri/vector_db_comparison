import time
import numpy as np

import config as config


# ─────────────────────────────────────────────────────────────
# Warm-up
# ─────────────────────────────────────────────────────────────

def warm_up(
    client,
    query_vectors,
):

    print(
        f"\nRunning {config.WARM_UP_QUERIES} warm-up queries..."
    )

    for q in query_vectors[:config.WARM_UP_QUERIES]:

        client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=q.tolist(),
            limit=config.TOP_K,
            search_params={
                "hnsw_ef": config.HNSW_EF,
                "exact": False,
            },
        )

    print("Warm-up complete.")


# ─────────────────────────────────────────────────────────────
# Ground truth (exact search)
# ─────────────────────────────────────────────────────────────

def build_ground_truth(
    client,
    query_vectors,
):

    print("\nBuilding brute-force ground truth...")

    ground_truth = []

    for i, q in enumerate(query_vectors):

        response = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=q.tolist(),
            limit=config.TOP_K,
            search_params={
                "exact": True,
            },
        )

        hits = response.points

        ids = [hit.id for hit in hits]

        ground_truth.append(ids)

        if i > 0 and i % 25 == 0:
            print(f"  Processed {i}/{len(query_vectors)} queries")

    print("Ground truth complete.")

    return ground_truth


# ─────────────────────────────────────────────────────────────
# ANN search benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_search(
    client,
    query_vectors,
):

    print("\nRunning ANN search benchmark...")

    latencies = []

    for q in query_vectors:

        start = time.perf_counter()

        client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=q.tolist(),
            limit=config.TOP_K,
            search_params={
                "hnsw_ef": config.HNSW_EF,
                "exact": False,
            },
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

    avg_latency = float(np.mean(latencies))

    metrics = {
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": float(
            np.percentile(latencies, 95)
        ),
        "qps": 1000 / avg_latency,
    }

    return metrics


# ─────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────

def compute_recall(
    client,
    query_vectors,
    ground_truth,
    k=config.TOP_K,
):

    print(f"\nComputing Recall@{k}...")

    recalls = []

    for i, q in enumerate(query_vectors):

        response = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=q.tolist(),
            limit=k,
            search_params={
                "hnsw_ef": config.HNSW_EF,
                "exact": False,
            },
        )

        ann_hits = response.points

        ann_ids = {
            hit.id for hit in ann_hits
        }

        gt_ids = set(ground_truth[i][:k])

        recall = (
            len(ann_ids.intersection(gt_ids))
            / k
        )

        recalls.append(recall)

    mean_recall = float(np.mean(recalls))

    print(f"Recall@{k}: {mean_recall:.4f}")

    return mean_recall