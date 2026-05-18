import numpy as np
from datasets import load_dataset

import config as config
import db as db
import benchmark as benchmark


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

def load_sift_vectors():

    print(
        f"\nLoading dataset '{config.DATASET_NAME}' "
        f"(first {config.DATASET_SIZE:,} vectors)..."
    )

    dataset = load_dataset(
        config.DATASET_NAME,
        "train",
    )

    vectors = dataset["train"]["emb"][:config.DATASET_SIZE]

    vectors = np.asarray(vectors, dtype=np.float32)

    dimension = vectors.shape[1]

    print(
        f"Loaded {len(vectors):,} vectors "
        f"(dim={dimension})"
    )

    return vectors, dimension


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_report(
    insert_stats: dict,
    index_stats: dict,
    latency_metrics: dict,
    recall: float,
    memory_mb: float,
    dimension: int,
    collection_info,
) -> None:

    sep = "=" * 52
    thin = "-" * 28

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)

    print(f"\n  Dataset size : {config.DATASET_SIZE:,}")
    print(f"  Dimension    : {dimension}")

    print(
        f"  Index type   : HNSW  "
        f"m={config.HNSW_M}  "
        f"ef_construct={config.HNSW_EF_CONSTRUCT}"
    )

    print(f"  Distance     : {config.DISTANCE}")

    print(
        f"  TopK         : "
        f"{config.TOP_K}   "
        f"hnsw_ef={config.HNSW_EF}"
    )

    # ─────────────────────────
    # Collection status
    # ─────────────────────────

    if collection_info:

        vectors_count = getattr(
            collection_info,
            "vectors_count",
            0,
        ) or 0

        indexed_count = getattr(
            collection_info,
            "indexed_vectors_count",
            0,
        ) or 0

        segments_count = getattr(
            collection_info,
            "segments_count",
            0,
        ) or 0

        print(f"\n{thin}")
        print("  COLLECTION STATUS")
        print(thin)

        print(f"  Status        : {collection_info.status}")
        print(f"  Vectors       : {vectors_count:,}")
        print(f"  Indexed       : {indexed_count:,}")
        print(f"  Segments      : {segments_count}")

    # ─────────────────────────
    # Insertion
    # ─────────────────────────

    print(f"\n{thin}")
    print("  INSERTION")
    print(thin)

    print(
        f"  Time          : "
        f"{insert_stats['insert_time']:.2f} s"
    )

    print(
        f"  Throughput    : "
        f"{insert_stats['throughput']:,.0f} vectors/s"
    )

    # ─────────────────────────
    # Index build
    # ─────────────────────────

    print(f"\n{thin}")
    print("  HNSW INDEX BUILD")
    print(thin)

    print(
        f"  Build time    : "
        f"{index_stats['index_time']:.2f} s"
    )

    print(
        f"  Throughput    : "
        f"{index_stats['index_throughput']:,.0f} vectors/s"
    )

    # ─────────────────────────
    # Memory
    # ─────────────────────────

    print(f"\n{thin}")
    print("  MEMORY (estimate)")
    print(thin)

    print(f"  RAM usage     : {memory_mb:.1f} MB")

    # ─────────────────────────
    # Query latency
    # ─────────────────────────

    print(f"\n{thin}")
    print(f"  QUERY LATENCY  (n={config.NQ})")
    print(thin)

    print(
        f"  Avg           : "
        f"{latency_metrics['avg_latency_ms']:.2f} ms"
    )

    print(
        f"  P95           : "
        f"{latency_metrics['p95_latency_ms']:.2f} ms"
    )

    print(
        f"  QPS           : "
        f"{latency_metrics['qps']:.1f}"
    )

    # ─────────────────────────
    # Recall
    # ─────────────────────────

    print(f"\n{thin}")
    print("  ACCURACY")
    print(thin)

    print(
        f"  Recall@{config.TOP_K:<3}   : "
        f"{recall * 100:.2f}%"
    )

    print(f"\n{sep}")
    print("  Benchmark complete.")
    print(sep)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

    # 1. Connect
    client = db.connect()

    # 2. Load vectors
    vectors, dimension = load_sift_vectors()

    query_vectors = vectors[:config.NQ]

    # 3. Recreate collection
    db.recreate_collection(
        client,
        dimension,
    )

    # 4. Insert vectors
    insert_stats = db.insert_vectors(
        client,
        vectors.tolist(),
    )

    # 5. Build HNSW index
    index_stats = db.build_index(
        client,
        len(vectors),
    )

    # 6. Build brute-force ground truth
    ground_truth = benchmark.build_ground_truth(
        client,
        query_vectors,
    )

    # 7. Benchmark ANN search
    latency_metrics = benchmark.benchmark_search(
        client,
        query_vectors,
    )

    # 8. Recall@K
    recall = benchmark.compute_recall(
        client=client,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        k=config.TOP_K,
    )

    # 9. Memory estimate
    memory_mb = db.estimate_memory_mb(
        client,
        dimension,
        len(vectors),
    )

    # 10. Collection info
    collection_info = db.get_collection_info(
        client,
    )

    # 11. Report
    print_report(
        insert_stats=insert_stats,
        index_stats=index_stats,
        latency_metrics=latency_metrics,
        recall=recall,
        memory_mb=memory_mb,
        dimension=dimension,
        collection_info=collection_info,
    )


if __name__ == "__main__":
    main()