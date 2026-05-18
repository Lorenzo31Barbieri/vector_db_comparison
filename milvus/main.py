from datasets import load_dataset

import config as config
import db as db
import benchmark as benchmark


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_sift_vectors():
    print(f"\nLoading dataset '{config.DATASET_NAME}' (first {config.DATASET_SIZE} vectors)...")
    dataset = load_dataset(config.DATASET_NAME, "train")
    vectors = dataset["train"]["emb"][: config.DATASET_SIZE]
    dimension = len(vectors[0])
    print(f"Loaded {len(vectors)} vectors  (dim={dimension})")
    return vectors, dimension


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(
    insert_stats: dict,
    index_stats: dict,
    load_stats: dict,
    latency_metrics: dict,
    batch_stats: dict,
    recall: float,
    memory_mb: float,
    dimension: int,
) -> None:
    sep = "=" * 52
    thin = "-" * 28

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)

    print(f"\n  Dataset size : {config.DATASET_SIZE:,}")
    print(f"  Dimension    : {dimension}")
    print(f"  Index type   : {config.INDEX_TYPE}  nlist={config.NLIST}")
    print(f"  Metric       : {config.METRIC_TYPE}")
    print(f"  TopK         : {config.TOP_K}   nprobe={config.NPROBE}")

    print(f"\n{thin}")
    print("  INSERTION")
    print(thin)
    print(f"  Time          : {insert_stats['insert_time']:.2f} s")
    print(f"  Throughput    : {insert_stats['throughput']:,.0f} vectors/s")

    print(f"\n{thin}")
    print("  INDEXING")
    print(thin)
    print(f"  Build time    : {index_stats['index_time']:.2f} s")
    print(f"  Throughput    : {index_stats['index_throughput']:,.0f} vectors/s")

    print(f"\n{thin}")
    print("  MEMORY")
    print(thin)
    print(f"  Load time     : {load_stats['load_time']:.2f} s")
    if memory_mb > 0:
        print(f"  Segment mem   : {memory_mb:.1f} MB")
    else:
        print("  Segment mem   : n/a")

    print(f"\n{thin}")
    print(f"  SINGLE-QUERY LATENCY  (n={config.NQ})")
    print(thin)
    print(f"  Avg           : {latency_metrics['avg_ms']:.2f} ms")
    print(f"  Median (P50)  : {latency_metrics['median_ms']:.2f} ms")
    print(f"  P95           : {latency_metrics['p95_ms']:.2f} ms")
    print(f"  P99           : {latency_metrics['p99_ms']:.2f} ms")
    print(f"  Min           : {latency_metrics['min_ms']:.2f} ms")
    print(f"  Max           : {latency_metrics['max_ms']:.2f} ms")
    print(f"  Std-dev       : {latency_metrics['stddev_ms']:.2f} ms")
    print(f"  QPS (serial)  : {latency_metrics['qps']:.1f}")

    print(f"\n{thin}")
    print("  BATCH QUERY")
    print(thin)
    print(f"  Total time    : {batch_stats['batch_time_ms']:.2f} ms")
    print(f"  QPS (batch)   : {batch_stats['batch_qps']:.1f}")

    print(f"\n{thin}")
    print("  ACCURACY")
    print(thin)
    print(f"  Recall@{config.TOP_K:<3}   : {recall * 100:.2f}%")

    print(f"\n{sep}")
    print("  Benchmark complete.")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Connect
    db.connect()

    # 2. Load dataset
    vectors, dimension = load_sift_vectors()
    query_vectors = vectors[: config.NQ]

    # 3. Create collection and ingest data
    collection = db.recreate_collection(dimension)
    insert_stats = db.insert_vectors(collection, vectors)

    # 4. Build index and load
    index_stats  = db.build_index(collection, len(vectors))
    load_stats   = db.load_collection(collection)

    # 5. Warm up
    benchmark.warm_up(collection, query_vectors)

    # 6. Build brute-force ground truth (needed for recall)
    ground_truth = benchmark.build_ground_truth(collection, query_vectors)

    # 7. Single-query benchmark
    single_stats = benchmark.run_single_query_benchmark(collection, query_vectors)
    latency_metrics = benchmark.aggregate_latency_metrics(single_stats["latencies"])

    # 8. Batch-query benchmark
    batch_stats = benchmark.run_batch_query_benchmark(collection, query_vectors)

    # 9. Recall@K
    recall = benchmark.compute_recall(
        ann_results=single_stats["results"],
        brute_force_results=ground_truth,
        k=config.TOP_K,
    )

    # 10. Memory from segment info
    segments   = db.get_segment_info()
    memory_mb  = db.estimate_memory_mb(segments)

    # 11. Print report
    print_report(
        insert_stats=insert_stats,
        index_stats=index_stats,
        load_stats=load_stats,
        latency_metrics=latency_metrics,
        batch_stats=batch_stats,
        recall=recall,
        memory_mb=memory_mb,
        dimension=dimension,
    )


if __name__ == "__main__":
    main()