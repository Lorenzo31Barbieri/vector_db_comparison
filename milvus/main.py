import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config as config
import db as db
import benchmark as benchmark
from common.dataset import load_sift_vectors
from common.reporting import build_report_row, print_standard_report, append_report_csv


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Connect
    db.connect()

    # 2. Load dataset
    vectors, dimension = load_sift_vectors(
        dataset_name=config.DATASET_NAME,
        dataset_size=config.DATASET_SIZE,
        as_numpy=False,
    )
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

    # 11. Standard report + CSV
    report_row = build_report_row(
        backend="milvus",
        dataset_name=config.DATASET_NAME,
        dataset_size=config.DATASET_SIZE,
        dimension=dimension,
        index_type=f"{config.INDEX_TYPE} (nlist={config.NLIST}, nprobe={config.NPROBE})",
        distance_metric=config.METRIC_TYPE,
        top_k=config.TOP_K,
        query_count=config.NQ,
        insert_stats=insert_stats,
        index_stats=index_stats,
        latency_metrics=latency_metrics,
        batch_stats=batch_stats,
        recall=recall,
        memory_mb=memory_mb,
        load_time_s=load_stats["load_time"],
    )

    print_standard_report(report_row)

    output_csv = ROOT_DIR / "benchmark_results.csv"
    append_report_csv(report_row, output_csv)
    print(f"\nSaved benchmark results to: {output_csv}")


if __name__ == "__main__":
    main()