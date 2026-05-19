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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

    # 1. Connect
    client = db.connect()

    # 2. Load vectors
    vectors, dimension = load_sift_vectors(
        dataset_name=config.DATASET_NAME,
        dataset_size=config.DATASET_SIZE,
        as_numpy=True,
    )

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

    # 7. Single-query benchmark
    single_stats = benchmark.run_single_query_benchmark(
        client,
        query_vectors,
    )
    latency_metrics = benchmark.aggregate_latency_metrics(single_stats["latencies"])

    # 8. Recall@K
    recall = benchmark.compute_recall(
        ann_results=single_stats["results"],
        ground_truth=ground_truth,
        k=config.TOP_K,
    )

    # 9. Memory estimate
    memory_mb = db.estimate_memory_mb(
        client,
        dimension,
        len(vectors),
    )

    # 10. Standard report + CSV
    report_row = build_report_row(
        backend="qdrant",
        dataset_name=config.DATASET_NAME,
        dataset_size=config.DATASET_SIZE,
        dimension=dimension,
        index_type=f"HNSW (m={config.HNSW_M}, ef_construct={config.HNSW_EF_CONSTRUCT}, hnsw_ef={config.HNSW_EF})",
        distance_metric=config.DISTANCE,
        top_k=config.TOP_K,
        query_count=config.NQ,
        insert_stats=insert_stats,
        index_stats=index_stats,
        latency_metrics=latency_metrics,
        recall=recall,
        memory_mb=memory_mb,
        load_time_s=None,
        batch_stats=None,
    )

    print_standard_report(report_row)

    output_csv = ROOT_DIR / "benchmark_results.csv"
    append_report_csv(report_row, output_csv)
    print(f"\nSaved benchmark results to: {output_csv}")


if __name__ == "__main__":
    main()