from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.perf import aggregate_latency_metrics
from milvus import benchmark as milvus_benchmark
from milvus import config as milvus_config
from milvus import db as milvus_db


@dataclass
class MilvusAdapter:
    backend_name: str = "milvus"

    def configure(
        self,
        *,
        dataset_name: str,
        dataset_size: int,
        top_k: int,
        query_count: int,
        batch_size: int,
        warm_up_queries: int,
    ) -> None:
        milvus_config.DATASET_NAME = dataset_name
        milvus_config.DATASET_SIZE = dataset_size
        milvus_config.TOP_K = top_k
        milvus_config.NQ = query_count
        milvus_config.BATCH_SIZE = batch_size
        milvus_config.WARM_UP_QUERIES = warm_up_queries

    def prepare(self, vectors: np.ndarray) -> dict:
        milvus_db.connect()
        collection = milvus_db.recreate_collection(int(vectors.shape[1]))

        vector_list = vectors.astype(np.float32).tolist()
        insert_stats = milvus_db.insert_vectors(collection, vector_list)
        index_stats = milvus_db.build_index(collection, len(vector_list))
        load_stats = milvus_db.load_collection(collection)

        segments = milvus_db.get_segment_info()
        memory_mb = milvus_db.estimate_memory_mb(segments)

        self.collection = collection

        return {
            "insert": insert_stats,
            "index": index_stats,
            "load": load_stats,
            "memory_mb": memory_mb,
            "storage_mb": None,
            "dimension": int(vectors.shape[1]),
        }

    def warm_up(self, query_vectors: np.ndarray) -> None:
        milvus_benchmark.warm_up(self.collection, query_vectors.tolist())

    def run_ann(self, query_vectors: np.ndarray, *, nprobe: int | None = None, limit: int | None = None) -> dict:
        benchmark_result = milvus_benchmark.run_single_query_benchmark(
            self.collection,
            query_vectors.tolist(),
            nprobe=nprobe,
            limit=limit,
        )
        return {
            "ids": milvus_benchmark.extract_hit_ids(benchmark_result["results"], k=limit or milvus_config.TOP_K),
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
        }

    def run_ann_batch(self, query_vectors: np.ndarray, *, nprobe: int | None = None, limit: int | None = None) -> dict:
        return milvus_benchmark.run_batch_query_benchmark(
            self.collection,
            query_vectors.tolist(),
            nprobe=nprobe,
            limit=limit,
        )

    def build_ground_truth(self, query_vectors: np.ndarray, *, limit: int) -> list[list[int]]:
        results = milvus_benchmark.search(
            self.collection,
            query_vectors.tolist(),
            nprobe=milvus_config.NLIST,
            limit=limit,
        )
        return milvus_benchmark.extract_hit_ids(list(results), k=limit)

    def build_filtered_ground_truth(self, query_vectors: np.ndarray, *, limit: int, selectivity: float) -> list[list[int]]:
        threshold = self._selectivity_threshold(selectivity)
        expr = f"bucket < {threshold}"
        results = milvus_benchmark.build_ground_truth_filtered(
            self.collection,
            query_vectors.tolist(),
            expr,
            limit=limit,
        )
        return milvus_benchmark.extract_hit_ids(results, k=limit)

    def run_filtered_ann(
        self,
        query_vectors: np.ndarray,
        *,
        selectivity: float,
        nprobe: int | None = None,
        limit: int | None = None,
    ) -> dict:
        threshold = self._selectivity_threshold(selectivity)
        expr = f"bucket < {threshold}"
        benchmark_result = milvus_benchmark.run_single_query_benchmark(
            self.collection,
            query_vectors.tolist(),
            expr=expr,
            nprobe=nprobe,
            limit=limit,
        )
        effective_k = limit or milvus_config.TOP_K
        return {
            "ids": milvus_benchmark.extract_hit_ids(benchmark_result["results"], k=effective_k),
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
            "filter_expr": expr,
        }

    def search_one(self, query_vector: np.ndarray, *, nprobe: int | None = None, limit: int | None = None) -> list[int]:
        hits = milvus_benchmark.search(
            self.collection,
            [query_vector.tolist()],
            nprobe=nprobe,
            limit=limit,
        )[0]
        effective_k = limit or milvus_config.TOP_K
        return [hit.id for hit in hits[:effective_k]]

    @staticmethod
    def _selectivity_threshold(selectivity: float) -> int:
        bounded = max(0.0, min(1.0, float(selectivity)))
        return max(1, int(round(100 * bounded)))
