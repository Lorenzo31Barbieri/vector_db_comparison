from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.benchmark_workloads import selectivity_to_filter_bucket
from common.perf import aggregate_latency_metrics
from common.runtime_config import apply_runtime_overrides
from milvus import benchmark as milvus_benchmark
from milvus import config as milvus_config
from milvus import db as milvus_db


@dataclass
class MilvusAdapter:
    backend_name: str = "milvus"

    def supported_index_types(self) -> list[str]:
        return ["HNSW", "IVF_FLAT", "FLAT"]

    def set_index_type(self, index_type: str) -> None:
        normalized = str(index_type).upper()
        if normalized not in self.supported_index_types():
            raise ValueError(f"Unsupported Milvus index type '{index_type}'")
        milvus_config.INDEX_TYPE = normalized

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
        apply_runtime_overrides(
            milvus_config,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            top_k=top_k,
            query_count=query_count,
            batch_size=batch_size,
            warm_up_queries=warm_up_queries,
        )

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

    def run_ann(self, query_vectors: np.ndarray, *, ef: int | None = None, limit: int | None = None) -> dict:
        benchmark_result = milvus_benchmark.run_single_query_benchmark(
            self.collection,
            query_vectors.tolist(),
            ef=ef,
            limit=limit,
        )
        return {
            "ids": milvus_benchmark.extract_hit_ids(benchmark_result["results"], k=limit or milvus_config.TOP_K),
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
        }

    def build_ground_truth(self, query_vectors: np.ndarray, *, limit: int) -> list[list[int]]:
        results = milvus_benchmark.search(
            self.collection,
            query_vectors.tolist(),
            ef=milvus_config.HNSW_EF * 4,
            limit=limit,
        )
        return milvus_benchmark.extract_hit_ids(list(results), k=limit)

    def build_filtered_ground_truth(self, query_vectors: np.ndarray, *, limit: int, selectivity: float) -> list[list[int]]:
        threshold = selectivity_to_filter_bucket(selectivity)
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
        ef: int | None = None,
        limit: int | None = None,
    ) -> dict:
        threshold = selectivity_to_filter_bucket(selectivity)
        expr = f"bucket < {threshold}"
        benchmark_result = milvus_benchmark.run_single_query_benchmark(
            self.collection,
            query_vectors.tolist(),
            expr=expr,
            ef=ef,
            limit=limit,
        )
        effective_k = limit or milvus_config.TOP_K
        return {
            "ids": milvus_benchmark.extract_hit_ids(benchmark_result["results"], k=effective_k),
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
            "filter_expr": expr,
        }

    def search_one(self, query_vector: np.ndarray, *, ef: int | None = None, limit: int | None = None) -> list[int]:
        hits = milvus_benchmark.search(
            self.collection,
            [query_vector.tolist()],
            ef=ef,
            limit=limit,
        )[0]
        effective_k = limit or milvus_config.TOP_K
        return [hit.id for hit in hits[:effective_k]]
