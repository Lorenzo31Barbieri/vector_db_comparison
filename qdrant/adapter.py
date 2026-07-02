from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
from qdrant_client import QdrantClient

from common.benchmark_workloads import selectivity_to_filter_bucket
from common.perf import aggregate_latency_metrics
from common.runtime_config import apply_runtime_overrides
from qdrant import benchmark as qdrant_benchmark
from qdrant import config as qdrant_config
from qdrant import db as qdrant_db


@dataclass
class QdrantAdapter:
    backend_name: str = "qdrant"

    def supported_index_types(self) -> list[str]:
        return ["HNSW"]

    def set_index_type(self, index_type: str) -> None:
        normalized = str(index_type).upper()
        if normalized not in self.supported_index_types():
            raise ValueError(f"Unsupported Qdrant index type '{index_type}'")
        qdrant_config.VECTOR_INDEX_TYPE = normalized

    def set_hnsw_m(self, hnsw_m: int) -> None:
        qdrant_config.HNSW_M = int(hnsw_m)

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
            qdrant_config,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            top_k=top_k,
            query_count=query_count,
            batch_size=batch_size,
            warm_up_queries=warm_up_queries,
        )

    def prepare(self, vectors: np.ndarray) -> dict:
        client = qdrant_db.connect()
        qdrant_db.recreate_collection(client, int(vectors.shape[1]))

        vector_list = vectors.astype(np.float32).tolist()
        insert_stats = qdrant_db.insert_vectors(client, vector_list)
        index_stats = qdrant_db.build_index(client, len(vector_list))

        self.client = client
        self._thread_local = threading.local()

        return {
            "insert": insert_stats,
            "index": index_stats,
            "load": {"load_time": None},
            "dimension": int(vectors.shape[1]),
        }

    def _get_thread_client(self) -> QdrantClient:
        thread_client = getattr(self._thread_local, "client", None)
        if thread_client is None:
            thread_client = QdrantClient(
                host=qdrant_config.QDRANT_HOST,
                port=qdrant_config.QDRANT_PORT,
            )
            self._thread_local.client = thread_client
        return thread_client

    def warm_up(self, query_vectors: np.ndarray) -> None:
        qdrant_benchmark.warm_up(self.client, query_vectors)

    def run_ann(self, query_vectors: np.ndarray, *, hnsw_ef: int | None = None, limit: int | None = None) -> dict:
        benchmark_result = qdrant_benchmark.run_single_query_benchmark(
            self.client,
            query_vectors,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )
        return {
            "ids": benchmark_result["results"],
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
        }

    def build_ground_truth(self, query_vectors: np.ndarray, *, limit: int) -> list[list[int]]:
        return qdrant_benchmark.build_ground_truth(self.client, query_vectors, limit=limit)

    def build_filtered_ground_truth(self, query_vectors: np.ndarray, *, limit: int, selectivity: float) -> list[list[int]]:
        query_filter = qdrant_benchmark.build_bucket_filter(selectivity_to_filter_bucket(selectivity))
        return qdrant_benchmark.build_ground_truth(
            self.client,
            query_vectors,
            query_filter=query_filter,
            limit=limit,
        )

    def run_filtered_ann(
        self,
        query_vectors: np.ndarray,
        *,
        selectivity: float,
        hnsw_ef: int | None = None,
        limit: int | None = None,
    ) -> dict:
        threshold = selectivity_to_filter_bucket(selectivity)
        query_filter = qdrant_benchmark.build_bucket_filter(threshold)
        benchmark_result = qdrant_benchmark.run_single_query_benchmark(
            self.client,
            query_vectors,
            query_filter=query_filter,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )
        return {
            "ids": benchmark_result["results"],
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
            "filter_expr": {"bucket_lt": threshold},
        }

    def search_one(self, query_vector: np.ndarray, *, hnsw_ef: int | None = None, limit: int | None = None) -> list[int]:
        thread_client = self._get_thread_client()
        response = qdrant_benchmark.search(
            thread_client,
            query_vector,
            exact=False,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )
        effective_k = limit or qdrant_config.TOP_K
        return [hit.id for hit in response.points[:effective_k]]
