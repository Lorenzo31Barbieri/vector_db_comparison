from __future__ import annotations

from dataclasses import dataclass
import threading
import sys
from pathlib import Path

import numpy as np

from common.benchmark_workloads import selectivity_to_filter_bucket
from common.perf import aggregate_latency_metrics
from common.runtime_config import apply_runtime_overrides

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

import benchmark as weaviate_benchmark
import config as weaviate_config
import db as weaviate_db


@dataclass
class WeaviateAdapter:
    backend_name: str = "weaviate"

    def supported_index_types(self) -> list[str]:
        return ["hnsw", "flat"]

    def set_index_type(self, index_type: str) -> None:
        normalized = str(index_type).lower()
        if normalized not in self.supported_index_types():
            raise ValueError(f"Unsupported Weaviate index type '{index_type}'")
        weaviate_config.VECTOR_INDEX_TYPE = normalized

    def set_hnsw_m(self, hnsw_m: int) -> None:
        weaviate_config.HNSW_M = int(hnsw_m)

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
            weaviate_config,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            top_k=top_k,
            query_count=query_count,
            batch_size=batch_size,
            warm_up_queries=warm_up_queries,
        )

    def prepare(self, vectors: np.ndarray) -> dict:
        client = weaviate_db.connect()
        weaviate_db.recreate_collection(client, int(vectors.shape[1]))

        vector_list = vectors.astype(np.float32).tolist()
        insert_stats = weaviate_db.insert_vectors(client, vector_list)
        index_stats = weaviate_db.build_index(client, len(vector_list))

        self.client = client
        self.collection = client.collections.get(weaviate_config.COLLECTION_NAME)
        self._thread_local = threading.local()
        self._thread_clients: list = []
        self._thread_clients_lock = threading.Lock()

        return {
            "insert": insert_stats,
            "index": index_stats,
            "load": {"load_time": None},
            "dimension": int(vectors.shape[1]),
        }

    def _get_thread_collection(self):
        thread_client = getattr(self._thread_local, "client", None)
        if thread_client is None:
            thread_client = weaviate_db.connect()
            self._thread_local.client = thread_client
            with self._thread_clients_lock:
                self._thread_clients.append(thread_client)

        return thread_client.collections.get(weaviate_config.COLLECTION_NAME)

    def warm_up(self, query_vectors: np.ndarray) -> None:
        weaviate_benchmark.warm_up(self.collection, query_vectors)

    def run_ann(self, query_vectors: np.ndarray, *, hnsw_ef: int | None = None, limit: int | None = None) -> dict:
        benchmark_result = weaviate_benchmark.run_single_query_benchmark(
            self.collection,
            query_vectors,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )
        return {
            "ids": benchmark_result["results"],
            "latency": aggregate_latency_metrics(benchmark_result["latencies"]),
            "raw": benchmark_result,
        }

    def build_ground_truth(self, query_vectors: np.ndarray, *, limit: int) -> list[list[str]]:
        return weaviate_benchmark.build_ground_truth(self.collection, query_vectors, limit=limit)

    def build_filtered_ground_truth(self, query_vectors: np.ndarray, *, limit: int, selectivity: float) -> list[list[str]]:
        query_filter = weaviate_benchmark.build_bucket_filter(selectivity_to_filter_bucket(selectivity))
        return weaviate_benchmark.build_ground_truth(
            self.collection,
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
        query_filter = weaviate_benchmark.build_bucket_filter(threshold)
        benchmark_result = weaviate_benchmark.run_single_query_benchmark(
            self.collection,
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

    def search_one(self, query_vector: np.ndarray, *, hnsw_ef: int | None = None, limit: int | None = None) -> list[str]:
        collection = self._get_thread_collection()
        response = weaviate_benchmark.search(
            collection,
            query_vector,
            hnsw_ef=hnsw_ef,
            limit=limit,
        )
        effective_k = limit or weaviate_config.TOP_K
        return [str(obj.uuid) for obj in response.objects[:effective_k]]

    def teardown(self) -> None:
        seen: set[int] = set()

        for client in [getattr(self, "client", None), *getattr(self, "_thread_clients", [])]:
            if client is None:
                continue
            client_id = id(client)
            if client_id in seen:
                continue
            seen.add(client_id)
            try:
                client.close()
            except Exception:
                pass