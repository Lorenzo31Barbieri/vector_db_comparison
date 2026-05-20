from __future__ import annotations


def apply_runtime_overrides(
    config_module,
    *,
    dataset_name: str,
    dataset_size: int,
    top_k: int,
    query_count: int,
    batch_size: int,
    warm_up_queries: int,
) -> None:
    config_module.DATASET_NAME = dataset_name
    config_module.DATASET_SIZE = dataset_size
    config_module.TOP_K = top_k
    config_module.NQ = query_count
    config_module.BATCH_SIZE = batch_size
    config_module.WARM_UP_QUERIES = warm_up_queries
