from __future__ import annotations

import time
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject

try:
    from . import config as config
except ImportError:
    import config as config
from common.perf import safe_throughput


def connect() -> weaviate.WeaviateClient:
    print(
        "Connecting to Weaviate at "
        f"{config.WEAVIATE_HOST}:{config.WEAVIATE_HTTP_PORT} "
        f"(grpc={config.WEAVIATE_GRPC_PORT})..."
    )

    client = weaviate.connect_to_local(
        host=config.WEAVIATE_HOST,
        port=config.WEAVIATE_HTTP_PORT,
        grpc_port=config.WEAVIATE_GRPC_PORT,
    )

    if not client.is_ready():
        raise RuntimeError("Connected to Weaviate, but instance is not ready")

    print("Connected!")
    return client


def recreate_collection(client: weaviate.WeaviateClient, dimension: int) -> None:
    exists = client.collections.exists(config.COLLECTION_NAME)

    if exists:
        print(f"Deleting existing collection '{config.COLLECTION_NAME}'...")
        client.collections.delete(config.COLLECTION_NAME)

    index_type = str(config.VECTOR_INDEX_TYPE).lower()
    if index_type == "hnsw":
        print(
            f"Creating collection '{config.COLLECTION_NAME}' "
            f"(index=hnsw, dim={dimension}, distance={config.DISTANCE}, "
            f"m={config.HNSW_M}, ef_construct={config.HNSW_EF_CONSTRUCT})..."
        )
        vector_index_config = Configure.VectorIndex.hnsw(
            max_connections=config.HNSW_M,
            ef_construction=config.HNSW_EF_CONSTRUCT,
            ef=config.HNSW_EF,
            distance_metric=VectorDistances[config.DISTANCE],
        )
    elif index_type == "flat":
        print(
            f"Creating collection '{config.COLLECTION_NAME}' "
            f"(index=flat, dim={dimension}, distance={config.DISTANCE})..."
        )
        vector_index_config = Configure.VectorIndex.flat(
            distance_metric=VectorDistances[config.DISTANCE],
        )
    else:
        raise ValueError(f"Unsupported Weaviate index type '{config.VECTOR_INDEX_TYPE}'")

    client.collections.create(
        name=config.COLLECTION_NAME,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=vector_index_config
        ),
        properties=[
            Property(name="bucket", data_type=DataType.INT),
            Property(name="label", data_type=DataType.TEXT),
        ],
    )

    print("Collection created!")


def insert_vectors(client: weaviate.WeaviateClient, vectors: list[list[float]]) -> dict:
    total = len(vectors)

    print(f"\nInserting {total:,} vectors in batches of {config.BATCH_SIZE:,}...")

    collection = client.collections.get(config.COLLECTION_NAME)
    start = time.time()

    for i in range(0, total, config.BATCH_SIZE):
        end_idx = min(i + config.BATCH_SIZE, total)

        batch_objects = [
            DataObject(
                properties={
                    "bucket": idx % 100,
                    "label": f"label_{idx % 10}",
                },
                vector=vectors[idx],
            )
            for idx in range(i, end_idx)
        ]

        collection.data.insert_many(batch_objects)

        if i > 0 and i % 50_000 == 0:
            print(f"  {i:,} vectors inserted...")

    elapsed = time.time() - start
    throughput = safe_throughput(total, elapsed)

    print(f"Insertion done in {elapsed:.2f}s ({throughput:,.0f} vectors/s)")

    return {
        "insert_time": elapsed,
        "throughput": throughput,
    }


def _object_count(collection: Any) -> int | None:
    try:
        aggregate = collection.aggregate.over_all(total_count=True)
        return int(getattr(aggregate, "total_count", 0) or 0)
    except Exception:
        return None


def build_index(
    client: weaviate.WeaviateClient,
    num_vectors: int,
) -> dict:
    print(f"\nSynchronizing {str(config.VECTOR_INDEX_TYPE).lower()} index...")

    collection = client.collections.get(config.COLLECTION_NAME)
    start = time.time()

    deadline = time.time() + 60
    while time.time() < deadline:
        count = _object_count(collection)
        if count is not None and count >= num_vectors:
            break
        time.sleep(0.5)

    elapsed = time.time() - start
    throughput = safe_throughput(num_vectors, elapsed)

    print(f"Index sync done in {elapsed:.2f}s ({throughput:,.0f} vectors/s)")

    return {
        "index_time": elapsed,
        "index_throughput": throughput,
    }