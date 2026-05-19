import time
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
)

import config as config
from common.perf import safe_throughput


# ─────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────

def connect() -> QdrantClient:
    print(f"Connecting to Qdrant at "
          f"{config.QDRANT_HOST}:{config.QDRANT_PORT}...")

    client = QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
    )

    client.get_collections()

    print("Connected!")
    return client


# ─────────────────────────────────────────────────────────────
# Collection lifecycle
# ─────────────────────────────────────────────────────────────

def recreate_collection(client: QdrantClient, dimension: int) -> None:

    existing = {
        c.name for c in client.get_collections().collections
    }

    if config.COLLECTION_NAME in existing:
        print(f"Deleting existing collection "
              f"'{config.COLLECTION_NAME}'...")
        client.delete_collection(config.COLLECTION_NAME)

    print(
        f"Creating collection '{config.COLLECTION_NAME}' "
        f"(dim={dimension}, "
        f"distance={config.DISTANCE}, "
        f"m={config.HNSW_M}, "
        f"ef_construct={config.HNSW_EF_CONSTRUCT})..."
    )

    client.create_collection(
        collection_name=config.COLLECTION_NAME,

        vectors_config=VectorParams(
            size=dimension,
            distance=Distance[config.DISTANCE],
        ),

        hnsw_config=HnswConfigDiff(
            m=config.HNSW_M,
            ef_construct=config.HNSW_EF_CONSTRUCT,
            on_disk=False,
        ),

        # Disable indexing during bulk ingestion
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0
        ),
    )

    print("Collection created!")


# ─────────────────────────────────────────────────────────────
# Insert vectors
# ─────────────────────────────────────────────────────────────

def insert_vectors(
    client: QdrantClient,
    vectors: List[List[float]],
) -> dict:

    total = len(vectors)

    print(f"\nInserting {total:,} vectors "
          f"in batches of {config.BATCH_SIZE:,}...")

    start = time.time()

    for i in range(0, total, config.BATCH_SIZE):

        end_idx = min(i + config.BATCH_SIZE, total)

        batch_vectors = vectors[i:end_idx]

        batch = [
            PointStruct(
                id=i + offset,
                vector=vec,
            )
            for offset, vec in enumerate(batch_vectors)
        ]

        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=batch,
            wait=False,
        )

        if i > 0 and i % 50_000 == 0:
            print(f"  {i:,} vectors inserted...")

    elapsed = time.time() - start
    throughput = safe_throughput(total, elapsed)

    print(f"Insertion done in {elapsed:.2f}s "
          f"({throughput:,.0f} vectors/s)")

    return {
        "insert_time": elapsed,
        "throughput": throughput,
    }


# ─────────────────────────────────────────────────────────────
# Build index
# ─────────────────────────────────────────────────────────────

def build_index(
    client: QdrantClient,
    num_vectors: int,
) -> dict:

    print(
        f"\nBuilding HNSW index "
        f"(m={config.HNSW_M}, "
        f"ef_construct={config.HNSW_EF_CONSTRUCT})..."
    )

    start = time.time()

    # Re-enable indexing
    client.update_collection(
        collection_name=config.COLLECTION_NAME,
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=config.INDEXING_THRESHOLD
        ),
    )

    time.sleep(2)

    consecutive_idle = 0

    while True:

        info = client.get_collection(config.COLLECTION_NAME)

        optimizer_status = getattr(
            info,
            "optimizer_status",
            None,
        )

        idle = False

        if optimizer_status is None:
            idle = True

        elif hasattr(optimizer_status, "error"):
            raise RuntimeError(
                f"Qdrant optimizer error: "
                f"{optimizer_status.error}"
            )

        elif hasattr(optimizer_status, "ok"):
            idle = bool(optimizer_status.ok)

        else:
            idle = "ok" in str(optimizer_status).lower()

        indexed = getattr(
            info,
            "indexed_vectors_count",
            0,
        ) or 0

        if idle:
            consecutive_idle += 1

            if consecutive_idle >= 2:
                break

        else:
            consecutive_idle = 0

            if indexed > 0:
                print(
                    f"  Indexing... "
                    f"{indexed:,}/{num_vectors:,}"
                )
            else:
                print("  Indexing in progress...")

        time.sleep(1)

    elapsed = time.time() - start

    throughput = safe_throughput(num_vectors, elapsed)

    print(
        f"Index built in {elapsed:.2f}s "
        f"({throughput:,.0f} vectors/s)"
    )

    return {
        "index_time": elapsed,
        "index_throughput": throughput,
    }


# ─────────────────────────────────────────────────────────────
# Collection info
# ─────────────────────────────────────────────────────────────

def get_collection_info(client: QdrantClient):
    return client.get_collection(config.COLLECTION_NAME)


# ─────────────────────────────────────────────────────────────
# Memory estimate
# ─────────────────────────────────────────────────────────────

def estimate_memory_mb(
    client: QdrantClient,
    dimension: int,
    num_vectors: int,
) -> float:

    try:
        info = client.get_collection(config.COLLECTION_NAME)

        ram = getattr(info, "ram_data_size", None)

        if ram:
            return ram / (1024 ** 2)

    except Exception:
        pass

    # fallback estimate

    vectors_bytes = dimension * 4 * num_vectors

    hnsw_bytes = (
        config.HNSW_M
        * 2
        * 8
        * num_vectors
    )

    total = vectors_bytes + hnsw_bytes

    return total / (1024 ** 2)