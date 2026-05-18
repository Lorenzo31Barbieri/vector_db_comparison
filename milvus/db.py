import time

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

import config as config


# ── Connection ──────────────────────────────────────────────────────────────

def connect() -> None:
    print("Connecting to Milvus...")
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
        max_receive_message_length=config.MILVUS_MAX_MSG_SIZE,
    )
    print("Connected!")


# ── Collection lifecycle ─────────────────────────────────────────────────────

def recreate_collection(dimension: int) -> Collection:
    """Drop (if exists) and create a fresh collection."""
    if utility.has_collection(config.COLLECTION_NAME):
        print(f"Dropping existing collection '{config.COLLECTION_NAME}'...")
        utility.drop_collection(config.COLLECTION_NAME)

    print("Creating collection...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]
    schema = CollectionSchema(fields, description="SIFT Benchmark")
    collection = Collection(name=config.COLLECTION_NAME, schema=schema)
    print("Collection created!")
    return collection


# ── Insertion ────────────────────────────────────────────────────────────────

def insert_vectors(collection: Collection, vectors: list) -> dict:
    """
    Insert all vectors in batches.

    Returns
    -------
    dict with keys: insert_time (s), throughput (vectors/s)
    """
    total = len(vectors)
    print(f"\nInserting {total} vectors in batches of {config.BATCH_SIZE}...")

    start = time.time()

    for i in range(0, total, config.BATCH_SIZE):
        end_idx = min(i + config.BATCH_SIZE, total)
        collection.insert([list(range(i, end_idx)), vectors[i:end_idx]])
        if i > 0 and i % 100_000 == 0:
            print(f"  {i} vectors inserted...")

    collection.flush()
    elapsed = time.time() - start

    print(f"Insertion done in {elapsed:.2f}s  ({total / elapsed:.0f} vectors/s)")
    return {"insert_time": elapsed, "throughput": total / elapsed}


# ── Indexing ─────────────────────────────────────────────────────────────────

def build_index(collection: Collection, num_vectors: int) -> dict:
    """
    Build an ANN index on the vector field.

    Returns
    -------
    dict with keys: index_time (s), index_throughput (vectors/s)
    """
    print(f"\nBuilding {config.INDEX_TYPE} index (nlist={config.NLIST})...")
    index_params = {
        "index_type": config.INDEX_TYPE,
        "metric_type": config.METRIC_TYPE,
        "params": {"nlist": config.NLIST},
    }

    start = time.time()
    collection.create_index(field_name="vector", index_params=index_params)
    elapsed = time.time() - start

    print(f"Index built in {elapsed:.2f}s  ({num_vectors / elapsed:.0f} vectors/s)")
    return {"index_time": elapsed, "index_throughput": num_vectors / elapsed}


# ── Load into memory ─────────────────────────────────────────────────────────

def load_collection(collection: Collection) -> dict:
    """
    Load the collection into query-node memory.

    Returns
    -------
    dict with key: load_time (s)
    """
    print("\nLoading collection into memory...")
    start = time.time()
    collection.load()
    elapsed = time.time() - start
    print(f"Loaded in {elapsed:.2f}s")
    return {"load_time": elapsed}


# ── Segment / memory info ─────────────────────────────────────────────────────

def get_segment_info() -> list:
    return utility.get_query_segment_info(config.COLLECTION_NAME)


def estimate_memory_mb(segments: list) -> float:
    """
    Sum memory_size field from all segments (bytes → MB).
    Falls back to 0.0 if the field is unavailable.
    """
    total_bytes = sum(getattr(seg, "memory_size", 0) for seg in segments)
    return total_bytes / (1024 ** 2)