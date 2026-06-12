import time

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

try:
    from . import config as config
except ImportError:
    import config as config
from common.perf import safe_throughput


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
        FieldSchema(name="bucket", dtype=DataType.INT64),
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
        ids = list(range(i, end_idx))
        buckets = [value % 100 for value in ids]
        collection.insert([ids, vectors[i:end_idx], buckets])
        if i > 0 and i % 100_000 == 0:
            print(f"  {i} vectors inserted...")

    collection.flush()
    elapsed = time.time() - start
    throughput = safe_throughput(total, elapsed)

    print(f"Insertion done in {elapsed:.2f}s  ({throughput:.0f} vectors/s)")
    return {"insert_time": elapsed, "throughput": throughput}


# ── Indexing ─────────────────────────────────────────────────────────────────

def build_index(collection: Collection, num_vectors: int) -> dict:
    """
    Build an ANN index on the vector field.

    Returns
    -------
    dict with keys: index_time (s), index_throughput (vectors/s)
    """
    index_type = str(config.INDEX_TYPE).upper()

    if index_type == "HNSW":
        print(f"\nBuilding {index_type} index (M={config.HNSW_M}, efConstruction={config.HNSW_EF_CONSTRUCT})...")
        index_params = {
            "index_type": index_type,
            "metric_type": config.METRIC_TYPE,
            "params": {"M": config.HNSW_M, "efConstruction": config.HNSW_EF_CONSTRUCT},
        }
    elif index_type == "IVF_FLAT":
        print(f"\nBuilding {index_type} index (nlist={config.IVF_NLIST})...")
        index_params = {
            "index_type": index_type,
            "metric_type": config.METRIC_TYPE,
            "params": {"nlist": config.IVF_NLIST},
        }
    elif index_type == "FLAT":
        print(f"\nBuilding {index_type} index...")
        index_params = {
            "index_type": index_type,
            "metric_type": config.METRIC_TYPE,
            "params": {},
        }
    else:
        raise ValueError(f"Unsupported Milvus index type '{config.INDEX_TYPE}'")

    start = time.time()
    collection.create_index(field_name="vector", index_params=index_params)
    elapsed = time.time() - start
    throughput = safe_throughput(num_vectors, elapsed)

    print(f"Index built in {elapsed:.2f}s  ({throughput:.0f} vectors/s)")
    return {"index_time": elapsed, "index_throughput": throughput}


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

