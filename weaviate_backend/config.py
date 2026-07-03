# ============================================================
# CONFIGURATION
# ============================================================

# --- Weaviate connection ---
WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051

# --- Collection ---
COLLECTION_NAME = "sift_benchmark"

# --- Dataset ---
DATASET_NAME = "open-vdb/sift-128-euclidean"
DATASET_SIZE = 100_000

# --- Insertion ---
BATCH_SIZE = 10_000

# --- HNSW ---
VECTOR_INDEX_TYPE = "hnsw"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100

# --- Sharding & Replication ---
SHARD_COUNT = 1
REPLICATION_FACTOR = 1

# --- Distance metric ---
DISTANCE = "L2_SQUARED"

# --- Query ---
TOP_K = 10
NQ = 100
HNSW_EF = 128
WARM_UP_QUERIES = 5