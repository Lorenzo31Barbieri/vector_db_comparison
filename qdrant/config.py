# ============================================================
# CONFIGURATION
# ============================================================

# --- Qdrant connection ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# --- Collection ---
COLLECTION_NAME = "sift_benchmark"

# --- Dataset ---
DATASET_NAME = "open-vdb/sift-128-euclidean"
DATASET_SIZE = 100_000

# --- Insertion ---
BATCH_SIZE = 10_000

# --- HNSW ---
VECTOR_INDEX_TYPE = "HNSW"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100

# --- Distance metric ---
DISTANCE = "EUCLID"

# --- Query ---
TOP_K = 10
NQ = 100
HNSW_EF = 128
WARM_UP_QUERIES = 5

# --- Optimizer ---
INDEXING_THRESHOLD = 20_000

# --- Cluster ---
SHARD_NUMBER = 1
REPLICATION_FACTOR = 1