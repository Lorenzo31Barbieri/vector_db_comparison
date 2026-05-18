# --- Milvus connection ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_MAX_MSG_SIZE = 1_073_741_824  # 1 GB

# --- Collection ---
COLLECTION_NAME = "sift_benchmark"

# --- Dataset ---
DATASET_NAME = "open-vdb/sift-128-euclidean"
DATASET_SIZE = 100_000   # number of vectors to use from the dataset

# --- Insertion ---
BATCH_SIZE = 10_000

# --- Index ---
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NLIST = 1024

# --- Query ---
TOP_K = 10
NQ = 100          # number of query vectors
NPROBE = 5
WARM_UP_QUERIES = 5   # queries run before recording latencies (cache warm-up)