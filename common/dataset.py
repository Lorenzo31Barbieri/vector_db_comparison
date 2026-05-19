from datasets import load_dataset
import numpy as np


def load_sift_vectors(dataset_name: str, dataset_size: int, as_numpy: bool = False):
    print(f"\nLoading dataset '{dataset_name}' (first {dataset_size:,} vectors)...")
    dataset = load_dataset(dataset_name, "train")
    vectors = dataset["train"]["emb"][:dataset_size]

    if as_numpy:
        vectors = np.asarray(vectors, dtype=np.float32)
        dimension = int(vectors.shape[1])
    else:
        dimension = len(vectors[0])

    print(f"Loaded {len(vectors):,} vectors (dim={dimension})")
    return vectors, dimension
