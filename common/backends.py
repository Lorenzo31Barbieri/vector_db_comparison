from __future__ import annotations

from collections.abc import Sequence

from milvus.adapter import MilvusAdapter
from qdrant.adapter import QdrantAdapter
from weaviate_backend.adapter import WeaviateAdapter

SUPPORTED_BACKENDS: tuple[str, ...] = ("milvus", "qdrant", "weaviate")

def adapter_for_backend(backend: str):
    if backend == "milvus":
        return MilvusAdapter()
    if backend == "qdrant":
        return QdrantAdapter()
    if backend == "weaviate":
        return WeaviateAdapter()

def resolve_backends(backends: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    for backend in backends:
        normalized = str(backend).strip().lower()
        if normalized not in SUPPORTED_BACKENDS:
            supported = ", ".join(SUPPORTED_BACKENDS)
            raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {supported}")
        if normalized in seen:
            continue
        resolved.append(normalized)
        seen.add(normalized)

    if not resolved:
        raise ValueError("At least one backend must be configured")

    return resolved
