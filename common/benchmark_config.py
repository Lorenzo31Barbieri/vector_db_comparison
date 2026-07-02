from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.backends import resolve_backends


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    repeats: int = 5
    output_dir: str = "results"
    notes: str = ""


@dataclass
class DatasetConfig:
    name: str
    sizes: list[int]
    query_count: int
    top_k: int


@dataclass
class ScenarioConfig:
    enabled: bool = True


@dataclass
class AnnScenarioConfig(ScenarioConfig):
    hnsw_ef_values: list[int] = field(default_factory=lambda: [64, 128, 256])


@dataclass
class ConcurrencyScenarioConfig(ScenarioConfig):
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])


@dataclass
class FilteringScenarioConfig(ScenarioConfig):
    selectivities: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.01])


@dataclass
class HybridScenarioConfig(ScenarioConfig):
    mode: str = "metadata_rerank_placeholder"


@dataclass
class IndexComparisonScenarioConfig(ScenarioConfig):
    index_types_by_backend: dict[str, list[str]] = field(
        default_factory=lambda: {
            "milvus": ["HNSW", "IVF_FLAT", "FLAT"],
            "qdrant": ["HNSW"],
            "weaviate": ["hnsw", "flat"],
        }
    )
    ann_hnsw_ef: int = 128


@dataclass
class HnswMScenarioConfig(ScenarioConfig):
    enabled: bool = False
    m_values: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    ann_hnsw_ef: int = 128


@dataclass
class SuiteConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    backends: list[str]
    ann: AnnScenarioConfig = field(default_factory=AnnScenarioConfig)
    concurrency: ConcurrencyScenarioConfig = field(default_factory=ConcurrencyScenarioConfig)
    filtering: FilteringScenarioConfig = field(default_factory=FilteringScenarioConfig)
    hybrid: HybridScenarioConfig = field(default_factory=HybridScenarioConfig)
    index_comparison: IndexComparisonScenarioConfig = field(default_factory=IndexComparisonScenarioConfig)
    hnsw_m: HnswMScenarioConfig = field(default_factory=HnswMScenarioConfig)


DEFAULT_CONFIG = {
    "experiment": {
        "name": "vectordb_phase2_suite",
        "seed": 42,
        "repeats": 5,
        "output_dir": "results",
        "notes": "Phase 2 benchmark suite",
    },
    "dataset": {
        "name": "open-vdb/sift-128-euclidean",
        "sizes": [100000],
        "query_count": 200,
        "top_k": 10,
    },
    "backends": ["milvus", "qdrant", "weaviate"],
    "ann": {
        "enabled": True,
        "hnsw_ef_values": [64, 128, 256],
    },
    "concurrency": {
        "enabled": True,
        "concurrency_levels": [1, 2, 4, 8, 16],
    },
    "filtering": {
        "enabled": True,
        "selectivities": [0.5, 0.1, 0.01],
    },
    "hybrid": {
        "enabled": False,
        "mode": "metadata_rerank_placeholder",
    },
    "index_comparison": {
        "enabled": False,
        "index_types_by_backend": {
            "milvus": ["HNSW", "IVF_FLAT", "FLAT"],
            "qdrant": ["HNSW"],
            "weaviate": ["hnsw", "flat"]
        },
        "ann_hnsw_ef": 128
    },
    "hnsw_m": {
        "enabled": False,
        "m_values": [8, 16, 32, 64],
        "ann_hnsw_ef": 128
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_suite_config(config_path: str | Path) -> SuiteConfig:
    path = Path(config_path)
    raw = _load_json(path)

    experiment = ExperimentConfig(**raw["experiment"])
    dataset = DatasetConfig(**raw["dataset"])
    ann = AnnScenarioConfig(**raw.get("ann", {}))
    concurrency = ConcurrencyScenarioConfig(**raw.get("concurrency", {}))
    filtering = FilteringScenarioConfig(**raw.get("filtering", {}))
    hybrid = HybridScenarioConfig(**raw.get("hybrid", {}))
    index_comparison = IndexComparisonScenarioConfig(**raw.get("index_comparison", {}))
    hnsw_m = HnswMScenarioConfig(**raw.get("hnsw_m", {}))

    return SuiteConfig(
        experiment=experiment,
        dataset=dataset,
        backends=resolve_backends(raw["backends"]),
        ann=ann,
        concurrency=concurrency,
        filtering=filtering,
        hybrid=hybrid,
        index_comparison=index_comparison,
        hnsw_m=hnsw_m,
    )


def write_default_suite_config(config_path: str | Path) -> Path:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(DEFAULT_CONFIG, file, indent=2)
    return path
