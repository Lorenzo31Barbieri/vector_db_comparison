from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentTracker:
    output_dir: Path
    experiment_name: str

    def __post_init__(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.output_dir / f"{self.experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl_path = self.run_dir / "results.jsonl"
        self.results_csv_path = self.run_dir / "results.csv"
        self.manifest_path = self.run_dir / "manifest.json"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

    def append_result(self, result: dict[str, Any]) -> None:
        with self.results_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result) + "\n")

    def write_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        all_columns: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in all_columns:
                    all_columns.append(key)

        with self.results_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=all_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def flatten_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner_value in value.items():
                _visit(f"{prefix}.{key}" if prefix else str(key), inner_value)
            return

        if isinstance(value, list):
            flat[prefix] = json.dumps(value)
            return

        flat[prefix] = value

    _visit("", record)
    return flat
