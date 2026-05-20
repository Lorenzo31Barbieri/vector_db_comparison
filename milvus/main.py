from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from benchmark_runner import run_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suite for Milvus only")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "benchmark_configs" / "suite.default.json"),
        help="Path to benchmark suite config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_suite(args.config, backends=["milvus"])
    print(f"\nMilvus benchmark run complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()
