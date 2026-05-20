import numpy as np


def safe_throughput(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return float("inf")
    return count / elapsed_seconds


def aggregate_latency_metrics(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "stddev_ms": 0.0,
            "qps": 0.0,
        }

    arr = np.array(latencies_ms)
    avg = float(np.mean(arr))
    return {
        "avg_ms": avg,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "stddev_ms": float(np.std(arr)),
        "qps": 1000.0 / avg if avg > 0 else float("inf"),
    }
