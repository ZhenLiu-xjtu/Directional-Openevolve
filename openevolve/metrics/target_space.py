# openevolve/metrics/target_space.py
from dataclasses import dataclass
import numpy as np

@dataclass
class RunningStats:
    mean: dict
    std: dict
    count: int = 0

DEFAULT_KEYS = ["combined_score", "params", "latency_ms", "flops", "mem_mb"]

def _z(x, m, s):
    return 0.0 if s == 0 else (x - m) / s

def make_target_vector(metrics: dict, stats: RunningStats, weights: dict) -> np.ndarray:
    # 方向符号：性能正向，资源负向（越小越好）
    raw = {
        "combined_score": +metrics.get("combined_score", 0.0),
        "params":        -metrics.get("params", 0.0),
        "latency_ms":    -metrics.get("latency_ms", metrics.get("latency", 0.0)),
        "flops":         -metrics.get("flops", metrics.get("FLOPs", 0.0)),
        "mem_mb":        -metrics.get("mem_mb", metrics.get("memory_mb", 0.0)),
    }
    vec = []
    for k in DEFAULT_KEYS:
        w = float(weights.get(k if k!="combined_score" else "score", 1.0))
        vec.append(w * _z(raw[k], stats.mean.get(k, 0.0), stats.std.get(k, 1.0)))
    return np.asarray(vec, dtype=np.float32)
