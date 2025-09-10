# openevolve/direction/types.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class Anchor:
    program_id: str
    island_id: int
    metrics: Dict[str, float]     # e.g., {"acc": 0.91, "latency_ms": 2371.5}
    resources: Dict[str, float]   # e.g., {"params": 3.01e6, "flops": 1.578e6, "mem_mb": 17.0}
    features: Dict[str, float]    # MAP-Elites 或你自定义的特征维度

@dataclass
class DirectionFeedback:
    # 供 LLM 的自然语言提示
    text: str
    # 目标方向的“向量表征”（便于调度层或可视化）
    direction_vec: List[float]
    # 建议的“步长”（投影增益估计，防止一步跨过最优）
    step_size: float
    # 选中的锚点（可能为 None）
    anchor: Optional[Anchor] = None
    # 附加信息（日志、可视化、动作建议明细）
    meta: Optional[Dict[str, Any]] = None
