# openevolve/direction.py
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

@dataclass
class IslandState:
    baseline: np.ndarray | None = None
    slopes: Deque[float] = None

class DirectionTracker:
    """
    为每个 island 维护一个EMA基准方向 b_i，并记录最近k次斜率以检测平台期。
    """
    def __init__(self, dim: int, k_window=8, ema_decay=0.8):
        self.dim = dim
        self.k_window = k_window
        self.ema = ema_decay
        self.state = defaultdict(lambda: IslandState(None, deque(maxlen=k_window)))

    def update(self, island_id: int, v_parent: np.ndarray, v_child: np.ndarray, improved: bool):
        dv = v_child - v_parent
        st = self.state[island_id]
        if st.baseline is None:
            st.baseline = dv.copy()
        if improved:
            st.baseline = self.ema * st.baseline + (1 - self.ema) * dv
        b = st.baseline
        b_norm = np.linalg.norm(b) + 1e-8
        slope = float(np.dot(dv, b / b_norm))  # dv 在 b 方向的投影增益
        st.slopes.append(slope)
        slope_avg = float(np.mean(st.slopes))
        return dv, slope, slope_avg, st
