# openevolve/direction/anchor_manager.py
import math
from typing import Dict, Any, List, Optional, Tuple
from .types import Anchor, DirectionFeedback

def _cosine(a: List[float], b: List[float]) -> float:
    import numpy as np
    ax, bx = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(ax) + 1e-12, np.linalg.norm(bx) + 1e-12
    return float(np.dot(ax, bx) / (na * nb))

def _sub(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> List[float]:
    return [float(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys]

class AnchorManager:
    """
    负责：
      1) 从全局/各岛历史中挑选“前沿锚点”（同资源、异源、幻想）
      2) 计算目标“方向向量”与“步长”（单位增益/投影增益）
      3) 产出面向 LLM 的自然语言 directional feedback
    """
    def __init__(self, *,
                 get_all_programs,     # Callable[[], List[Dict[str, Any]]]
                 get_island_frontiers, # Callable[[], Dict[int, Dict[str, Any]]]
                 config: Dict[str, Any]):
        self.get_all_programs = get_all_programs
        self.get_island_frontiers = get_island_frontiers
        self.cfg = config or {}

        # 可调权重/阈值（提供缺省）
        dcfg = self.cfg.get("directional_feedback", {})
        self.metric_keys = dcfg.get("metric_keys", ["acc", "latency_ms", "params"])
        self.resource_tol = dcfg.get("resource_tolerances",
                                     {"params_pct": 0.1, "flops_pct": 0.2, "mem_pct": 0.15})
        self.weights = dcfg.get("weights", {"perf": 1.0, "latency": 0.5, "params": 0.3})
        self.stag = dcfg.get("stagnation", {"k": 5, "slope_eps": 1e-3})
        self.diverse_penalty_cos = dcfg.get("diversify_penalty_cos", 0.92)
        self.actions = dcfg.get("actions", {
            "prefer_ops": ["addmv"], "avoid_ops": ["mm", "dot"],
            "max_param_increase_pct": 0.05
        })

    # ----------------- 公开入口 -----------------
    def suggest(self, *, current: Dict[str, Any],
                island_id: int,
                history_of_island: List[Dict[str, Any]],
                other_island_dirs: Optional[List[List[float]]] = None) -> DirectionFeedback:
        """
        current: 当前岛屿的“父程序/最优程序”字典（含 metrics/resources/features）
        history_of_island: 该岛最近若干轮最佳历史（按时间升序）
        other_island_dirs: 其它岛屿本轮已选方向（用于相似度惩罚/正交）
        """
        # 1) 选锚点
        anchor = ( self._same_resource_anchor(current)
                   or self._hetero_anchor(current)
                   or self._hallucinated_anchor(current, history_of_island) )

        # 2) 计算方向与步长
        direction_vec, step = self._direction_and_step(current, anchor, history_of_island)

        # 3) 与其它岛屿方向去重 / 正交微旋
        if other_island_dirs:
            for d in other_island_dirs:
                if _cosine(direction_vec, d) > self.diverse_penalty_cos:
                    # 简单“补洞”：对未覆盖维度做微旋（+/- 5 度）
                    direction_vec = self._small_rotate(direction_vec, degrees=5.0)
                    break

        # 4) 生成自然语言动作建议
        text = self._render_text(current, anchor, direction_vec, step)

        return DirectionFeedback(
            text=text, direction_vec=direction_vec, step_size=step,
            anchor=anchor, meta={"actions": self.actions}
        )

    # ----------------- 细节实现 -----------------
    def _same_resource_anchor(self, cur: Dict[str, Any]) -> Optional[Anchor]:
        """ 同资源锚点：资源（params/flops/mem）在一定容差内、指标更优 """
        all_ps = self.get_all_programs()
        cur_r = cur.get("resources", {})
        best = None; best_acc_gain = 0.0
        for p in all_ps:
            if p.get("id") == cur.get("id"):
                continue
            r = p.get("resources", {})
            if not self._resource_close(cur_r, r):  # 资源相近
                continue
            acc_gain = p["metrics"].get("acc", 0) - cur["metrics"].get("acc", 0)
            if acc_gain > best_acc_gain:
                best_acc_gain = acc_gain; best = p
        return self._wrap(best)

    def _hetero_anchor(self, cur: Dict[str, Any]) -> Optional[Anchor]:
        """ 异源锚点：在特征空间（多维）上与当前差异更大、但目标指标更强或潜力更高 """
        all_ps = self.get_all_programs()
        cur_f = cur.get("features", {})
        cur_acc = cur["metrics"].get("acc", 0)
        best = None; best_score = -1e9
        for p in all_ps:
            if p.get("id") == cur.get("id"):
                continue
            # 用 L2 / 角距离等衡量“思路差异”
            diff = self._feature_distance(cur_f, p.get("features", {}))
            # 兼顾性能（acc）与差异性
            score = 0.7 * (p["metrics"].get("acc", 0) - cur_acc) + 0.3 * diff
            if score > best_score:
                best_score = score; best = p
        return self._wrap(best)

    def _hallucinated_anchor(self, cur: Dict[str, Any], hist: List[Dict[str, Any]]) -> Optional[Anchor]:
        """ 幻想锚点：用最近 k 轮“单位增益”外推一个目标 """
        k = self.stag.get("k", 5)
        if len(hist) < 2:
            return None
        tail = hist[-min(k, len(hist)):]
        # 估算 acc/unit_param、acc/unit_ms 的斜率（避免 0 除）
        def safe_div(a, b): return a / (b + 1e-9)
        d_acc = tail[-1]["metrics"]["acc"] - tail[0]["metrics"]["acc"]
        d_param = tail[-1]["resources"]["params"] - tail[0]["resources"]["params"]
        d_time  = tail[-1]["metrics"].get("latency_ms", 0) - tail[0]["metrics"].get("latency_ms", 0)

        acc_target = cur["metrics"]["acc"] + 0.8 * ( safe_div(d_acc, abs(d_param)) + safe_div(d_acc, abs(d_time)) )
        params_target = cur["resources"]["params"] * (1.0 + self.actions.get("max_param_increase_pct", 0.05))

        fake = {
            "id": "virtual_anchor",
            "island_id": cur.get("island_id", -1),
            "metrics": {"acc": float(acc_target), "latency_ms": max(0.0, cur["metrics"].get("latency_ms", 0) - 0.1*abs(d_time))},
            "resources": {"params": float(params_target),
                          "flops": cur["resources"].get("flops", 0.0),
                          "mem_mb": cur["resources"].get("mem_mb", 0.0)},
            "features": cur.get("features", {}).copy()
        }
        return self._wrap(fake)

    def _direction_and_step(self, cur: Dict[str, Any], anc: Anchor,
                            hist: List[Dict[str, Any]]) -> Tuple[List[float], float]:
        # 方向：在 (性能↑, 延迟↓, 参数↓) 三维合成向量
        cur_m = cur["metrics"]; cur_r = cur["resources"]
        tar_m = anc.metrics;    tar_r = anc.resources
        # Δ（注意把 “想要减小的量”取负方向）
        dv = [
            self.weights["perf"]    * (tar_m.get("acc", 0.0) - cur_m.get("acc", 0.0)),
            -self.weights["latency"]* ((tar_m.get("latency_ms", 0.0) - cur_m.get("latency_ms", 0.0))),
            -self.weights["params"] * ((tar_r.get("params", 0.0) - cur_r.get("params", 0.0))),
        ]
        # 步长：最近 k 轮 “目标指标” 在该方向上的投影增益（防跨越最优）
        k = self.stag.get("k", 5)
        tail = hist[-min(k, len(hist)):] if hist else []
        if len(tail) >= 2:
            gain = (tail[-1]["metrics"]["acc"] - tail[0]["metrics"]["acc"])
            delta_params = (tail[-1]["resources"]["params"] - tail[0]["resources"]["params"])
            step = float(max(0.1, min(1.0, gain / (abs(delta_params) + 1e-9))))
        else:
            step = 0.3
        return dv, step

    def _render_text(self, cur: Dict[str, Any], anc: Anchor,
                     direction_vec: List[float], step: float) -> str:
        # 生成供 LLM 使用的“动作化方向提示”
        cur_acc = cur["metrics"].get("acc", 0.0)
        tar_acc = anc.metrics.get("acc", cur_acc)
        params_budget = cur["resources"]["params"] * (1.0 + self.actions.get("max_param_increase_pct", 0.05))
        prefer_ops = ", ".join(self.actions.get("prefer_ops", []))
        avoid_ops  = ", ".join(self.actions.get("avoid_ops", []))
        return (
            f"[Directional Feedback]\n"
            f"Goal: raise ACC from {cur_acc:.4f} to ≥ {tar_acc:.4f} while keeping params ≤ {params_budget:.0f}.\n"
            f"Latency target: ≤ {max(0.0, cur['metrics'].get('latency_ms', 0.0) - abs(direction_vec[1])):.1f} ms (projected).\n"
            f"Anchor: program={anc.program_id} (island={anc.island_id}); features={anc.features}.\n"
            f"Suggested actions: bias towards [{prefer_ops}]; avoid [{avoid_ops}]; "
            f"try sparsify/tile_size or reduce unroll if params risk exceeding budget.\n"
            f"Step size (normalized): {step:.2f}."
        )

    # ---------- 工具 ----------
    def _wrap(self, p: Optional[Dict[str, Any]]) -> Optional[Anchor]:
        if not p: return None
        return Anchor(program_id=p.get("id", "unknown"),
                      island_id=int(p.get("island_id", -1)),
                      metrics=p.get("metrics", {}),
                      resources=p.get("resources", {}),
                      features=p.get("features", {}))

    def _resource_close(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        def ok(key, pct):
            if key not in a or key not in b: return False
            denom = max(1e-9, a[key])
            return abs(a[key] - b[key]) / denom <= pct
        return ( ok("params", self.resource_tol.get("params_pct", 0.1)) and
                 ok("flops",  self.resource_tol.get("flops_pct", 0.2)) and
                 ok("mem_mb", self.resource_tol.get("mem_pct", 0.15)) )

    def _feature_distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        # 统一 key 集合的 L2 距离
        keys = set(a.keys()) | set(b.keys())
        import math
        return math.sqrt(sum((a.get(k,0.0)-b.get(k,0.0))**2 for k in keys))

    def _small_rotate(self, v: List[float], degrees: float = 5.0) -> List[float]:
        # 对三维向量做一个极小旋转（简单起见：绕 z 轴）
        import math
        rad = math.radians(degrees)
        x, y, z = v
        xr = x*math.cos(rad) - y*math.sin(rad)
        yr = x*math.sin(rad) + y*math.cos(rad)
        return [xr, yr, z]
