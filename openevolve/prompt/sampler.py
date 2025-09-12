# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict as dc_asdict
from typing import Any, Dict, List, Optional

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager

logger = logging.getLogger(__name__)


class PromptSampler:
    def __init__(self, config: PromptConfig, direction_cfg: Optional[object] = None) -> None:
        self.config = config
        self.direction_cfg = direction_cfg
        self.template_manager = TemplateManager(template_dir=config.template_dir)
        self._system_template_key = "system_message"
        self._user_template_key_diff = "diff_user"
        self._user_template_key_full = "full_rewrite_user"
        self._am_get_all_programs = lambda: (self._ctx_prev or []) + (self._ctx_top or [])
        self._am_get_island_frontiers = lambda: {}

    def set_templates(self, system_key="system_message", user_key_diff="diff_user", user_key_full="full_rewrite_user"):
        self._system_template_key = system_key
        self._user_template_key_diff = user_key_diff
        self._user_template_key_full = user_key_full

    def build_prompt(
        self,
        *,
        current_program: str,
        parent_program: str,
        program_metrics: Dict[str, Any],
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
        evolution_round: int,
        diff_based_evolution: bool,
        program_artifacts: Optional[Dict[str, Any]] = None,
        direction_guidance: Optional[str] = None,
        parent_program_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        system_template = self.template_manager.get_template(self._system_template_key)
        user_template_key = (self._user_template_key_diff if diff_based_evolution else self._user_template_key_full)
        user_template = self.template_manager.get_template(user_template_key)

        metrics_str = self._render_metrics(program_metrics)
        evolution_history = self._render_history(previous_programs)
        inspiration_block = self._render_inspirations(inspirations)
        top_block = self._render_top_programs(top_programs)

        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        improvement_areas = self._render_improvement_areas(program_metrics, previous_programs)
        # direction_guidance = direction_guidance or "Tend towards lower FLOPs; Attempt to tile/unroll and constrain the number of parameters"
        direction_guidance = direction_guidance or ""
        # print("direction_guidance:",direction_guidance)
        # print("self.config:",self.config)
        # print(",self.direction_cfg:",self.direction_cfg)
        self._ctx_prev = previous_programs
        self._ctx_top  = top_programs


        direction_block = self._build_direction_block(
            evolution_round=evolution_round,
            previous_programs=previous_programs,
            parent_program_dict=parent_program_dict,
            direction_guidance=direction_guidance,
            program_metrics=program_metrics,
        )
        # print("direction_block:",direction_block)
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            direction_feedback=direction_block,
            inspirations=inspiration_block,
            top_programs=top_block,
            **kwargs,
        )

        system_message = system_template

        if getattr(self.config, "save_prompts_text", False):
            outdir = self.config.prompts_dir or "openevolve_output/prompts"
            try:
                os.makedirs(outdir, exist_ok=True)
                fname = os.path.join(outdir, f"round_{int(evolution_round):05d}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(user_message)
            except Exception as _e:
                logger.debug("Failed to dump prompt text: %s", _e)

        logger.info("[Prompt] round=%s user_template=%s sampler=%s", evolution_round, user_template_key, __file__)
        return {"system": system_message, "user": user_message}

    # ---------- blocks ----------
    def _render_metrics(self, metrics: Dict[str, Any]) -> str:
        if not metrics:
            return ""
        try:
            keys = sorted(metrics.keys())
            parts = []
            for k in keys:
                v = metrics[k]
                parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
            return ", ".join(parts)
        except Exception:
            return json.dumps(metrics, ensure_ascii=False)

    def _render_history(self, prev: List[Dict[str, Any]]) -> str:
        if not prev:
            return ""
        lines = []
        for i, p in enumerate(prev[-5:]):
            pid = p.get("id", f"prog_{i}")
            pm = p.get("metrics", {}) or {}
            cs = pm.get("combined_score", None)
            lines.append(f"- {pid}: combined_score={cs:.6f}" if cs is not None else f"- {pid}: (no combined_score)")
        return "\n".join(lines)

    def _render_inspirations(self, inspirations: List[Dict[str, Any]]) -> str:
        if not inspirations:
            return ""
        keep = inspirations[: self.config.num_diverse_programs]
        lines = []
        for i, p in enumerate(keep):
            pid = p.get("id", f"insp_{i}")
            chg = (p.get("metadata", {}) or {}).get("changes", "N/A")
            lines.append(f"- {pid}: {chg}")
        return "\n".join(lines)

    def _render_top_programs(self, top: List[Dict[str, Any]]) -> str:
        if not top:
            return ""
        keep = top[: self.config.num_top_programs]
        lines = []
        for i, p in enumerate(keep):
            pid = p.get("id", f"top_{i}")
            pm = p.get("metrics", {}) or {}
            cs = pm.get("combined_score", None)
            lines.append(f"- {pid}: combined_score={cs:.6f}" if cs is not None else f"- {pid}: (no combined_score)")
        return "\n".join(lines)

    def _render_artifacts(self, artifacts: Dict[str, Any]) -> str:
        def summarize(text: str) -> List[str]:
            lines = text.splitlines()
            keys = ("slow path", "python loop", "inefficient", "timeout", "OOM", "error", "warning")
            hit = [ln for ln in lines if any(k.lower() in ln.lower() for k in keys)]
            return hit[:3] if hit else lines[:2]

        try:
            blob = json.dumps(artifacts, ensure_ascii=False, indent=2)
        except Exception:
            blob = str(artifacts)

        summary = summarize(blob)
        summary_txt = "\n".join(f"- {s}" for s in summary)

        if self.config.artifact_security_filter:
            blob = blob.replace(self._safe_home(), "~")

        max_bytes = int(self.config.max_artifact_bytes or 0)
        if max_bytes > 0 and len(blob.encode("utf-8")) > max_bytes:
            enc = blob.encode("utf-8")
            blob = enc[-max_bytes:].decode("utf-8", errors="ignore")
            blob = "[TRUNCATED ARTIFACTS]\n" + blob

        return f"Summary:\n{summary_txt}\n\nRaw:\n{blob}"

    def _safe_home(self) -> str:
        try:
            return os.path.expanduser("~")
        except Exception:
            return "/home/user"

    def _render_improvement_areas(self, program_metrics: dict, previous_programs: list) -> str:
        tips = []
        if not program_metrics:
            return ""
        last = None
        if previous_programs:
            last = (previous_programs[-1] or {}).get("metrics", {}) or {}

        cs   = program_metrics.get("combined_score")
        macs = program_metrics.get("macs", program_metrics.get("flops"))
        params = program_metrics.get("params")
        lat_ms = program_metrics.get("latency_ms", program_metrics.get("infer_time_s", 0) * 1000.0)

        if last:
            d_cs   = (cs - last.get("combined_score")) if (cs is not None and "combined_score" in last) else None
            d_macs = (macs - last.get("macs", last.get("flops", 0))) if (macs is not None) else None
            d_lat  = (lat_ms - (last.get("latency_ms", last.get("infer_time_s", 0) * 1000.0)))
            d_params = (params - last.get("params", 0)) if (params is not None) else None

            if (d_lat is not None and d_lat > 0) and (d_macs is not None and abs(d_macs) < 0.02 * (macs + 1e-9)):
                tips.append("The inference time increases but the MACs remain basically unchanged: avoid using batch dimension as the innermost loop, and prioritize performing tile+unroll on continuous dimensions.")

            if (d_macs is not None and d_macs > 0) and (d_cs is not None and d_cs <= 0):
                tips.append("The number of multiplications increases but the accuracy does not improve: try low rank/grouping/pointwise decomposition or sharing intermediate products to reduce MACs.")

            if (d_params is not None and d_params > 0.05 * (params + 1e-9)):
                tips.append("Parameter growth too fast: Limit the width of newly added layers or switch to block accumulation to avoid parameter volume expansion.")

        if not tips:
            tips.append("Prioritize speed and multiplication: small tile (16/32)+low order unroll (2/4/8), inner loop avoids batch dimension.")
        return "\n".join(f"- {t}" for t in tips)

    def _build_direction_block(
        self,
        *,
        evolution_round: int,
        previous_programs: List[Dict[str, Any]],
        parent_program_dict: Optional[Dict[str, Any]],
        direction_guidance: Optional[str],
        program_metrics: Optional[Dict[str, Any]],
    ) -> str:
        explicit_dir = (direction_guidance or "").strip()
        if not parent_program_dict and previous_programs:
            parent_program_dict = previous_programs[-1]

        auto_dir = ""
        try:
            # 用 parent 作为基，但若它缺核心指标（常见于首轮），就退化为“当前指标 stub”
            base_prog = parent_program_dict or {}
            if not self._has_core_metrics(base_prog):
                base_prog = {
                    "id": f"round_{int(evolution_round):05d}_current",
                    "metrics": program_metrics or {},
                    "resources": {
                        "params": (program_metrics or {}).get("params", 0.0),
                        "flops": (program_metrics or {}).get("macs", 0.0),
                        "mem_mb": 0.0,
                    },
                }

            auto_dir = self._format_direction_feedback(
                base_prog,  # ← 传“有 metrics 的那个”
                cur_metrics_override=None  # 已经塞进去了，这里不再 override
            ).strip()

            if auto_dir:
                    logger.info("DF preview(auto): %s", auto_dir.splitlines()[:3])

        except Exception as _e:
            logger.debug("direction_feedback formatting skipped: %s", _e)

        # print("auto_dir:",auto_dir)
        # print("explicit_dir:",explicit_dir)
        merged_dir = auto_dir if auto_dir else explicit_dir
        # print("merged_dir:",merged_dir)
        cfg_dir: Dict[str, Any] = {}
        # print("self.direction_cfg:",self.direction_cfg)
        if self.direction_cfg is not None:
            if isinstance(self.direction_cfg, dict):
                cfg_dir = self.direction_cfg
            else:
                try:
                    cfg_dir = dc_asdict(self.direction_cfg)
                except Exception:
                    cfg_dir = getattr(self.direction_cfg, "__dict__", {}) or {}
        # print("cfg_dir:",cfg_dir)
        enabled = bool(cfg_dir.get("enabled", False))
        # print("enabled:",enabled)
        freq = int(cfg_dir.get("frequency", 1) or 1)
        max_lines = int(cfg_dir.get("max_df_lines", 12))

        logger.info("[DIRFB] enabled=%s freq=%s explicit_len=%d auto_len=%d (sampler=%s)",
                    enabled, freq, len(explicit_dir), len(auto_dir), __file__)
        # print("[DIRFB] enabled=%s freq=%s explicit_len=%d auto_len=%d (sampler=%s)",
        #             enabled, freq, len(explicit_dir), len(auto_dir), __file__)
        should_inject = bool(enabled and (int(evolution_round) % freq == 0))
        # print("enabled:",enabled,evolution_round,freq,should_inject)
        #  enabled: True 1 1 True
        if not should_inject:
            return ""

        lines: List[str] = []
        if merged_dir:
            lines.append(merged_dir.strip())
        # lines.append("-Objective: Prioritize reducing the number of multiplications (MACs/FLOPs) and inference latency")

        allowed_ops = cfg_dir.get("allowed_ops", []) or []
        forbidden = cfg_dir.get("forbidden_patterns", []) or []
        if allowed_ops:
            lines.append("**Allowed knobs/ops:** " + ", ".join(allowed_ops))
        if forbidden:
            lines.append("**Forbidden patterns:** " + ", ".join(forbidden))

        text = "\n".join(lines[:max_lines])
        if not text.startswith("## "):
            text = "## Directional Guidance\n" + text
        return text

    def _format_direction_feedback(self, program: Dict[str, Any],
                                   cur_metrics_override: Optional[Dict[str, Any]] = None) -> str:

        """
        Anchor-aware Directional Feedback:
        1) 选锚点：同资源 > 异源 > 幻想
        2) 计算方向向量与步长
        3) 生成可操作的自然语言 DF
        4) 任何缺失 → 回退到旧的 slope/stagnation 提示
        """
        cfg = self._aa_cfg()
        prev = getattr(self, "_ctx_prev", []) or []
        top = getattr(self, "_ctx_top", []) or []
        hist = prev

        prog = dict(program or {})
        pm = (prog.get("metrics") or {})
        merged = dict(pm)
        merged.update(cur_metrics_override or {})  # ← 无条件合并当前轮指标
        prog["metrics"] = merged

        # 若没 resources，就从 metrics 回填常用资源键
        res = dict(prog.get("resources") or {})
        for k in ("params", "macs", "flops", "mem_mb"):
            if k not in res and k in merged:
                res[k] = merged[k]
        prog["resources"] = res
        program = prog

        self._anchor_manager = None
        self._use_anchor_manager = True
        try:
            import importlib
            AnchorManager = None
            for modname in ("openevolve.direction.anchor_manager",
                            "direction.anchor_manager",
                            "anchor_manager"):
                try:
                    m = importlib.import_module(modname)
                    AnchorManager = getattr(m, "AnchorManager", None)
                    if AnchorManager:
                        break
                except Exception:
                    pass
            if AnchorManager is not None:
                try:
                    self._anchor_manager = AnchorManager(
                        get_all_programs=self._am_get_all_programs,
                        get_island_frontiers=self._am_get_island_frontiers,
                        config=getattr(self, "direction_cfg", None) or {}
                    )
                    logger.info("[DIRFB] AnchorManager enabled via %s", AnchorManager.__module__)
                except Exception as e:
                    logger.info("[DIRFB] AnchorManager init failed (%s); fallback to local anchor logic.", e)
            else:
                logger.info("[DIRFB] AnchorManager module not found; using local anchor logic.")
        except Exception as e:
            logger.info("[DIRFB] AnchorManager not used (%s); fallback to local anchor logic.", e)

        # Anchor object we will use
        anchor = None
        anchor_id = None

        try:
            # 为了兼容不同实现，尝试若干常见参数名
            suggest = None
            am = self._anchor_manager
            # 不同仓库可能方法名不同：suggest(...) 或 select_anchor(...)
            if hasattr(am, "suggest"):
                try:
                    suggest = am.suggest(current=program, previous=prev, top=top)
                except TypeError:
                    # 退而求其次的签名
                    suggest = am.suggest(program, prev, top)
            elif hasattr(am, "select_anchor"):
                suggest = am.select_anchor(program, prev, top)

            if isinstance(suggest, dict):
                # 取出锚点实体与 id
                anchor = suggest.get("anchor") or suggest.get("program") or suggest.get("candidate")
                anchor_id = suggest.get("anchor_id") or suggest.get("id") or \
                            (anchor.get("id") if isinstance(anchor, dict) else None)
            elif isinstance(suggest, (list, tuple)) and suggest:
                anchor = suggest[0]
                anchor_id = getattr(anchor, "id", None) or (anchor.get("id") if isinstance(anchor, dict) else None)

        except Exception as e:
            logger.debug("AnchorManager.suggest failed; fallback to local anchor selector: %s", e)

            # ========== 2) Local fallback (same→hetero→hallucinated) ==========
        try:
            if anchor is None:
                anchor = self._select_anchor(program, hist, prev, top, cfg)
                anchor_id = (anchor.get("id") if isinstance(anchor, dict) else None)
        except Exception as e:
            logger.debug("Local anchor selection failed: %s", e)
        print("anchor:",anchor,anchor_id)
            # anchor: None None
            # ========== 3) Direction + step + render ==========
        try:
            if anchor:
                dv, step, targets = self._direction_and_step(program, anchor, cfg["weights"], hist,
                                                             cfg["stagnation"]["k"])
                text = self._render_anchor_text(
                    program, anchor, dv, step,
                    actions=cfg.get("actions", {}),
                    max_params_pct=cfg.get("actions", {}).get("max_param_increase_pct", 0.05),
                )
                # 可选打印锚点 id
                if cfg.get("print_anchor_id", False) and anchor_id:
                    text = text + f"\n[anchor_id: {anchor_id}]"
                return text
        except Exception as e:
            logger.debug("Direction rendering failed, fallback to legacy DF: %s", e)

            # ========== 4) Legacy fallback ==========
        md = (program or {}).get("metadata", {}) or {}
        slope = md.get("slope_on_baseline", None)
        slope_avg = md.get("slope_mean_k", None)
        stagnating = md.get("stagnating", None)
        warmup = md.get("warmup", False)
        invalid = md.get("invalid", False)

        lines: List[str] = []
        if warmup:  lines.append("- WARMUP: collecting statistics; real direction will start after warmup_k")
        if invalid: lines.append("- INVALID: this round not used for direction update")
        if slope is not None:
            try:
                lines.append(f"- slope_on_island_baseline: {float(slope):.3f}")
            except Exception:
                lines.append(f"- slope_on_island_baseline: {slope}")
        if slope_avg is not None:
            try:
                lines.append(f"- slope_mean_k: {float(slope_avg):.3f}")
            except Exception:
                lines.append(f"- slope_mean_k: {slope_avg}")
        if stagnating is not None:
            lines.append(f"- stagnating: {bool(stagnating)}")

        if stagnating and not warmup and not invalid:
            lines.append("- hint: plateau detected → try smaller-step structural edits; constrain params/FLOPs")
        elif not warmup and not invalid:
            lines.append("- hint: maintain direction; watch resource constraints")

        return "\n".join(lines).strip()

    def _apply_template_variations(self, template: str) -> str:
        try:
            variations = self.config.template_variations or {}
            if "improvement_suggestion" in variations:
                choices = variations["improvement_suggestion"]
                if isinstance(choices, list) and choices:
                    picked = random.choice(choices)
                    template = template.replace("{improvement_suggestion}", picked)
        except Exception as _e:
            logger.debug("Template variation skipped: %s", _e)
        return template


    # ======= Anchor-aware DF: helpers (NEW) =======
    # === Linear-DDO helpers: robust normalization / Pareto / crowding / AI regime ===

    def _obj_vec(self, prog: dict) -> List[float]:
        """2-obj (all 'smaller is better'): [-acc, infer_time_s]"""
        return [
            -float(self._m(prog, "acc", 0.0)),
            float(self._m(prog, "infer_time_s", 0.0)),
        ]


    def _robust_norm_stats(self, pool: List[dict]):
        dim = len(self._obj_vec(pool[0])) if pool else 2
        cols = list(zip(*[self._obj_vec(p) for p in pool])) if pool else [[] for _ in range(dim)]
        stats = []
        for col in cols:
            if not col:
                stats.append((0.0, 1.0, 0.0))  # median, IQR, min
                continue
            xs = sorted(col)
            n = len(xs)
            med = xs[n // 2]
            q1, q3 = xs[n // 4], xs[(3 * n) // 4]
            iqr = max(1e-9, q3 - q1)
            mn = xs[0]
            stats.append((med, iqr, mn))
        return stats


    def _normalize_obj(self, f: List[float], stats) -> List[float]:
        out = []
        for j, val in enumerate(f):
            med, iqr, mn = stats[j]
            z = (val - med) / iqr
            out.append(z - min(0.0, (mn - med) / iqr))  # shift so min≥0
        return out

    def _pareto_frontier(self, pool: List[dict]) -> List[dict]:
        """Simple non-dominated filter on raw f (not normalized)"""
        Fs = [self._obj_vec(p) for p in pool]
        if not Fs:
            return []
        m = len(Fs[0])
        nd = []
        for i, Fi in enumerate(Fs):
            dominated = False
            for j, Fj in enumerate(Fs):
                if i == j:
                    continue
                if all(Fj[k] <= Fi[k] for k in range(m)) and any(Fj[k] < Fi[k] for k in range(m)):
                    dominated = True
                    break
            if not dominated:
                nd.append(pool[i])
        return nd


    def _crowding_distance(self, frontier: List[dict], stats) -> Dict[str, float]:
        """NSGA-II style crowding distance on normalized f-hat"""
        if not frontier: return {}
        Fh = [self._normalize_obj(self._obj_vec(p), stats) for p in frontier]
        m, n = len(Fh[0]), len(frontier)
        dist = [0.0] * n
        for j in range(m):
            idx = sorted(range(n), key=lambda i: Fh[i][j])
            dist[idx[0]] = dist[idx[-1]] = float('inf')
            for t in range(1, n - 1):
                dist[idx[t]] += Fh[idx[t + 1]][j] - Fh[idx[t - 1]][j]
        return {frontier[i].get("id", str(i)): dist[i] for i in range(n)}

    def _ai_proxy(self, prog: dict) -> float:
        """Cheap arithmetic-intensity proxy for Linear: MACs per parameter."""
        macs = float(self._m(prog, "flops", 0.0))
        params = max(1e-9, float(self._m(prog, "params", 0.0)))
        return macs / params

    def _ai_regime(self, prog: dict, cfg: dict) -> str:
        """Classify as bandwidth / compute / balanced via AI proxy against a threshold."""
        ai = self._ai_proxy(prog)
        thr = float(cfg.get("ai_mb_threshold", 32.0))  # configurable, device-dependent
        if ai < 0.75 * thr: return "bandwidth"
        if ai > 1.25 * thr: return "compute"
        return "balanced"

    def _regime_aware_weights(self, base_w: dict, regime: str) -> dict:
        """Reweight latency/params according to regime, keep perf as anchor."""
        w = {"perf": 1.0, "latency": 0.5, "params": 0.3}
        w.update({k: float(v) for k, v in (base_w or {}).items()})
        if regime == "bandwidth":
            w["params"] *= 1.6
            w["latency"] *= 0.9
        elif regime == "compute":
            w["latency"] *= 1.6
            w["params"] *= 0.9
        return w

    def _aa_cfg(self) -> dict:
        """Anchor-aware 配置（含 Linear-DDO 的 AI 阈值），兼容老键"""
        cfg_dir = {}
        if self.direction_cfg is not None:
            if isinstance(self.direction_cfg, dict):
                cfg_dir = self.direction_cfg
            else:
                try:
                    from dataclasses import asdict as _asdict
                    cfg_dir = _asdict(self.direction_cfg)
                except Exception:
                    cfg_dir = getattr(self.direction_cfg, "__dict__", {}) or {}

        actions = cfg_dir.get("actions", {})
        if not actions and (("allowed_ops" in cfg_dir) or ("forbidden_patterns" in cfg_dir)):
            actions = {
                "prefer_ops": cfg_dir.get("allowed_ops", []),
                "avoid_ops": cfg_dir.get("forbidden_patterns", []),
                "max_param_increase_pct": cfg_dir.get("max_param_increase_pct", 0.05),
            }

        return {
            "metric_keys": cfg_dir.get("metric_keys", ["acc", "infer_time_s"]),
            "weights": cfg_dir.get("weights", {"perf": 1.0, "latency": 0.5}),
            "resource_tolerances": cfg_dir.get("resource_tolerances", {"infer_time_pct": 0.10}),
            "stagnation": cfg_dir.get("stagnation", {"k": 5, "slope_eps": 1e-3}),
            "diversify_penalty_cos": cfg_dir.get("diversify_penalty_cos", 0.92),
            "actions": actions,  # 可在 actions 里加 max_time_decrease_pct
            "bootstrap_infer_time_default": cfg_dir.get("bootstrap_infer_time_default", 0.0),
        }


    def _m(self, prog: dict, key: str, default=0.0):
        """安全获取 metrics 的键；支持常见别名"""
        m = (prog or {}).get("metrics", {}) or {}
        alias = {
            "acc": ["acc", "accuracy", "top1", "top1_acc"],
            "infer_time_s": ["infer_time_s", "infer_times_s", "latency_s"],  # 接受你提到的拼写变体
            "latency_ms": ["latency_ms", "infer_time_ms"],  # 仅为兼容旧日志
            "combined_score": ["combined_score"],
        }

        keys = alias.get(key, [key])
        for k in keys:
            v = m.get(k, None)
            if v is None and k.endswith("_s"):
                # 秒 -> 毫秒
                try:
                    v = float(m.get(k, 0.0)) * 1000.0
                except Exception:
                    pass
            if v is not None:
                try: return float(v)
                except Exception: return v
        # 资源类也可能放在 resources 里
        if key in ("params", "flops", "mem_mb", "infer_time_s"):
            r = (prog or {}).get("resources", {}) or {}
            v = r.get(key, None)
            if v is not None:
                try: return float(v)
                except Exception: return v
        return default

    def _r(self, prog: dict, key: str, default=0.0):
        """安全获取 resources 的键"""
        r = (prog or {}).get("resources", {}) or {}
        v = r.get(key, None)
        if v is None and key in ("flops", "macs"):
            v = (prog or {}).get("metrics", {}).get("macs", None)
        return float(v) if isinstance(v, (int, float)) else (v if v is not None else default)

    def _resource_close(self, a: dict, b: dict, tol: dict) -> bool:
        def ok_time(pct):
            av = self._r(a, "infer_time_s", None); bv = self._r(b, "infer_time_s", None)
            if av is None or bv is None:
                return False
            denom = max(1e-9, float(av))
            return abs(float(av) - float(bv)) / denom <= pct
        return ok_time(tol.get("infer_time_pct", 0.10))


    def _pick_same_resource_anchor(self, cur: dict, candidates: list, tol: dict):
        best = None; best_gain = 0.0
        cur_acc = self._m(cur, "acc", 0.0)
        for p in candidates:
            if p is cur:
                continue
            if not self._resource_close(cur, p, tol):
                continue
            gain = self._m(p, "acc", 0.0) - cur_acc
            if gain > best_gain:
                best_gain = gain; best = p
        return best

    def _pick_hetero_anchor(self, cur: dict, candidates: list):
        # 以时间差异 + acc 提升的线性组合
        cur_vec = [self._m(cur, "acc", 0.0), self._m(cur, "infer_time_s", 0.0)]
        best = None; best_score = -1e18
        for p in candidates:
            if p is cur:
                continue
            pv = [self._m(p, "acc", 0.0), self._m(p, "infer_time_s", 0.0)]
            diff = abs(pv[1] - cur_vec[1])  # 时间差异
            score = (pv[0] - cur_vec[0]) + 0.3 * diff
            if score > best_score:
                best_score = score; best = p
        return best


    def _is_stagnating(self, history: list, k: int, eps: float) -> bool:
        """用最近 k 条历史的 acc 变化判断是否平台期"""
        if not history or len(history) < 2:
            return False
        tail = history[-min(k, len(history)):]
        acc0 = self._m(tail[0], "acc", 0.0)
        acc1 = self._m(tail[-1], "acc", 0.0)
        return (acc1 - acc0) < float(eps)

    def _hallucinated_anchor(self, cur: dict, history: list, k: int, max_param_inc_pct: float):
        cur_acc = self._m(cur, "acc", 0.0)
        cur_time = self._m(cur, "infer_time_s", 0.0)
        if cur_time <= 0.0:
            cur_time = self._r(cur, "infer_time_s", 0.0)
        if cur_time <= 0.0:
            cur_time = float(self._aa_cfg().get("bootstrap_infer_time_default", 0.0))

        # 历史不足：给“启动锚点”（acc +2%，时间 -10%）
        if not history or len(history) < 2:
            return {
                "id": "virtual_anchor_bootstrap",
                "metrics": {
                    "acc": max(0.0, cur_acc + 0.02),
                },
                "resources": {
                    "infer_time_s": max(0.0, cur_time * 0.90),
                },
            }

        # 历史充分：用 acc/时间 的单位增益外推
        tail = history[-min(k, len(history)):]
        d_acc = self._m(tail[-1], "acc", 0.0) - self._m(tail[0], "acc", 0.0)
        d_time = self._m(tail[-1], "infer_time_s", 0.0) - self._m(tail[0], "infer_time_s", 0.0)

        def sdiv(a, b):
            return float(a) / (abs(float(b)) + 1e-9)

        acc_target = self._m(cur, "acc", 0.0) + 0.8 * sdiv(d_acc, d_time)
        time_target = max(0.0, cur_time - 0.1 * abs(d_time))
        return {
            "id": "virtual_anchor",
            "metrics": {"acc": acc_target},
            "resources": {"infer_time_s": time_target},
        }

    def _direction_and_step(self, cur: dict, anc: dict, weights: dict, history: list, k: int):
        pool = (history or []) + [cur, anc]
        stats = self._robust_norm_stats(pool)
        f_cur = self._normalize_obj(self._obj_vec(cur), stats)
        f_tar = self._normalize_obj(self._obj_vec(anc), stats)

        m = len(f_cur)
        dv = [f_tar[j] - f_cur[j] for j in range(m)]
        norm = (sum(x * x for x in dv) ** 0.5) or 1.0
        dv = [x / norm for x in dv]

        # 信任域：用 acc/时间 的单位增益
        step = 0.3
        if history and len(history) >= 2:
            tail = history[-min(k, len(history)):]
            dacc = self._m(tail[-1], "acc", 0.0) - self._m(tail[0], "acc", 0.0)
            dt   = self._m(tail[-1], "infer_time_s", 0.0) - self._m(tail[0], "infer_time_s", 0.0)
            g = (dacc / (abs(dt) + 1e-9))
            step = max(0.1, min(1.0, 0.5 * g + 0.3))

        edit_budget = max(1, min(4, int(round(1 + 3 * step))))
        cur_time = self._m(cur, "infer_time_s", 0.0)
        targets = {
            "acc_target":  max(0.0, self._m(cur, "acc", 0.0) - dv[0] * 0.02),    # dv[0] 对应 -acc
            "time_target": max(0.0, cur_time + dv[1] * 0.10 * max(1e-9, cur_time)),
            "edit_budget": edit_budget,
        }
        return dv, step, targets

    def _render_anchor_text(self, cur: dict, anchor: dict, dv: list, step: float, actions: dict, max_params_pct: float):
        cur_acc = self._m(cur, "acc", 0.0)
        tar_acc = self._m(anchor, "acc", cur_acc)

        cur_t  = self._m(cur, "infer_time_s", 0.0)
        anch_t = self._r(anchor, "infer_time_s", None)
        if anch_t is None:
            anch_t = self._m(anchor, "infer_time_s", cur_t)
        max_time_dec = float(self._aa_cfg().get("actions", {}).get("max_time_decrease_pct", 0.10))
        budget = max(0.0, min(cur_t * (1.0 - max_time_dec), anch_t))

        prefer_ops = ", ".join(actions.get("prefer_ops", [])) if actions else ""
        avoid_ops  = ", ".join(actions.get("avoid_ops", [])) if actions else ""

        lines = [
            "## Directional Guidance",
            f"Aim: Acc↑, Time↓.",
            f"Target: acc {cur_acc:.4f} → ≥ {tar_acc:.4f}; infer_time_s ≤ {budget:.6f}.",
            f"Trust-region step: {step:.2f} → edit budget: {max(1, min(4, int(round(1 + 3 * step))))}.",
        ]
        if prefer_ops: lines.append("**Preferred ops/knobs:** " + prefer_ops)
        if avoid_ops:  lines.append("**Avoid patterns:** " + avoid_ops)
        return "\n".join(lines)


    def _has_core_metrics(self, prog: dict) -> bool:
        m = (prog or {}).get("metrics", {}) or {}
        if any(k in m for k in ("acc", "accuracy", "top1", "infer_time_s")):
            return True
        r = (prog or {}).get("resources", {}) or {}
        return "infer_time_s" in r

    def _select_anchor(self, cur: dict, history: list, prev: list, top: list, cfg: dict):
        # 0) 组候选池（带核心指标）；没有就把当前点放进去
        pool = [p for p in ((top or []) + (prev or [])) if p and self._has_core_metrics(p)]
        if not pool and self._has_core_metrics(cur):
            pool = [cur]

        # 兜底：候选为空 → 直接幻想锚点
        if not pool:
            return self._hallucinated_anchor(
                cur, history, cfg["stagnation"]["k"],
                cfg["actions"].get("max_param_increase_pct", 0.05)
            )

        # 1) 同资源优选（更稳的 exploitation）
        tol = cfg.get("resource_tolerances", {"params_pct": 0.10, "flops_pct": 0.20, "mem_pct": 0.15})
        same_res = [p for p in pool if self._resource_close(cur, p, tol)]
        if same_res:
            cand = self._pick_same_resource_anchor(cur, same_res, tol)
            if cand is not None:
                return cand

        # 2) 平台期 or 无同资源候选 → 走异质探索（diversity）
        stag_cfg = cfg.get("stagnation", {"k": 5, "slope_eps": 1e-3})
        if self._is_stagnating(history, int(stag_cfg.get("k", 5)), float(stag_cfg.get("slope_eps", 1e-3))):
            cand = self._pick_hetero_anchor(cur, pool)
            if cand is not None:
                return cand

        # 3) 正常情况：Pareto + 拥挤度 + AI-regime（你现有逻辑）
        frontier = self._pareto_frontier(pool)
        stats = self._robust_norm_stats(pool + [cur])
        Fh = [self._normalize_obj(self._obj_vec(p), stats) for p in frontier]
        Z = list(map(min, zip(*Fh))) if Fh else [0.0, 0.0]

        base_w = cfg.get("weights", {"perf": 1.0, "latency": 0.5})
        W = {"perf": float(base_w.get("perf", 1.0)), "latency": float(base_w.get("latency", 0.5))}
        crowd = self._crowding_distance(frontier, stats)

        best, bestU = None, -1e18
        for p in frontier:
            pid = p.get("id", "p")
            fhat = self._normalize_obj(self._obj_vec(p), stats)
            # 2-D Tchebychev：perf(=acc) & latency(=time)
            tche = max(W["perf"] * abs(fhat[0] - Z[0]),
                       W["latency"] * abs(fhat[1] - Z[1]))
            U = -tche + 0.2 * crowd.get(pid, 0.0)
            if U > bestU:
                bestU, best = U, p

        # 4) 仍无 → 幻想锚点（永不返回 None）
        return best or self._hallucinated_anchor(
            cur, history, cfg["stagnation"]["k"],
            cfg["actions"].get("max_param_increase_pct", 0.05)
        )
