# openevolve/prompt/sampler.py
# -*- coding: utf-8 -*-
"""
PromptSampler: assemble final prompts (system + user) for code evolution,
with optional Directional Feedback injection controlled by a separate config.
"""

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
    """
    Build prompts for LLM evolution with modular blocks:
    - task/context blocks
    - top programs / inspirations
    - execution artifacts (optional)
    - evaluator improvement hints
    - directional feedback (reads from `direction_cfg`, not PromptConfig)
    """

    def __init__(self, config: PromptConfig, direction_cfg: Optional[object] = None) -> None:
        """
        Args:
            config: PromptConfig
            direction_cfg: DirectionFeedbackConfig (dataclass) or dict, or None
        """
        self.config = config
        self.direction_cfg = direction_cfg
        self.template_manager = TemplateManager(config.template_dir)

        self._system_template_key = "system_message"
        self._user_template_key_diff = "diff_user"
        self._user_template_key_full = "full_rewrite_user"

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def set_templates(
        self,
        system_key: str = "system_message",
        user_key_diff: str = "diff_user",
        user_key_full: str = "full_rewrite_user",
    ) -> None:
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
        # Directional Feedback related (explicit)
        direction_guidance: Optional[str] = None,
        # for auto direction formatting
        parent_program_dict: Optional[Dict[str, Any]] = None,
        # misc passthrough
        **kwargs,
    ) -> Dict[str, str]:
        """
        Return:
            {"system": <system_prompt>, "user": <user_prompt>}
        """
        # 1) fetch templates
        system_template = self.template_manager.get_template(self._system_template_key)
        user_template_key = (
            self._user_template_key_diff if diff_based_evolution else self._user_template_key_full
        )
        user_template = self.template_manager.get_template(user_template_key)

        # 2) prepare contextual strings
        metrics_str = self._render_metrics(program_metrics)
        evolution_history = self._render_history(previous_programs)
        inspiration_block = self._render_inspirations(inspirations)
        top_block = self._render_top_programs(top_programs)

        # 3) optional artifacts
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # 4) variations
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # 5) auto improvement areas
        improvement_areas = self._render_improvement_areas(program_metrics, previous_programs)

        # 6) Directional Feedback
        direction_block = self._build_direction_block(
            evolution_round=evolution_round,
            previous_programs=previous_programs,
            parent_program_dict=parent_program_dict,
            direction_guidance=direction_guidance,
            program_metrics=program_metrics,
        )

        # 7) render user prompt
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

        # 8) system prompt
        system_message = system_template

        # 9) dump final user prompt (optional)
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

    # --------------------------------------------------------------------- #
    # internals – render blocks
    # --------------------------------------------------------------------- #
    def _render_metrics(self, metrics: Dict[str, Any]) -> str:
        if not metrics:
            return ""
        try:
            keys = sorted(metrics.keys())
            parts = []
            for k in keys:
                v = metrics[k]
                if isinstance(v, float):
                    parts.append(f"{k}={v:.6f}")
                else:
                    parts.append(f"{k}={v}")
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
            if cs is not None:
                lines.append(f"- {pid}: combined_score={cs:.6f}")
            else:
                lines.append(f"- {pid}: (no combined_score)")
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
            if cs is not None:
                lines.append(f"- {pid}: combined_score={cs:.6f}")
            else:
                lines.append(f"- {pid}: (no combined_score)")
        return "\n".join(lines)

    def _render_artifacts(self, artifacts: Dict[str, Any]) -> str:
        """
        先给 2–3 行摘要，再给截断后的 Raw。
        """
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

    # --------------------------------------------------------------------- #
    # Improvement areas（自动 1–3 条）
    # --------------------------------------------------------------------- #
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
                tips.append("推理时间上升但 MACs 基本不变：避免将 batch 维作为最内层循环，优先对连续维做 tile+unroll。")

            if (d_macs is not None and d_macs > 0) and (d_cs is not None and d_cs <= 0):
                tips.append("乘法次数增加但精度无提升：尝试低秩/分组/逐点分解或共享中间乘积，减少 MACs。")

            if (d_params is not None and d_params > 0.05 * (params + 1e-9)):
                tips.append("参数增长过快：限制新增层宽度或改为分块累加，避免参数体积膨胀。")

        if not tips:
            tips.append("优先优化速度与乘法：小 tile (16/32) + 低阶 unroll (2/4/8)，内层循环避开 batch 维。")
        return "\n".join(f"- {t}" for t in tips)

    # --------------------------------------------------------------------- #
    # Directional Feedback helpers
    # --------------------------------------------------------------------- #
    def _build_direction_block(
        self,
        *,
        evolution_round: int,
        previous_programs: List[Dict[str, Any]],
        parent_program_dict: Optional[Dict[str, Any]],
        direction_guidance: Optional[str],
        program_metrics: Optional[Dict[str, Any]],
    ) -> str:
        """
        Merge explicit guidance with auto-generated hints from program metadata,
        then inject based on direction_cfg.enabled/frequency.
        """
        explicit_dir = (direction_guidance or "").strip()

        if not parent_program_dict and previous_programs:
            parent_program_dict = previous_programs[-1]

        auto_dir = ""
        try:
            if parent_program_dict:
                auto_dir = self._format_direction_feedback(parent_program_dict).strip()
                if auto_dir:
                    logger.info("DF preview(auto): %s", auto_dir.splitlines()[:3])
        except Exception as _e:
            logger.debug("direction_feedback formatting skipped: %s", _e)

        merged_dir = explicit_dir if explicit_dir else auto_dir

        # read cfg (dataclass or dict → dict)
        cfg_dir: Dict[str, Any] = {}
        if self.direction_cfg is not None:
            if isinstance(self.direction_cfg, dict):
                cfg_dir = self.direction_cfg
            else:
                try:
                    cfg_dir = dc_asdict(self.direction_cfg)
                except Exception:
                    cfg_dir = getattr(self.direction_cfg, "__dict__", {}) or {}

        enabled = bool(cfg_dir.get("enabled", False))
        freq = int(cfg_dir.get("frequency", 1) or 1)
        max_lines = int(cfg_dir.get("max_df_lines", 12))

        logger.info(
            "[DIRFB] enabled=%s freq=%s explicit_len=%d auto_len=%d (sampler=%s)",
            enabled, freq, len(explicit_dir), len(auto_dir), __file__,
        )

        should_inject = bool(enabled and (int(evolution_round) % freq == 0))
        if not should_inject:
            return ""

        # 组装输出：数值/自动块 + 目标 + 允许/禁止清单（强调减少乘法/提速）
        lines: List[str] = []

        # warmup/空文本：也给动作目标
        if merged_dir:
            lines.append(merged_dir.strip())
        lines.append("- objective: 优先减少乘法次数 (MACs/FLOPs) 与推理时延")

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

    def _format_direction_feedback(self, program: Dict[str, Any]) -> str:
        """
        Build a concise, header-less DF body from program['metadata'].
        Do NOT include the '## Directional Guidance' title here (template handles header).
        """
        md = (program or {}).get("metadata", {}) or {}

        slope = md.get("slope_on_baseline", None)
        slope_avg = md.get("slope_mean_k", None)
        stagnating = md.get("stagnating", None)
        warmup = md.get("warmup", False)
        invalid = md.get("invalid", False)

        lines: List[str] = []

        if warmup:
            lines.append("- WARMUP: collecting statistics; real direction will start after warmup_k")
        if invalid:
            lines.append("- INVALID: this round not used for direction update")

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

    # --------------------------------------------------------------------- #
    # template variations
    # --------------------------------------------------------------------- #
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
