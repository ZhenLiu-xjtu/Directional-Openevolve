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
        self.template_manager = TemplateManager(config.template_dir)
        self._system_template_key = "system_message"
        self._user_template_key_diff = "diff_user"
        self._user_template_key_full = "full_rewrite_user"

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

        direction_block = self._build_direction_block(
            evolution_round=evolution_round,
            previous_programs=previous_programs,
            parent_program_dict=parent_program_dict,
            direction_guidance=direction_guidance,
            program_metrics=program_metrics,
        )

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
        return "\n".oin(lines)

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
            if parent_program_dict:
                auto_dir = self._format_direction_feedback(parent_program_dict).strip()
                if auto_dir:
                    logger.info("DF preview(auto): %s", auto_dir.splitlines()[:3])
        except Exception as _e:
            logger.debug("direction_feedback formatting skipped: %s", _e)

        merged_dir = explicit_dir if explicit_dir else auto_dir

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

        logger.info("[DIRFB] enabled=%s freq=%s explicit_len=%d auto_len=%d (sampler=%s)",
                    enabled, freq, len(explicit_dir), len(auto_dir), __file__)

        should_inject = bool(enabled and (int(evolution_round) % freq == 0))
        if not should_inject:
            return ""

        lines: List[str] = []
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
            try: lines.append(f"- slope_on_island_baseline: {float(slope):.3f}")
            except Exception: lines.append(f"- slope_on_island_baseline: {slope}")
        if slope_avg is not None:
            try: lines.append(f"- slope_mean_k: {float(slope_avg):.3f}")
            except Exception: lines.append(f"- slope_mean_k: {slope_avg}")
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
