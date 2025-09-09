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
    - error feedback (if any, from caller)
    - directional feedback (this class reads from `direction_cfg`, not PromptConfig)
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

        # which system/user template keys to use (can be overridden by set_templates)
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

        # 4) template stochasticity (variations) — only affects wording, not which blocks appear
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # 5) Directional Feedback: merge explicit + auto, then gate by direction_cfg
        direction_block = self._build_direction_block(
            evolution_round=evolution_round,
            previous_programs=previous_programs,
            parent_program_dict=parent_program_dict,
            direction_guidance=direction_guidance,
        )

        # 6) render user prompt
        # NOTE: keep placeholders conservative; these are present in your templates.py
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas="",  # reserved; caller can pass in via kwargs if needed
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            direction_feedback=direction_block,
            inspirations=inspiration_block,
            top_programs=top_block,
            **kwargs,
        )

        # 7) system prompt
        system_message = system_template

        # 8) optional: dump final user prompt (plain text) for debugging/inspection
        if getattr(self.config, "save_prompts_text", False):
            outdir = self.config.prompts_dir or "openevolve_output/prompts"
            try:
                os.makedirs(outdir, exist_ok=True)
                fname = os.path.join(outdir, f"round_{int(evolution_round):05d}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(user_message)
            except Exception as _e:
                logger.debug("Failed to dump prompt text: %s", _e)

        # 9) log a compact preview (avoids flooding)
        logger.info(
            "[Prompt] round=%s user_template=%s sampler=%s",
            evolution_round,
            user_template_key,
            __file__,
        )

        return {"system": system_message, "user": user_message}

    # --------------------------------------------------------------------- #
    # internals – render blocks
    # --------------------------------------------------------------------- #
    def _render_metrics(self, metrics: Dict[str, Any]) -> str:
        if not metrics:
            return ""
        try:
            # deterministic ordering for readability
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
            # Safe fallback
            return json.dumps(metrics, ensure_ascii=False)

    def _render_history(self, prev: List[Dict[str, Any]]) -> str:
        if not prev:
            return ""
        lines = []
        # only take a few recent to avoid bloating prompt
        for i, p in enumerate(prev[-5:]):
            pid = p.get("id", f"prog_{i}")
            pm = p.get("metrics", {}) or {}
            cs = pm.get("combined_score", None)
            if cs is not None:
                lines.append(f"- {pid}: combined_score={cs:.6f}")
            else:
                # no score => make visible to LLM that it's missing
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
        Render execution artifacts with a size cap and a simple security filter (optional).
        """
        try:
            blob = json.dumps(artifacts, ensure_ascii=False)
        except Exception:
            blob = str(artifacts)

        if self.config.artifact_security_filter:
            # minimal redaction (paths, tokens, etc.) – keep it simple & safe
            blob = blob.replace(self._safe_home(), "~")

        max_bytes = int(self.config.max_artifact_bytes or 0)
        if max_bytes > 0 and len(blob.encode("utf-8")) > max_bytes:
            # trim from the head to keep recent errors/outputs
            enc = blob.encode("utf-8")
            blob = enc[-max_bytes:].decode("utf-8", errors="ignore")
            blob = "[TRUNCATED ARTIFACTS]\n" + blob

        return blob

    def _safe_home(self) -> str:
        try:
            return os.path.expanduser("~")
        except Exception:
            return "/home/user"

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
    ) -> str:
        """
        Merge explicit guidance with auto-generated hints from program metadata,
        then inject based on direction_cfg.enabled/frequency.
        """
        # 1) collect candidate text
        explicit_dir = (direction_guidance or "").strip()

        # if parent not provided, take the last of previous as a fallback
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

        # 2) read direction config (dataclass or dict → dict)
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

        logger.info(
            "[DIRFB] enabled=%s freq=%s explicit_len=%d auto_len=%d (sampler=%s)",
            enabled,
            freq,
            len(explicit_dir),
            len(auto_dir),
            __file__,
        )

        # 3) gating
        should_inject = bool(enabled and merged_dir and (int(evolution_round) % freq == 0))
        if not should_inject:
            return ""

        # Ensure block has a header – templates expect just {direction_feedback}
        if not merged_dir.startswith("## "):
            return "## Directional Guidance\n" + merged_dir
        return merged_dir

    def _format_direction_feedback(self, program: Dict[str, Any]) -> str:
        """
        Build a concise, header-less DF body from program['metadata'].
        Do NOT include the '## Directional Guidance' title here (template handles header).
        """
        md = (program or {}).get("metadata", {}) or {}

        slope = md.get("slope_on_baseline", None)
        slope_avg = md.get("slope_mean_k", None)
        stagnating = md.get("stagnating", None)

        lines: List[str] = []
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

        # lightweight action hint
        if stagnating:
            lines.append("- hint: plateau detected → try smaller-step structural edits; constrain params/FLOPs")
        else:
            lines.append("- hint: maintain direction; watch resource constraints")

        return "\n".join(lines).strip()

    # --------------------------------------------------------------------- #
    # template variations
    # --------------------------------------------------------------------- #
    def _apply_template_variations(self, template: str) -> str:
        """
        Apply small word-level variations to reduce overfitting to a single phrasing.
        """
        try:
            variations = self.config.template_variations or {}
            # only one key right now ("improvement_suggestion"), easy to extend later
            if "improvement_suggestion" in variations:
                choices = variations["improvement_suggestion"]
                if isinstance(choices, list) and choices:
                    picked = random.choice(choices)
                    template = template.replace("{improvement_suggestion}", picked)
        except Exception as _e:
            logger.debug("Template variation skipped: %s", _e)
        return template
