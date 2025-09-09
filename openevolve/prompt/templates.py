# -*- coding: utf-8 -*-
from typing import Dict


class TemplateManager:
    def __init__(self, template_dir=None):
        self.template_dir = template_dir

    def get_template(self, key: str) -> str:
        return TEMPLATES[key]


TEMPLATES: Dict[str, str] = {
    "system_message": (
        "You are an expert coder helping to improve programs through evolution."
    ),

    "evaluator_system_message": (
        "You are an expert code reviewer."
    ),

    # ---------- full rewrite user ----------
    "full_rewrite_user": (
        "We are evolving a program in {language}.\n\n"
        "Current performance metrics:\n"
        "{metrics}\n\n"
        "Areas identified for improvement:\n"
        "{improvement_areas}\n\n"
        "Evolution history (top programs):\n"
        "{evolution_history}\n\n"
        "Directional Feedback:\n"
        "{direction_feedback}\n\n"
        "If available, recent execution artifacts:\n"
        "{artifacts}\n\n"
        "Inspirations:\n"
        "{inspirations}\n\n"
        "Top programs summary:\n"
        "{top_programs}\n\n"
        "You MUST output the entire improved program in {language}.\n"
        "Be concise but correct. Keep modifications minimal yet impactful.\n"
        "\n"
        "Hard constraints (must follow):\n"
        "- DO NOT use torch.dot / mm / mv / matmul / @ / einsum / bmm.\n"
        "- DO NOT write through a view (e.g., `out_b = out[b]; out_b[j] = ...`). Always assign with `out[b, j] = ...`.\n"
        "- Keep batch dimension OUT of the innermost loop.\n"
        "- Prefer elementwise multiply + sum: `(a * b).sum()`; use tile/unroll only on non-batch dims.\n"
    ),

    # ---------- diff-based user ----------
    "diff_user": (
        "We are evolving a program in {language} via patch diffs (SEARCH/REPLACE).\n\n"
        "Current performance metrics:\n"
        "{metrics}\n\n"
        "Areas identified for improvement:\n"
        "{improvement_areas}\n\n"
        "Evolution history (top programs):\n"
        "{evolution_history}\n\n"
        "Directional Feedback:\n"
        "{direction_feedback}\n\n"
        "If available, recent execution artifacts:\n"
        "{artifacts}\n\n"
        "Inspirations:\n"
        "{inspirations}\n\n"
        "Top programs summary:\n"
        "{top_programs}\n\n"
        "You MUST use the exact SEARCH/REPLACE diff format. Keep edits minimal and focused.\n"
        "\n"
        "Hard constraints (must follow):\n"
        "- DO NOT use torch.dot / mm / mv / matmul / @ / einsum / bmm.\n"
        "- DO NOT write through a view (e.g., `out_b = out[b]; out_b[j] = ...`). Always assign with `out[b, j] = ...`.\n"
        "- Keep batch dimension OUT of the innermost loop.\n"
        "- Prefer elementwise multiply + sum: `(a * b).sum()`; use tile/unroll only on non-batch dims.\n"
    ),
}
