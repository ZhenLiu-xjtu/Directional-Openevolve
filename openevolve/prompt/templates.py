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

    # —— full rewrite 模板保持不变（如你需要）——

    # —— diff-based 用户模板（强化为“只回补丁” + 示例）——
    "diff_user": (
        "We are evolving a program in {language} via patch diffs.\n\n"
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
        "OUTPUT FORMAT (MANDATORY):\n"
        "Return ONLY one or more PATCH blocks. NO prose, NO markdown code fences.\n"
        "Each patch uses the exact SEARCH/REPLACE syntax shown below.\n"
        "Do NOT modify unrelated lines. Keep edits minimal and focused.\n\n"
        "Example (follow exactly):\n"
        "PATCH:\n"
        "SEARCH:\n"
        "out_b = out[b]\\nout_b[j] = acc\n"
        "REPLACE:\n"
        "out[b, j] = acc\n"
        "ENDPATCH\n\n"
        "Another example:\n"
        "PATCH:\n"
        "SEARCH:\n"
        "acc = torch.dot(xb, self.weight[j])\n"
        "REPLACE:\n"
        "acc = (xb * self.weight[j]).sum()\n"
        "ENDPATCH\n\n"
        "Hard constraints (must follow):\n"
        "- DO NOT use torch.dot / mm / mv / matmul / @ / einsum / bmm / addmm / addmv.\n"
        "- DO NOT write through a view (e.g., `out_b = out[b]; out_b[j] = ...`). Always assign with `out[b, j] = ...`.\n"
        "- Keep batch dimension OUT of the innermost loop.\n"
        "- Prefer elementwise multiply + sum: `(a * b).sum()`; use tile/unroll only on non-batch dims.\n"
    ),
}
