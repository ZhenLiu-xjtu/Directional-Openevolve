"""
Prompt templates for OpenEvolve
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for evolution
BASE_SYSTEM_TEMPLATE = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
"""

BASE_EVALUATOR_SYSTEM_TEMPLATE = """You are an expert code reviewer.
Your job is to analyze the provided code and evaluate it systematically."""

# User message template for diff-based evolution
DIFF_USER_TEMPLATE = """# 

- Do NOT introduce new Python identifiers that are not present in the current program context
  (e.g., keep variable names like `x`, `labels`, etc. unchanged). In particular, do not invent
  variables such as `xb` if they do not already exist.
- Do NOT change function/class signatures (keep `forward(self, x)` and return types the same).
- Do NOT use forbidden APIs: `nn.Linear`, `torch.matmul`, `torch.mm`, `einsum`, the `@` operator,
  `F.linear`, or any high-level linear layers. If you need matrix multiplication, implement it
  explicitly with nested for-loops only.
- Keep tensor device/dtype unchanged. Do not move tensors across devices.
- Keep label ranges valid (e.g., 0..num_classes-1). Do not create out-of-range indices.
- Only modify code inside the patch region; do NOT rewrite unrelated parts of the file.

Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Directional Feedback
{direction_feedback}



# Current Program
```{language}
{current_program}
```

# Task
Suggest improvements to the program that will lead to better performance on the specified metrics.
[Accounting rule for MACs]
- The evaluator estimates MACs from declared hyperparameters in `build_model` metadata. Pure loop reordering/tiling/unrolling WILL NOT reduce MACs; they only affect latency.
- To reduce MACs, prefer structure changes that can be expressed via hyperparameters (e.g., low-rank factorization rank `r`, group count `g`, sparsity ratio).
- Whenever you apply a structural change, also update the returned metadata hyperparameters accordingly so the MACs estimator captures it.

[Objective shaping]
- Primary: increase Top-1 accuracy.
- Secondary (strong): reduce MACs as computed by the evaluator; then reduce latency and parameters.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
- Do NOT introduce new Python identifiers that are not present in the current program context
  (e.g., keep variable names like `x`, `labels`, etc. unchanged). In particular, do not invent
  variables such as `xb` if they do not already exist.
- Do NOT change function/class signatures (keep `forward(self, x)` and return types the same).
=======
- Keep the external I/O contract identical (input/output tensor shapes and value ranges unchanged).
- You MAY introduce new local variables, module attributes (e.g., extra parameters/buffers), and small helper functions inside this file if needed for structure changes (e.g., low-rank factors, group partitions, sparsity masks).
- Do NOT change public entry points: keep `forward(self, x)` and the returned tensor shape the same.
>>>>>>> REPLACE


Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Program Evolution History
{evolution_history}

# Directional Feedback
{direction_feedback}

# Current Program
```{language}
{current_program}
```

# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.
[Invariants — MUST KEEP]
- Export a function: build_model() -> (nn.Module, meta: dict)
- Keep forward(self, x) input/output SHAPES and ranges unchanged.
- Return meta["hyperparams"] with keys:
  in_dim, num_classes, hidden_dim (0 if single stage),
  lowrank_rank, groups, sparsity
- Do NOT change dataset/training/eval pipeline.

[You MAY rewrite freely]
- You may add/remove local variables, module attributes (parameters/buffers), and helpers.
- You may switch structure: low-rank factorization, group linear, sparsify, 2-layer MLP, or others.
- You MAY use matmul/mm/einsum/F.linear if available (runtime switch controls this).

[Scoring / Accounting RULES (very important)]
- MACs are computed ONLY from meta["hyperparams"]; loop tricks or tiling do NOT reduce MACs.
- If you use dense ops but do NOT declare lowrank_rank/groups/sparsity, evaluator will count dense MACs.
- Primary objective: Top-1 ↑. Secondary (strong): MACs ↓ via the hyperparams above; then latency/params ↓.

[Deliverable]
- Provide complete rewritten code for this file. Keep build_model and meta consistent.

Tip: Reducing MACs requires changing lowrank_rank/groups/sparsity in meta["hyperparams"].
If you keep them as (0,1,1.0), evaluator will count dense MACs and you gain no MACs bonus.
[How to output your patch]
Return one or more SEARCH/REPLACE blocks. Each SEARCH MUST exactly match CURRENT source code text.
Example:
<<<<<<< SEARCH
for j in range(self.out_features):
    s = (xb * self.weight[j]).sum()
    if self.bias is not None:
        s = s + self.bias[j]
    out[b, j] = s
=======
# group-sliced accumulation (g groups); write hyperparams in build_model()
g = max(1, self.groups)
step = (self.in_features + g - 1) // g
gid = j % g
start = gid * step
end = min(self.in_features, start + step)
s = (xb[start:end] * self.weight[j, start:end]).sum()
if self.bias is not None:
    s = s + self.bias[j]
out[b, j] = s
>>>>>>> REPLACE
IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```
"""

# Template for formatting evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Attempts

{previous_attempts}

## Top Performing Programs

{top_programs}

{inspirations_section}
"""

# Template for formatting a previous attempt
PREVIOUS_ATTEMPT_TEMPLATE = """### Attempt {attempt_number}
- Changes: {changes}
- Performance: {performance}
- Outcome: {outcome}
"""

# Template for formatting a top program
TOP_PROGRAM_TEMPLATE = """### Program {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key features: {key_features}
"""

# Template for formatting inspirations section
INSPIRATIONS_SECTION_TEMPLATE = """## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

# Template for formatting an individual inspiration program
INSPIRATION_PROGRAM_TEMPLATE = """### Inspiration {program_number} (Score: {score}, Type: {program_type})
```{language}
{program_snippet}
```
Unique approach: {unique_features}
"""

# Template for evaluating a program via an LLM
EVALUATION_TEMPLATE = """Evaluate the following code on a scale of 0.0 to 1.0 for the following metrics:
1. Readability: How easy is the code to read and understand?
2. Maintainability: How easy would the code be to maintain and modify?
3. Efficiency: How efficient is the code in terms of time and space complexity?

For each metric, provide a score between 0.0 and 1.0, where 1.0 is best.

Code to evaluate:
```python
{current_program}
```

Return your evaluation as a JSON object with the following format:
{{
    "readability": [score],
    "maintainability": [score],
    "efficiency": [score],
    "reasoning": "[brief explanation of scores]"
}}
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "evaluator_system_message": BASE_EVALUATOR_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
    "inspirations_section": INSPIRATIONS_SECTION_TEMPLATE,
    "inspiration_program": INSPIRATION_PROGRAM_TEMPLATE,
    "evaluation": EVALUATION_TEMPLATE,
}


class TemplateManager:
    """Manages templates for prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
