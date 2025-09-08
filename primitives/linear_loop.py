# primitives/linear_loop.py
import torch
import torch.nn as nn

class LinearLoop(nn.Module):
    """完全不用 nn.Linear；双重 for，便于作为低效基线被 agent 优化。"""
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f))
        self.b = nn.Parameter(torch.randn(out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0); x = x.view(B, -1)
        out = x.new_zeros(B, self.W.size(0))
        for i in range(B):
            for j in range(self.W.size(0)):
                out[i, j] = torch.dot(x[i], self.W[j]) + self.b[j]
        return out
