# primitives/linear_matmul.py
import torch
import torch.nn as nn

class LinearMatmul(nn.Module):
    """完全矢量化实现（仍非 nn.Linear）。"""
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f))
        self.b = nn.Parameter(torch.randn(out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0); x = x.view(B, -1)
        return x.matmul(self.W.t()) + self.b
