# primitives/linear_addmv.py
import torch
import torch.nn as nn

class LinearAddmv(nn.Module):
    """用 addmv 调 BLAS GEMV，仍不使用 nn.Linear。"""
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_f, in_f))
        self.b = nn.Parameter(torch.randn(out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0); x = x.view(B, -1)
        out = x.new_empty(B, self.W.size(0))
        for i in range(B):
            out[i] = torch.addmv(self.b, self.W, x[i], beta=1.0, alpha=1.0)
        return out
