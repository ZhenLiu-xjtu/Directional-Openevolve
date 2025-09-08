# initial_program.py — minimal baseline for evolution
from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn as nn


IN_DIM = 3 * 32 * 32
NUM_CLASSES = 10

class LinearLoopLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = (self.in_features) ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32] or [B, in]
        if x.dim() == 4:
            B = x.size(0)
            x = x.reshape(B, -1)
        else:
            B = x.size(0)

        out = x.new_zeros(B, self.out_features)

        # (no matmul/einsum/@/dot/mm/mv)
        for b in range(B):
            for j in range(self.out_features):
                acc = 0.0
                for i in range(self.in_features):
                    acc += x[b, i] * self.weight[j, i]
                if self.bias is not None:
                    acc += self.bias[j]
                out[b, j] = acc
        return out

class CandidateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = LinearLoopLayer(IN_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def build_model() -> Tuple[nn.Module, Dict]:
    model = CandidateNet()
    meta = {
        "hyperparams": {
            "in_dim": IN_DIM,
            "num_classes": NUM_CLASSES,
            "hidden_dim": 0,  # 0 single layer
        },
        "notes": "Minimal loop-based linear; no nn.Linear/matmul/einsum/dot."
    }
    return model, meta
