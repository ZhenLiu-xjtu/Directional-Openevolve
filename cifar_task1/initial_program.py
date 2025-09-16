# initial_program.py — evolvable linear with low-rank / groups / sparsity (no mm/matmul/einsum/dot)
from __future__ import annotations
from typing import Tuple, Dict
import math
import torch
import torch.nn as nn

IN_DIM = 3 * 32 * 32
NUM_CLASSES = 10

class LinearLoopLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, lowrank_rank: int = 0,
                 groups: int = 1, sparsity: float = 1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(lowrank_rank)
        self.groups = max(1, int(groups))
        self.sparsity = float(sparsity)

        # Parameters:
        if self.rank > 0:
            # Low-rank factors: U[r, in], V[out, r]
            self.U = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.V = nn.Parameter(torch.empty(self.out_features, self.rank))
            nn.init.kaiming_uniform_(self.U, a=5 ** 0.5)
            nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)
            with torch.no_grad():
                self.U.mul_(1.0 / math.sqrt(self.in_features))
                self.V.mul_(1.0 / math.sqrt(max(1, self.rank)))
            self.weight = None
        else:
            # Full matrix (used by groups/sparsity or plain baseline)
            self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            self.U = None
            self.V = None

        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
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

        # Deterministic sparsity stride (avoid randomness in eval)
        rho = max(0.0, min(1.0, self.sparsity))
        keep_stride = 1 if rho >= 0.999 else max(1, int(round(1.0 / max(1e-6, rho))))

        if self.rank > 0:
            # Low-rank: for each sample, compute t = U @ x (elementwise-mul+sum), then y = V @ t
            for b in range(B):
                xb = x[b]  # [in_features]
                # t[k] = sum_i U[k,i] * xb[i]
                t = xb.new_zeros(self.rank)
                for k in range(self.rank):
                    # Optional sparsity on input side
                    if keep_stride == 1:
                        t[k] = (xb * self.U[k]).sum()
                    else:
                        s = xb.new_tensor(0.0)
                        # strided deterministic subsampling to approximate rho
                        for i in range(0, self.in_features, keep_stride):
                            s = s + self.U[k, i] * xb[i]
                        t[k] = s
                # out[b, j] = sum_k V[j,k] * t[k]
                for j in range(self.out_features):
                    s = (t * self.V[j]).sum()
                    if self.bias is not None:
                        s = s + self.bias[j]
                    out[b, j] = s
            return out

        # Else: full matrix (optionally with groups and/or sparsity)
        if self.groups > 1:
            # Each output j only connects to its assigned input slice
            step = (self.in_features + self.groups - 1) // self.groups  # ceil division
            for b in range(B):
                xb = x[b]
                for j in range(self.out_features):
                    g = j % self.groups
                    start = g * step
                    end = min(self.in_features, start + step)
                    if keep_stride == 1:
                        s = (xb[start:end] * self.weight[j, start:end]).sum()
                    else:
                        s = xb.new_tensor(0.0)
                        for i in range(start, end, keep_stride):
                            s = s + self.weight[j, i] * xb[i]
                    if self.bias is not None:
                        s = s + self.bias[j]
                    out[b, j] = s
            return out

        # Plain baseline with optional sparsity
        for b in range(B):
            xb = x[b]
            for j in range(self.out_features):
                if keep_stride == 1:
                    s = (xb * self.weight[j]).sum()
                else:
                    s = xb.new_tensor(0.0)
                    for i in range(0, self.in_features, keep_stride):
                        s = s + self.weight[j, i] * xb[i]
                if self.bias is not None:
                    s = s + self.bias[j]
                out[b, j] = s
        return out


class CandidateNet(nn.Module):
    def __init__(self, hidden_dim: int = 0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim > 0:
            # 两层：in -> H -> out，仍使用你的 for-loop 线性层
            self.fc1 = LinearLoopLayer(IN_DIM, self.hidden_dim, bias=True,
                                       lowrank_rank=0, groups=1, sparsity=1.0)
            self.act = nn.ReLU(inplace=True)
            self.fc2 = LinearLoopLayer(self.hidden_dim, NUM_CLASSES, bias=True,
                                       lowrank_rank=0, groups=1, sparsity=1.0)
        else:
            # 单层基线
            self.fc = LinearLoopLayer(IN_DIM, NUM_CLASSES, bias=True,
                                      lowrank_rank=0, groups=1, sparsity=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            b = x.size(0)
            x = x.view(b, -1)
        if self.hidden_dim > 0:
            return self.fc2(self.act(self.fc1(x)))
        else:
            return self.fc(x)



from typing import Tuple, Dict

def build_model() -> Tuple[nn.Module, Dict]:
    # 默认 H=0；进化/提示可调整为 64/128 等
    model = CandidateNet(hidden_dim=0)

    # 从模型对象读取参数，自动写回 meta，保证评测方与实现一致
    hp = {}
    hp["in_dim"] = IN_DIM
    hp["num_classes"] = NUM_CLASSES
    hp["hidden_dim"] = int(getattr(model, "hidden_dim", 0) or 0)

    # 读取 for-loop 线性层上的超参（如果是两层，取第一层；按需也可汇总）
    if hp["hidden_dim"] > 0 and hasattr(model, "fc1"):
        fc_like = model.fc1
    else:
        fc_like = getattr(model, "fc", None)

    if fc_like is not None:
        hp["lowrank_rank"] = int(getattr(fc_like, "rank", 0) or 0)
        hp["groups"] = int(getattr(fc_like, "groups", 1) or 1)
        hp["sparsity"] = float(getattr(fc_like, "sparsity", 1.0))
    else:
        hp["lowrank_rank"] = 0
        hp["groups"] = 1
        hp["sparsity"] = 1.0

    meta = {"hyperparams": hp}
    return model, meta
