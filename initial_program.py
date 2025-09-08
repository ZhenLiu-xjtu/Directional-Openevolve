# initial_program.py
import torch
import torch.nn as nn
from primitives.linear_loop import LinearLoop  # 起步使用低效核，便于改进空间

ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

class Model(nn.Module):
    """
    最小可训练模型：3072 -> H -> 10，两层 MLP。
    结构/激活/隐藏维度可被 agent 直接编辑本文件来搜索。
    """
    def __init__(self, hidden: int = 256, act: str = "relu"):
        super().__init__()
        Act = ACT_MAP.get(act, nn.ReLU)
        self.fc1 = LinearLoop(3072, hidden)
        self.act = Act()
        self.fc2 = LinearLoop(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)
