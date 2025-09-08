# initial_program.py
import torch, torch.nn as nn

class EvoLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ========= EVOLVE-BLOCK-START =========
        # 目标：在保持输入输出维度一致的前提下，探索更高效/更稀疏/更低秩的结构。
        # 你可以在这里引入：
        #   - 稠密全连接（baseline）
        #   - 低秩分解（W = A @ B, rank=r）
        #   - 分组/块对角稀疏（grouped)
        #   - 输入特征子集选择/掩码
        #   - 是否使用 bias / 归一化 / dropout 等轻量组件
        #
        # 初始实现：三选一（LLM 可以改写此逻辑、参数、甚至加入新模式）
        self.mode = "dense"  # 可进化：["dense", "lowrank", "grouped"]

        if self.mode == "dense":
            self.impl = nn.Linear(in_features, out_features, bias=True)

        elif self.mode == "lowrank":
            # 低秩：W = A (out x r) @ B (r x in)
            r = 64  # 可进化：{16,32,64,128,...}
            self.A = nn.Parameter(torch.randn(out_features, r) * 0.02)
            self.B = nn.Parameter(torch.randn(r, in_features) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_features))

        elif self.mode == "grouped":
            # 分组稀疏：把输入分成 g 组，每组各自线性映射后拼接/求和
            g = 4  # 可进化：{2,4,8,16}
            assert in_features % g == 0
            self.group_in = in_features // g
            self.group_layers = nn.ModuleList(
                [nn.Linear(self.group_in, out_features, bias=False) for _ in range(g)]
            )
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        # ========= EVOLVE-BLOCK-END =========

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = x.view(b, -1)  # flatten

        if self.mode == "dense":
            return self.impl(x)

        elif self.mode == "lowrank":
            # (b,in) @ (in,r)^T -> (b,r) ; (b,r) @ (r,out) -> (b,out)  [等价 A@B]
            # 这里写成 (A @ B) 的右乘版本，避免先构 W（更省显存）
            # y = x @ B.T @ A.T + bias
            y = x @ self.B.t()
            y = y @ self.A.t()
            return y + self.bias

        elif self.mode == "grouped":
            xs = torch.split(x, self.group_in, dim=1)
            # 求和融合各组映射（也可改为拼接+线性降维，LLM可进化）
            y = sum(layer(xi) for layer, xi in zip(self.group_layers, xs)) + self.bias
            return y

def build_model():
    # 你原来的两层结构：3072->256->10
    return nn.Sequential(
        EvoLinearLayer(3072, 256),
        nn.ReLU(inplace=True),
        EvoLinearLayer(256, 10),
    )
