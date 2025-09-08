import math, time, torch, torch.nn as nn
from contextlib import contextmanager
import os, math, time, torch, torch.nn as nn
# ——在任何张量/模型创建之前、只设置一次——
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)  # 只能在程序一开始设置
except RuntimeError:
    # 若运行环境已启动并行，避免再次设置导致报错
    pass
# ---- Linear 层乘法统计工具 ----
import math
import torch.nn as nn

# 如果你的工程里这两个类名存在，就会被自动识别；没有也没关系
try:
    LoopLinear
except NameError:
    class LoopLinear:  # 占位，只用于 isinstance 检查失败时跳过
        pass

try:
    LinearLoopLayer
except NameError:
    class LinearLoopLayer:
        pass

def report_linear_muls(model: nn.Module, batch_size: int) -> None:
    """
    打印模型里所有 Linear/LoopLinear/LinearLoopLayer 的乘法次数（MACs）与 FLOPs。
    规则：
      - nn.Linear:         MACs = B * in_features * out_features
      - LoopLinear:        MACs = B * in_dim       * out_dim
      - LinearLoopLayer:   MACs = B * in_features  * out_features (按你类里定义的名字取)
    """
    total_macs = 0
    lines = []
    for name, m in model.named_modules():
        muls = None
        if isinstance(m, nn.Linear):
            muls = batch_size * m.in_features * m.out_features
            desc = f"nn.Linear({m.in_features}->{m.out_features})"
        elif isinstance(m, LoopLinear):
            # 你的 LoopLinear 定义里字段叫 in_dim/out_dim
            muls = batch_size * int(m.in_dim) * int(m.out_dim)
            desc = f"LoopLinear({m.in_dim}->{m.out_dim})"
        elif isinstance(m, LinearLoopLayer):
            # 你的 LinearLoopLayer 定义里字段叫 in_features/out_features
            muls = batch_size * int(m.in_features) * int(m.out_features)
            desc = f"LinearLoopLayer({m.in_features}->{m.out_features})"

        if muls is not None:
            flops = 2 * muls  # 乘法+加法各算一次 FLOP
            total_macs += muls
            lines.append(f"[{name}] {desc}  |  MACs={muls:,}  FLOPs={flops:,}")

    # 打印
    if lines:
        print("\n===== Linear Layers Multiply Count =====")
        for s in lines:
            print(s)
        print(f"Total MACs = {total_macs:,}")
        print(f"Total FLOPs = {2*total_macs:,}")
        print("========================================\n")
    else:
        print("\n(No Linear-like layers found for MACs report)\n")


# === 原始两层：逐样本逐通道点积（允许用 torch.dot；未用 matmul/einsum/nn.Linear）===
class LinearLoopLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias    = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, -1)
        out = x.new_zeros(B, self.out_features)
        for i in range(B):
            for j in range(self.out_features):
                out[i, j] = torch.dot(x[i], self.weights[j]) + self.bias[j]
        return out


# === 进化后的单层：批量点积/广播（不使用 matmul/einsum/nn.Linear）===
class EvolvedLoopLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 tile_size: int = 0, unroll: int = 1,
                 use_transposed_weight: bool = False,
                 bias: bool = True, sparsify_thresh: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tile_size = tile_size
        self.unroll = max(1, int(unroll))
        self.use_transposed_weight = bool(use_transposed_weight)
        self.sparsify_thresh = float(sparsify_thresh)
        # 参数
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))  # 统一保存为 [out, in]
        self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in = in_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)

        # 运行时可缓存转置
        self.register_buffer("_WT_cache", None, persistent=False)

    def _maybe_transpose(self):
        if self.use_transposed_weight:
            # 懒转置
            if self._WT_cache is None or self._WT_cache.shape != (self.in_dim, self.out_dim):
                self._WT_cache = self.W.t().contiguous()
            return self._WT_cache
        else:
            self._WT_cache = None
            return self.W

    def _apply_sparsify(self, W_tensor: torch.Tensor) -> torch.Tensor:
        if self.sparsify_thresh > 0.0 and self.training is False:
            # 只在 eval() 下做稀疏近似
            return torch.where(W_tensor.abs() < self.sparsify_thresh,
                               torch.zeros_like(W_tensor), W_tensor)
        return W_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        返回: [B, out_dim]
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # [B, in_dim]
        assert x.size(1) == self.in_dim, f"x.shape[1]={x.size(1)} != in_dim={self.in_dim}"
        B = x.shape[0]
        out = x.new_zeros((B, self.out_dim))
        W_like = self._maybe_transpose()  # 可能是 [out,in] 或 [in,out]
        W_like = self._apply_sparsify(W_like)

        # Simplified implementation that leverages PyTorch's optimized tensor operations
        # while still respecting the constraint of not using matmul/einsum

        # Reshape x to make it easier to broadcast
        x_expanded = x.unsqueeze(2)  # [B, in_dim, 1]
        W_expanded = W_like.unsqueeze(0)  # [1, out_dim, in_dim] or [1, in_dim, out_dim]

        if self.use_transposed_weight:
            # W is [in, out], broadcasting approach
            # We use x: [B, in_dim, 1], W: [in_dim, out_dim]
            # Result will be element-wise multiplication with shape [B, in_dim, out_dim]
            # Then we sum over in_dim dimension
            out = torch.sum(x_expanded * W_expanded, dim=1)
        else:
            # W is [out, in], broadcasting approach
            # We use x: [B, in_dim, 1], W: [out_dim, in_dim]
            # We'll transpose W to [in_dim, out_dim] for broadcasting
            W_transposed = W_like.t()
            out = torch.sum(x_expanded * W_transposed, dim=1)

        # Add bias if present
        if self.b is not None:
            out += self.b

        if self.b is not None:
            out += self.b
        return out

# === 统计与基准口径（与 evaluator 一致：计参、整套测试集前向时长、单线程） ===
def count_params(m):  # evaluator 同口径
    return sum(p.numel() for p in m.parameters())

def model_size_bytes(m, dtype_bytes=4):  # 近似：只算权重
    return count_params(m) * dtype_bytes

def linear_flops(in_dim, out_dim, count_by='FLOPs'):  # 线性层 FLOPs 估算
    mac = in_dim * out_dim
    return mac if count_by.lower() == 'mac' else 2 * mac

@contextmanager
def cuda_mem_debug(enabled=True):
    if enabled and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        yield
        torch.cuda.synchronize()
        print(f"[CUDA] peak allocated = {torch.cuda.max_memory_allocated()/1024/1024:.2f} MB")
    else:
        yield

def benchmark_latency(model, x, device, warmup=1, repeats=5):
    model.eval()
    x = x.to(device)
    # torch.set_num_threads(1); torch.set_num_interop_threads(1)  # 与 evaluator 的“限制并行”思路一致
    if device.type == "cuda": torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.time(); _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append(time.time() - t0)
    times.sort()
    return times[len(times)//2]  # 中位数

def report_model(model, name, x_sample, device, flops_layers):
    model = model.to(device)
    n_params = count_params(model)
    size_mb = model_size_bytes(model)/1024/1024
    with cuda_mem_debug(enabled=(device.type=="cuda")):
        lat = benchmark_latency(model, x_sample, device)
    total_flops = sum(flops_layers)  # 单样本 FLOPs 总计
    print(f"[{name}] params={n_params:,}  (~{size_mb:.2f} MB, fp32)  "
          f"FLOPs/sample≈{total_flops/1e6:.3f}M  latency(batch={x_sample.size(0)})={lat*1000:.2f} ms")

# === 构建两种模型 ===
def build_original_two_layer():
    return nn.Sequential(
        LinearLoopLayer(3072, 10)
    )

def build_evolved_single_layer():
    return nn.Sequential(EvolvedLoopLinear(3072, 10, bias=True, use_transposed_weight=False, tile_size=0))

# === Demo：用随机输入/或你真实的一个 batch 来测 ===
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    B = 256
    x = torch.randn(B, 3, 32, 32)  # 与 CIFAR-10 形状一致
    # 进化单层：FLOPs(样本) = 3072*10（同口径）
    single = build_evolved_single_layer()
    flops_single = [linear_flops(3072, 10)]
    report_model(single, "evolved-single-3072-10", x, device, flops_single)
    report_linear_muls(single, batch_size=B)  # <-- 在这里调用
    # 原始两层：FLOPs(样本) = 3072*256 + 256*10（按 FLOPs=2*MAC 可乘以2）
    orig = build_original_two_layer()
    flops_orig = [linear_flops(3072, 10)]
    report_model(orig, "orig-2layer", x, device, flops_orig)
    report_linear_muls(orig, batch_size=B)  # <-- 在这里调用

