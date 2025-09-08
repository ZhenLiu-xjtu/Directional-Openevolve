#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===================== 全局并行/线程设置（必须在最顶部，任何张量创建前） =====================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager
from tqdm import tqdm

# 仅在程序启动时设置（避免运行中报错）
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ===================== 参数与设备选择 =====================
def parse_args():
    p = argparse.ArgumentParser()
    # 设备选择
    p.add_argument("--gpu", type=str, default="cpu",
                   help="选择显卡：如 '0' 或 '1'；'cpu' 或 '-1' 表示用CPU；'1,3' 表示限制可见卡集合。")
    # 数据与训练预算
    p.add_argument("--data", type=str, default="/data/lz/openevolve/dataset/cifar-10-batches-py",
                   help="CIFAR-10 的解包文件夹路径（含 data_batch_* 与 test_batch）")
    p.add_argument("--max-train", type=int, default=1000,
                   help="训练集子集大小")
    p.add_argument("--max-test", type=int, default=500,
                   help="测试集子集大小")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    # 架构切换
    p.add_argument("--arch", choices=["evolved", "orig"], default="evolved",
                   help="orig=单层 conv(3->10)，evolved=两层 conv(3->32->10)")
    # 进化层可调开关（可按需微调）
    p.add_argument("--use-transposed-weight", action="store_true",
                   help="EvolvedLoopLinear 是否使用预转置权重路径（减少访存/中间量）")
    p.add_argument("--tile-size", type=int, default=0,
                   help=">0 时按通道/输入分块累加，降低一次性广播的中间量")
    return p.parse_args()

def configure_device(gpu_arg: str):
    if gpu_arg in ["cpu", "-1"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif "," in gpu_arg:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_arg)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

# ===================== 数据加载（直接读取 cifar-10-batches-py） =====================
def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].astype(np.float32) / 255.0
    labels = np.array(batch[b'labels'], dtype=np.int64)
    data = data.reshape(-1, 3, 32, 32)
    return torch.from_numpy(data), torch.from_numpy(labels)

def load_cifar10_data(data_dir, max_train=None, max_test=None, seed=42):
    trX, trY = [], []
    for i in range(1, 6):
        x, y = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
        trX.append(x); trY.append(y)
    trX = torch.cat(trX, 0); trY = torch.cat(trY, 0)
    teX, teY = load_cifar10_batch(os.path.join(data_dir, "test_batch"))

    g = torch.Generator().manual_seed(seed)
    if max_train is not None and max_train < len(trX):
        idx = torch.randperm(len(trX), generator=g)[:max_train]
        trX, trY = trX[idx], trY[idx]
    if max_test is not None and max_test < len(teX):
        idx = torch.randperm(len(teX), generator=g)[:max_test]
        teX, teY = teX[idx], teY[idx]
    return trX, trY, teX, teY

# ===================== 原始 for-loop Linear（保留以备对照/复用） =====================
class LinearLoopLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features))  # [out,in]
        self.b = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # [B, in]
        B = x.size(0)
        out = x.new_zeros(B, self.out_features)
        for i in range(B):
            for j in range(self.out_features):
                out[i, j] = torch.dot(x[i], self.W[j])
        if self.b is not None:
            out = out + self.b
        return out



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
            # Parameters
            self.W = nn.Parameter(torch.empty(out_dim, in_dim))  # Stored as [out, in]
            self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None

            # Initialize weights
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            if self.b is not None:
                fan_in = in_dim
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.b, -bound, bound)

            # Transposed weight cache
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
            B = x.shape[0]
            out = x.new_zeros((B, self.out_dim))
            W = self._maybe_transpose()
            W = self._apply_sparsify(W)

            # Simplified implementation with better memory access
            if self.use_transposed_weight:
                # When using transposed weights [in, out], optimize for input sparsity
                for b in range(B):
                    for i in range(self.in_dim):
                        xi = x[b, i].item()
                        if xi == 0.0:
                            continue
                        out[b] += xi * W[i]
            else:
                # For non-transposed weights [out, in], optimize for batch processing
                for b in range(B):
                    for j in range(self.out_dim):
                        # Use vectorized operations where possible
                        out[b, j] = (x[b] * W[j]).sum().item()

            if self.b is not None:
                out += self.b
            return out


# ===================== 进化后的 LoopLinear（向量化/广播乘加，无 matmul/einsum/nn.Linear） =====================
# ===================== 用 EvolvedLoopLinear 拼装的“卷积” =====================
class EvolvedLoopConv2d(nn.Module):
    """
    用 F.unfold(im2col) + EvolvedLoopLinear 实现 Conv2d：
      x[B,C,H,W] --unfold--> cols[B, C*kH*kW, L]
                       -> [B*L, C*kH*kW] --EvolvedLoopLinear--> [B*L, out_ch]
                       -> [B,out_ch,L] -> [B,out_ch,H_out,W_out]
    不使用 matmul/einsum/nn.Linear；卷积核权重由内部 EvolvedLoopLinear 管理。
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True,
                 use_transposed_weight: bool = False,
                 tile_size: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)
        self.kH, self.kW = int(kH), int(kW)
        self.stride = int(stride)
        self.padding = int(padding)

        in_feat = self.in_ch * self.kH * self.kW
        self.proj = EvolvedLoopLinear(
            in_dim=in_feat,
            out_dim=self.out_ch,
            bias=bias,
            use_transposed_weight=use_transposed_weight,
            tile_size=tile_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_ch, f"Expected {self.in_ch} input channels, got {C}"
        # im2col
        cols = F.unfold(
            x,
            kernel_size=(self.kH, self.kW),
            dilation=1,
            padding=self.padding,
            stride=self.stride
        )  # [B, in_ch*kH*kW, L]
        B_, in_feat, L = cols.shape
        cols = cols.transpose(1, 2).contiguous()  # [B, L, in_feat]
        cols = cols.view(B_ * L, in_feat)         # [B*L, in_feat]

        # 逐位置做“线性”变换
        out = self.proj(cols)                     # [B*L, out_ch]
        out = out.view(B_, L, self.out_ch).transpose(1, 2).contiguous()  # [B, out_ch, L]

        # 还原空间维
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        out = out.view(B_, self.out_ch, H_out, W_out)  # [B, out_ch, H_out, W_out]
        return out

# ===================== 构建两种模型（orig=1层conv；evolved=2层conv） =====================
def build_model(arch: str, evolved_use_T: bool, evolved_tile: int):
    """
    orig:    Conv(3->10,k3,s1,p1) + GAP + Flatten  => [B,10]
    evolved: Conv(3->32,k3,s1,p1) + ReLU + Conv(32->10,k3,s1,p1) + GAP + Flatten
    """
    if arch == "orig":
        model = nn.Sequential(
            EvolvedLoopConv2d(
                in_channels=3, out_channels=10,
                kernel_size=3, stride=1, padding=1,
                bias=True,
                use_transposed_weight=evolved_use_T,
                tile_size=evolved_tile
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # -> [B, 10]
        )

    elif arch == "evolved":
        model = nn.Sequential(
            EvolvedLoopConv2d(
                in_channels=3, out_channels=32,
                kernel_size=3, stride=1, padding=1,
                bias=True,
                use_transposed_weight=evolved_use_T,
                tile_size=evolved_tile
            ),
            nn.ReLU(inplace=True),
            EvolvedLoopConv2d(
                in_channels=32, out_channels=10,
                kernel_size=3, stride=1, padding=1,
                bias=True,
                use_transposed_weight=evolved_use_T,
                tile_size=evolved_tile
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # -> [B, 10]
        )
    else:
        raise ValueError("Unknown arch")
    return model

# ===================== 评测与工具函数 =====================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total

def count_params(m):  # 与 evaluator 同口径
    return sum(p.numel() for p in m.parameters())

@contextmanager
def cuda_mem_debug(enabled=True):
    if enabled and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        yield
        torch.cuda.synchronize()
        print(f"[CUDA] peak allocated = {torch.cuda.max_memory_allocated()/1024/1024:.2f} MB")
    else:
        yield

def bench_latency(model, loader, device):
    # 整集推理时延：单线程，包含一次热身
    model.eval()
    # 热身
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            break
    # 计时
    t0 = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0

# ===================== 主流程 =====================
def main():
    args = parse_args()

    # 设备
    device = configure_device(args.gpu)
    print(f"[Device] {device} "
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device.type=='cuda' else ''}")

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 数据
    train_X, train_Y, test_X, test_Y = load_cifar10_data(
        args.data, args.max_train, args.max_test, seed=args.seed
    )
    pin = (device.type == "cuda")
    trainloader = DataLoader(TensorDataset(train_X, train_Y),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=pin)
    testloader  = DataLoader(TensorDataset(test_X,  test_Y),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)

    # 模型
    model = build_model(args.arch, args.use_transposed_weight, args.tile_size).to(device)
    n_params = count_params(model)
    print(f"[Model] arch={args.arch} | params={n_params:,}")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    best_train_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_correct, running_total = 0, 0
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_total += y.size(0)
                running_correct += (logits.argmax(1) == y).sum().item()

        train_acc = 100.0 * running_correct / running_total
        test_acc = evaluate(model, testloader, device)
        best_train_acc = max(best_train_acc, train_acc)
        best_test_acc = max(best_test_acc, test_acc)
        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 推理整集时延
    with cuda_mem_debug(enabled=(device.type=="cuda")):
        infer_time_s = bench_latency(model, testloader, device)

    final_acc = evaluate(model, testloader, device)
    print(f"\n===== Summary =====")
    print(f"Arch: {args.arch}")
    print(f"Params: {n_params:,}")
    print(f"Best Train Acc (@subset): {best_train_acc:.2f}%")
    print(f"Best Test  Acc (@subset): {best_test_acc:.2f}%")
    print(f"Final Test Acc (@subset): {final_acc:.2f}%")
    print(f"Infer Time (whole test subset): {infer_time_s:.4f} s")

if __name__ == "__main__":
    main()
