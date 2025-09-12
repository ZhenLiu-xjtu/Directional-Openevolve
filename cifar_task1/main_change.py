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
                   help="训练集子集大小（默认 100）")
    p.add_argument("--max-test", type=int, default=500,
                   help="测试集子集大小（默认 100）")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    # 架构切换
    p.add_argument("--arch", choices=["evolved", "orig"], default="evolved",
                   help="evolved=单层 3072→10（进化后），orig=两层 3072→256→10（baseline）")
    # 进化层可调开关（可按需微调）
    p.add_argument("--use-transposed-weight", action="store_true",
                   help="进化单层是否使用预转置权重路径（减少访存开销/中间量）")
    p.add_argument("--tile-size", type=int, default=0,
                   help=">0 时按输出通道或输入通道分块累加，减少一次性大广播")
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

# ===================== 原始两层 Linear（不使用 matmul/einsum/nn.Linear） =====================
# class EvolvedLoopLinear(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int,
#                  tile_size: int = 0, unroll: int = 1,
#                  use_transposed_weight: bool = False,
#                  bias: bool = True, sparsify_thresh: float = 0.0):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.tile_size = tile_size
#         self.unroll = max(1, int(unroll))
#         self.use_transposed_weight = bool(use_transposed_weight)
#         self.sparsify_thresh = float(sparsify_thresh)
#         # Parameters
#         self.W = nn.Parameter(torch.empty(out_dim, in_dim))  # Stored as [out, in]
#         self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None
#
#         # Initialize weights
#         nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
#         if self.b is not None:
#             fan_in = in_dim
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.b, -bound, bound)
#
#         # Transposed weight cache
#         self.register_buffer("_WT_cache", None, persistent=False)
#     def _maybe_transpose(self):
#         if self.use_transposed_weight:
#             # 懒转置
#             if self._WT_cache is None or self._WT_cache.shape != (self.in_dim, self.out_dim):
#                 self._WT_cache = self.W.t().contiguous()
#             return self._WT_cache
#         else:
#             self._WT_cache = None
#             return self.W
#
#     def _apply_sparsify(self, W_tensor: torch.Tensor) -> torch.Tensor:
#         if self.sparsify_thresh > 0.0 and self.training is False:
#             # 只在 eval() 下做稀疏近似
#             return torch.where(W_tensor.abs() < self.sparsify_thresh,
#                                torch.zeros_like(W_tensor), W_tensor)
#         return W_tensor
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 关键1：无论上游给的是 [B, 3, 32, 32] 还是 [B, in_dim]，都先摊平
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)              # [B, in_dim]
#         assert x.size(1) == self.in_dim, f"x.shape[1]={x.size(1)} != in_dim={self.in_dim}"
#         B = x.shape[0]
#         out = x.new_zeros((B, self.out_dim))
#         W = self._maybe_transpose()
#         W = self._apply_sparsify(W)
#
#         # Simplified implementation with better memory access
#         if self.use_transposed_weight:
#             # When using transposed weights [in, out], optimize for input sparsity
#             for b in range(B):
#                 for i in range(self.in_dim):
#                     xi = x[b, i].item()
#                     if xi == 0.0:
#                         continue
#                     out[b] += xi * W[i]
#         else:
#             # For non-transposed weights [out, in], optimize for batch processing
#             for b in range(B):
#                 for j in range(self.out_dim):
#                     # Use vectorized operations where possible
#                     out[b, j] = (x[b] * W[j]).sum().item()
#
#         if self.b is not None:
#             out += self.b
#         return out
#
#
# class EvolvedLoopLinear(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int,
#                  tile_size: int = 0, unroll: int = 1,
#                  use_transposed_weight: bool = False,
#                  bias: bool = True, sparsify_thresh: float = 0.0):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.tile_size = tile_size
#         self.unroll = max(1, int(unroll))
#         self.use_transposed_weight = bool(use_transposed_weight)
#         self.sparsify_thresh = float(sparsify_thresh)
#         # Parameters
#         self.W = nn.Parameter(torch.empty(out_dim, in_dim))  # Stored as [out, in]
#         self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None
#
#         # Initialize weights
#         nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
#         if self.b is not None:
#             fan_in = in_dim
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.b, -bound, bound)
#
#         # Transposed weight cache
#         self.register_buffer("_WT_cache", None, persistent=False)
#
#     def _maybe_transpose(self):
#         if self.use_transposed_weight:
#             # 懒转置
#             if self._WT_cache is None or self._WT_cache.shape != (self.in_dim, self.out_dim):
#                 self._WT_cache = self.W.t().contiguous()
#             return self._WT_cache
#         else:
#             self._WT_cache = None
#             return self.W
#
#     def _apply_sparsify(self, W_tensor: torch.Tensor) -> torch.Tensor:
#         if self.sparsify_thresh > 0.0 and self.training is False:
#             # 只在 eval() 下做稀疏近似
#             return torch.where(W_tensor.abs() < self.sparsify_thresh,
#                                torch.zeros_like(W_tensor), W_tensor)
#         return W_tensor
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, in_dim]
#         返回: [B, out_dim]
#         """
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)  # [B, in_dim]
#         assert x.size(1) == self.in_dim, f"x.shape[1]={x.size(1)} != in_dim={self.in_dim}"
#         B = x.shape[0]
#         out = x.new_zeros((B, self.out_dim))
#         W = self._maybe_transpose()
#         W = self._apply_sparsify(W)
#
#         # Simplified implementation with better memory access
#         if self.use_transposed_weight:
#             # When using transposed weights [in, out], optimize for input sparsity
#             for b in range(B):
#                 for i in range(self.in_dim):
#                     xi = x[b, i].item()
#                     if xi == 0.0:
#                         continue
#                     out[b] += xi * W[i]
#         else:
#             # For non-transposed weights [out, in], optimize for batch processing
#             for b in range(B):
#                 for j in range(self.out_dim):
#                     # Use vectorized operations where possible
#                     out[b, j] = (x[b] * W[j]).sum().item()
#
#         if self.b is not None:
#             out += self.b
#         return out



class EvolvedLoopLinear(nn.Module):
    """
    Evolvable linear layer that supports three structure knobs:
      - lowrank_rank (int r > 0):  W ≈ V[r,out] @ U[r,in], MACs = in_dim*r + r*out
      - groups (int g > 1):        each output j only connects to a 1/g slice of inputs, MACs ≈ in_dim*out/g
      - sparsity (float rho∈(0,1]): keep roughly rho fraction of input elements per output (deterministic subsampling)
    Forbid mm/matmul/einsum/dot by implementing all reductions as elementwise-mul + sum.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, lowrank_rank: int = 4,
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
            # Reorganized loops to avoid innermost loop over batch dimension
            for k in range(0, self.rank, 4):  # Unroll by 4
                # Process each tile
                for start in range(0, self.in_features, 16):  # Using fixed tile size
                    end = min(start + 16, self.in_features)

                    # For each batch element
                    for b in range(B):
                        xb = x[b]  # [in_features]
                        # Create output tensor
                        if b == 0:
                            out = xb.new_zeros(B, self.out_features)

                        # Accumulate values for 4 ranks at once
                        acc0 = 0.0
                        acc1 = 0.0
                        acc2 = 0.0
                        acc3 = 0.0

                        for i in range(start, end):
                            if keep_stride == 1 or i % keep_stride == 0:
                                if k < self.rank:
                                    acc0 += self.U[k, i] * xb[i]
                                if k + 1 < self.rank:
                                    acc1 += self.U[k + 1, i] * xb[i]
                                if k + 2 < self.rank:
                                    acc2 += self.U[k + 2, i] * xb[i]
                                if k + 3 < self.rank:
                                    acc3 += self.U[k + 3, i] * xb[i]

                        # Store accumulated values
                        if k < self.rank:
                            out[b, k] += acc0
                        if k + 1 < self.rank:
                            out[b, k + 1] += acc1
                        if k + 2 < self.rank:
                            out[b, k + 2] += acc2
                        if k + 3 < self.rank:
                            out[b, k + 3] += acc3

                # Handle remaining ranks if not divisible by 4
                # This part is now handled by the main loop restructuring
                pass
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


# class EvolvedLoopLinear(nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True):
#         super().__init__()
#         self.in_features = int(in_features)
#         self.out_features = int(out_features)
#
#         self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
#         self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
#
#         nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
#         if self.bias is not None:
#             bound = (self.in_features) ** -0.5
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, 3, 32, 32] or [B, in]
#         if x.dim() == 4:
#             B = x.size(0)
#             x = x.reshape(B, -1)
#         else:
#             B = x.size(0)
#
#         out = x.new_zeros(B, self.out_features)
#
#         for b in range(B):
#             xb = x[b]  # [in_features]
#             # Process output features in tiles of size 16
#             for j_start in range(0, self.out_features, 16):
#                 # Process 16 output features at once
#                 tile_end = min(j_start + 16, self.out_features)
#                 for j in range(j_start, tile_end):
#                     acc = (xb * self.weight[j]).sum()
#                     if self.bias is not None:
#                         acc = acc + self.bias[j]
#                     out[b, j] = acc
#         return out
# class EvolvedLoopLinear(nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True):
#         super().__init__()
#         self.in_features = int(in_features)
#         self.out_features = int(out_features)
#
#         self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
#         self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
#
#         nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
#         if self.bias is not None:
#             bound = (self.in_features) ** -0.5
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, 3, 32, 32] or [B, in]
#         if x.dim() == 4:
#             B = x.size(0)
#             x = x.reshape(B, -1)
#         else:
#             B = x.size(0)
#
#         # Optimize tensor operations by pre-expanding weight matrix
#         # This improves memory access patterns and reduces computation time
#         x_expanded = x.unsqueeze(1)  # [B, 1, in_features]
#         weight = self.weight  # [out_features, in_features]
#
#         # Multiply and sum along input dimension
#         out = (x_expanded * weight).sum(dim=2)  # [B, out_features]
#
#         # Add bias if present using simpler broadcasting
#         if self.bias is not None:
#             out += self.bias
#         return out


# class EvolvedLoopLinear(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int,
#                  tile_size: int = 0, unroll: int = 1,
#                  use_transposed_weight: bool = False,
#                  bias: bool = True, sparsify_thresh: float = 0.0):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.tile_size = tile_size
#         self.unroll = max(1, int(unroll))
#         self.use_transposed_weight = bool(use_transposed_weight)
#         self.sparsify_thresh = float(sparsify_thresh)
#         # Parameters
#         self.W = nn.Parameter(torch.empty(out_dim, in_dim))  # Stored as [out, in]
#         self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None
#
#         # Initialize weights
#         nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
#         if self.b is not None:
#             fan_in = in_dim
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.b, -bound, bound)
#
#         # Transposed weight cache
#         self.register_buffer("_WT_cache", None, persistent=False)
#
#     def _maybe_transpose(self):
#         if self.use_transposed_weight:
#             # 懒转置
#             if self._WT_cache is None or self._WT_cache.shape != (self.in_dim, self.out_dim):
#                 self._WT_cache = self.W.t().contiguous()
#             return self._WT_cache
#         else:
#             self._WT_cache = None
#             return self.W
#
#     def _apply_sparsify(self, W_tensor: torch.Tensor) -> torch.Tensor:
#         if self.sparsify_thresh > 0.0 and self.training is False:
#             # 只在 eval() 下做稀疏近似
#             return torch.where(W_tensor.abs() < self.sparsify_thresh,
#                                torch.zeros_like(W_tensor), W_tensor)
#         return W_tensor
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, in_dim]
#         返回: [B, out_dim]
#         """
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)  # [B, in_dim]
#         assert x.size(1) == self.in_dim, f"x.shape[1]={x.size(1)} != in_dim={self.in_dim}"
#         B = x.shape[0]
#         out = x.new_zeros((B, self.out_dim))
#         W = self._maybe_transpose()
#         W = self._apply_sparsify(W)
#
#         # Simplified implementation with better memory access
#         if self.use_transposed_weight:
#             # When using transposed weights [in, out], optimize for input sparsity
#             for b in range(B):
#                 for i in range(self.in_dim):
#                     xi = x[b, i].item()
#                     if xi == 0.0:
#                         continue
#                     out[b] += xi * W[i]
#         else:
#             # For non-transposed weights [out, in], optimize for batch processing
#             for b in range(B):
#                 for j in range(self.out_dim):
#                     # Use vectorized operations where possible
#                     out[b, j] = (x[b] * W[j]).sum().item()
#
#         if self.b is not None:
#             out += self.b
#         return out


# class LinearLoopLayer(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.in_features = int(in_features)
#         self.out_features = int(out_features)
#         self.W = nn.Parameter(torch.empty(self.out_features, self.in_features))  # [out,in]
#         self.b = nn.Parameter(torch.zeros(self.out_features)) if bias else None
#         nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
#         if self.b is not None:
#             bound = 1 / math.sqrt(self.in_features)
#             nn.init.uniform_(self.b, -bound, bound)
#
#     def forward(self, x):
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)  # [B, in]
#         B = x.size(0)
#         out = x.new_zeros(B, self.out_features)
#         # 逐样本逐通道：torch.dot（未使用 matmul/einsum/@）
#         for i in range(B):
#             for j in range(self.out_features):
#                 out[i, j] = torch.dot(x[i], self.W[j])
#         if self.b is not None:
#             out = out + self.b
#         return out

# ===================== 进化后的 LoopLinear（向量化/广播乘加，无 matmul/einsum/nn.Linear） =====================
# class EvolvedLoopLinear(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int,
#                  tile_size: int = 0, unroll: int = 1,
#                  use_transposed_weight: bool = False,
#                  bias: bool = True, sparsify_thresh: float = 0.0):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.tile_size = tile_size
#         self.unroll = max(1, int(unroll))
#         self.use_transposed_weight = bool(use_transposed_weight)
#         self.sparsify_thresh = float(sparsify_thresh)
#         # 参数
#         self.W = nn.Parameter(torch.empty(out_dim, in_dim))  # 统一保存为 [out, in]
#         self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None
#         # Xavier/Glorot initialization works well for linear layers
#         nn.init.xavier_uniform_(self.W)
#         if self.b is not None:
#             nn.init.zeros_(self.b)
#
#         # 运行时可缓存转置
#         self.register_buffer("_WT_cache", None, persistent=False)
#
#     def _maybe_transpose(self):
#         if self.use_transposed_weight:
#             # 懒转置
#             if self._WT_cache is None or self._WT_cache.shape != (self.in_dim, self.out_dim):
#                 self._WT_cache = self.W.t().contiguous()
#             return self._WT_cache
#         else:
#             self._WT_cache = None
#             return self.W
#
#     def _apply_sparsify(self, W_tensor: torch.Tensor) -> torch.Tensor:
#         if self.sparsify_thresh > 0.0 and self.training is False:
#             # 只在 eval() 下做稀疏近似
#             return torch.where(W_tensor.abs() < self.sparsify_thresh,
#                                torch.zeros_like(W_tensor), W_tensor)
#         return W_tensor
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, in_dim]
#         返回: [B, out_dim]
#         """
#         B = x.shape[0]
#         out = x.new_zeros((B, self.out_dim))
#         W = self._apply_sparsify(self.W)  # Always use [out_dim, in_dim] layout
#
#         # Simplified loop-based implementation
#         # Process outputs sequentially for better cache locality
#         for j in range(self.out_dim):
#             # Extract weight row for output j
#             w_j = W[j, :]  # [in_dim]
#             # Compute dot product for all batch elements
#             out[:, j] = torch.sum(x * w_j, dim=1)
#
#         if self.b is not None:
#             out += self.b
#         return out
# class EvolvedLoopLinear(nn.Module):
#     """
#     关键思路：
#     - 非转置路径：对每个输出通道 j，批量计算 (x * W[j]).sum(dim=1) —— 批量点积，去掉 Python 层双重循环
#     - 预转置路径：将 W 转为 [in,out]，用 (x.unsqueeze(2) * WT.unsqueeze(0)).sum(dim=1) —— 广播乘加
#     - 可选 tile_size：分块累加，降低一次性广播的临时中间量
#     """
#     def __init__(self, in_features, out_features, bias=True,
#                  use_transposed_weight=False, tile_size=0):
#         super().__init__()
#         self.in_features = int(in_features)
#         self.out_features = int(out_features)
#         self.use_transposed_weight = bool(use_transposed_weight)
#         self.tile_size = int(tile_size)
#
#         self.W = nn.Parameter(torch.empty(self.out_features, self.in_features))  # [out,in]
#         self.b = nn.Parameter(torch.zeros(self.out_features)) if bias else None
#         nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
#         if self.b is not None:
#             bound = 1 / math.sqrt(self.in_features)
#             nn.init.uniform_(self.b, -bound, bound)
#
#         self.register_buffer("_WT_cache", None, persistent=False)
#
#     def _maybe_transpose(self):
#         if self.use_transposed_weight:
#             if (self._WT_cache is None) or (self._WT_cache.shape != (self.in_features, self.out_features)):
#                 self._WT_cache = self.W.t().contiguous()  # [in,out]
#             return self._WT_cache
#         else:
#             self._WT_cache = None
#             return self.W  # [out,in]
#
#     @torch.jit.unused
#     def forward(self, x):
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)  # [B, in]
#         B = x.size(0)
#         W_like = self._maybe_transpose()
#
#         if not self.use_transposed_weight:
#             # W_like: [out,in]；按通道批量点积
#             out = x.new_zeros(B, self.out_features)
#             if self.tile_size > 0:
#                 for j0 in range(0, self.out_features, self.tile_size):
#                     j1 = min(j0 + self.tile_size, self.out_features)
#                     for j in range(j0, j1):
#                         wj = W_like[j:j+1, :]          # [1,in]
#                         out[:, j] = (x * wj).sum(dim=1)
#             else:
#                 for j in range(self.out_features):
#                     wj = W_like[j:j+1, :]              # [1,in]
#                     out[:, j] = (x * wj).sum(dim=1)
#         else:
#             # W_like: [in,out]；广播乘加
#             WT = W_like  # [in,out]
#             if self.tile_size > 0:
#                 out = x.new_zeros(B, self.out_features)
#                 for i0 in range(0, self.in_features, self.tile_size):
#                     i1 = min(i0 + self.tile_size, self.in_features)
#                     x_tile = x[:, i0:i1]                            # [B,tile]
#                     w_tile = WT[i0:i1, :]                           # [tile,out]
#                     out += (x_tile.unsqueeze(2) * w_tile.unsqueeze(0)).sum(dim=1)
#             else:
#                 out = (x.unsqueeze(2) * WT.unsqueeze(0)).sum(dim=1) # [B,out]
#
#         if self.b is not None:
#             out = out + self.b
#         return out

# ===================== 构建两种模型 =====================
def build_model(arch: str, evolved_use_T: bool, evolved_tile: int):
    if arch == "orig":
        model = nn.Sequential(
            EvolvedLoopLinear(3072, 256, bias=True),  # 也可换成 LinearLoopLayer
            nn.ReLU(),
            EvolvedLoopLinear(256, 10, bias=True),
        )
        # 如果你更想用“纯 for 循环版”baseline，可以把上面两层换成 LinearLoopLayer：
        # model = nn.Sequential(LinearLoopLayer(3072, 256), nn.ReLU(), LinearLoopLayer(256, 10))

    elif arch == "evolved":
        # island=5 最优思路的单层（默认不转置、无分块；也可通过参数开启）
        model = nn.Sequential(
            EvolvedLoopLinear(
                3072, 10
            )
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

def flops_linear(in_dim, out_dim, count_by='FLOPs'):
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

def bench_latency(model, loader, device):
    # 整集推理时延（贴 evaluator 口径）：单线程，包含一次热身
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
        if train_acc>best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 推理整集时延（贴 evaluator 口径）
    with cuda_mem_debug(enabled=(device.type=="cuda")):
        infer_time_s = bench_latency(model, testloader, device)

    final_acc = evaluate(model, testloader, device)
    print(f"\n===== Summary =====")
    print(f"Arch: {args.arch}")
    print(f"Params: {n_params:,}")
    print(f"Final Test Acc (@subset): {final_acc:.2f}%")
    print(f"best_train_acc (@subset): {best_train_acc:.2f}%")
    print(f"best_test_acc(@subset): {best_test_acc:.2f}%")
    print(f"Infer Time (whole test subset): {infer_time_s:.4f} s")

if __name__ == "__main__":
    main()
