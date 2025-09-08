#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------- 解析 GPU 选择（需在 import torch 前设置可见卡更稳妥） --------------------
import os, argparse, pickle, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="7",
                    help="选择显卡：如 '0' 或 '1'；'cpu' 或 '-1' 表示用CPU；也可传 '1,3' 来限制可见卡集合，此时脚本内部使用 cuda:0。")
# 新增：结构与实现选择
parser.add_argument("--arch", choices=["orig", "evolved"], default="evolved",
                    help="orig=单层线性卷积(3->10)，evolved=双层线性卷积(3->32->10)")
parser.add_argument("--conv-impl", choices=["loop", "evolved"], default="loop",
                    help="loop=用 LinearLoopLayer 组成卷积；evolved=用 EvolvedLoopLinear 组成卷积（向量化更快）")
# 可选：evolved 版线性层的一些开关
parser.add_argument("--use-transposed-weight", action="store_true",
                    help="EvolvedLoopLinear 是否使用预转置权重路径（降低中间量）")
parser.add_argument("--tile-size", type=int, default=0,
                    help=">0 时分块累加，降低一次性广播的中间量")
args, _ = parser.parse_known_args()

if args.gpu in ["cpu", "-1"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU
elif "," in args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# -------------------- 配置 --------------------
DATA_DIR   = r'/data/lz/openevolve/dataset/cifar-10-batches-py'
SEED       = 42
MAX_TRAIN  = 1000
MAX_TEST   = 500
BATCH_SIZE = 256
EPOCHS     = 300
LR         = 0.01

torch.manual_seed(SEED)
np.random.seed(SEED)

# 根据可见卡设置 device；若设置了 '1,3'，此处的 cuda:0 实际对应原系统的 1 号卡
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # 始终使用可见设备列表中的第0张
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"[Device] {device} "
      f"{'(' + torch.cuda.get_device_name(0) + ')' if device.type=='cuda' else ''}")

# -------------------- 数据加载 --------------------
def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].astype(np.float32) / 255.0
    labels = np.array(batch[b'labels'], dtype=np.int64)
    data = data.reshape(-1, 3, 32, 32)
    return torch.from_numpy(data), torch.from_numpy(labels)

def load_cifar10_data(data_dir, max_train=None, max_test=None):
    trX, trY = [], []
    for i in range(1, 6):
        x, y = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
        trX.append(x); trY.append(y)
    trX = torch.cat(trX, 0); trY = torch.cat(trY, 0)
    teX, teY = load_cifar10_batch(os.path.join(data_dir, "test_batch"))

    if max_train is not None and max_train < len(trX):
        idx = torch.randperm(len(trX))[:max_train]
        trX, trY = trX[idx], trY[idx]
    if max_test is not None and max_test < len(teX):
        idx = torch.randperm(len(teX))[:max_test]
        teX, teY = teX[idx], teY[idx]
    return trX, trY, teX, teY

train_X, train_Y, test_X, test_Y = load_cifar10_data(DATA_DIR, MAX_TRAIN, MAX_TEST)
train_ds = TensorDataset(train_X, train_Y)
test_ds  = TensorDataset(test_X,  test_Y)

pin = (device.type == "cuda")
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=pin)
testloader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=pin)

# -------------------- 线性层（循环实现，已存在） --------------------
class LinearLoopLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias    = nn.Parameter(torch.zeros(out_features)) if bias else None
        # 简单初始化
        nn.init.kaiming_uniform_(self.weights, a=5**-0.5)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # [B, in]
        B = x.size(0)
        out = x.new_zeros(B, self.out_features)
        for i in range(B):
            for j in range(self.out_features):
                out[i, j] = torch.dot(x[i], self.weights[j])
        if self.bias is not None:
            out = out + self.bias
        return out

# -------------------- 向量化的“循环线性层”（更快，不用 matmul/einsum/nn.Linear） --------------------
class EvolvedLoopLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 use_transposed_weight=False, tile_size=0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_transposed_weight = bool(use_transposed_weight)
        self.tile_size = int(tile_size)

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features))  # [out,in]
        self.b = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        nn.init.kaiming_uniform_(self.W, a=5**-0.5)

        self.register_buffer("_WT_cache", None, persistent=False)

    def _maybe_transpose(self):
        if self.use_transposed_weight:
            if (self._WT_cache is None) or (self._WT_cache.shape != (self.in_features, self.out_features)):
                self._WT_cache = self.W.t().contiguous()  # [in,out]
            return self._WT_cache
        else:
            self._WT_cache = None
            return self.W  # [out,in]

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # [B, in]
        B = x.size(0)
        W_like = self._maybe_transpose()

        if not self.use_transposed_weight:
            out = x.new_zeros(B, self.out_features)
            if self.tile_size > 0:
                for j0 in range(0, self.out_features, self.tile_size):
                    j1 = min(j0 + self.tile_size, self.out_features)
                    for j in range(j0, j1):
                        wj = W_like[j:j+1, :]          # [1,in]
                        out[:, j] = (x * wj).sum(dim=1)
            else:
                for j in range(self.out_features):
                    wj = W_like[j:j+1, :]
                    out[:, j] = (x * wj).sum(dim=1)
        else:
            WT = W_like  # [in,out]
            if self.tile_size > 0:
                out = x.new_zeros(B, self.out_features)
                for i0 in range(0, self.in_features, self.tile_size):
                    i1 = min(i0 + self.tile_size, self.in_features)
                    x_tile = x[:, i0:i1]
                    w_tile = WT[i0:i1, :]
                    out += (x_tile.unsqueeze(2) * w_tile.unsqueeze(0)).sum(dim=1)
            else:
                out = (x.unsqueeze(2) * WT.unsqueeze(0)).sum(dim=1)  # [B,out]

        if self.b is not None:
            out = out + self.b
        return out

# -------------------- 用 Linear 组合出来的“卷积” --------------------
class LoopConv2d(nn.Module):
    """
    用 F.unfold(im2col) + LinearLoopLayer 实现 Conv2d：
      x[B,C,H,W] --unfold--> [B, in_ch*kH*kW, L] -> reshape 成 [B*L, in_feat]
      -> LinearLoopLayer(in_feat, out_ch) 逐样本逐通道点积
      -> [B*L, out] -> 回到 [B, out, H_out, W_out]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
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
        self.proj = LinearLoopLayer(in_feat, self.out_ch, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_ch
        cols = F.unfold(x, kernel_size=(self.kH, self.kW),
                        dilation=1, padding=self.padding, stride=self.stride)  # [B,in_feat,L]
        B_, in_feat, L = cols.shape
        cols = cols.transpose(1, 2).contiguous().view(B_ * L, in_feat)         # [B*L, in_feat]
        out = self.proj(cols)                                                  # [B*L, out_ch]
        out = out.view(B_, L, self.out_ch).transpose(1, 2).contiguous()        # [B, out_ch, L]
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        return out.view(B_, self.out_ch, H_out, W_out)

class EvolvedLoopConv2d(nn.Module):
    """
    同上，但用 EvolvedLoopLinear（批量/广播乘加，速度友好）
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, use_transposed_weight=False, tile_size=0):
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
            in_features=in_feat, out_features=self.out_ch, bias=bias,
            use_transposed_weight=use_transposed_weight, tile_size=tile_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_ch
        cols = F.unfold(x, kernel_size=(self.kH, self.kW),
                        dilation=1, padding=self.padding, stride=self.stride)  # [B,in_feat,L]
        B_, in_feat, L = cols.shape
        cols = cols.transpose(1, 2).contiguous().view(B_ * L, in_feat)         # [B*L, in_feat]
        out = self.proj(cols)                                                  # [B*L, out_ch]
        out = out.view(B_, L, self.out_ch).transpose(1, 2).contiguous()        # [B, out_ch, L]
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        return out.view(B_, self.out_ch, H_out, W_out)

# -------------------- 构建两种网络（orig=1层线性卷积；evolved=2层线性卷积） --------------------
def build_model(arch: str, conv_impl: str):
    Conv = LoopConv2d if conv_impl == "loop" else \
           (lambda ic, oc, k, s=1, p=0, bias=True:
                EvolvedLoopConv2d(ic, oc, k, s, p, bias=bias,
                                  use_transposed_weight=args.use_transposed_weight,
                                  tile_size=args.tile_size))
    if arch == "orig":
        model = nn.Sequential(
            Conv(3, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # -> [B,10]
        )
    elif arch == "evolved":
        model = nn.Sequential(
            Conv(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            Conv(32, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # -> [B,10]
        )
    else:
        raise ValueError("Unknown arch")
    return model

# -------------------- 评测 --------------------
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

# -------------------- 训练 --------------------
model = build_model(args.arch, args.conv_impl).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

best_train_acc = 0.0
best_test_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_correct, running_total = 0, 0
    for x, y in tqdm(trainloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch"):
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
    print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

final_acc = evaluate(model, testloader, device)
print(f"\n===== Summary =====")
print(f"Arch: {args.arch} | Impl: {args.conv_impl}")
print(f"Best Train Acc (@subset): {best_train_acc:.2f}%")
print(f"Best Test  Acc (@subset): {best_test_acc:.2f}%")
print(f"Final Test Acc (@subset): {final_acc:.2f}%")
