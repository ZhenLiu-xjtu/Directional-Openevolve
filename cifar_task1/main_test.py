import os, argparse, pickle, numpy as np

# -------------------- 解析 GPU 选择（需在 import torch 前设置可见卡更稳妥） --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="7",
                    help="选择显卡：如 '0' 或 '1'；'cpu' 或 '-1' 表示用CPU；"
                         "也可传 '1,3' 来限制可见卡集合，此时脚本内部使用 cuda:0。")
args, _ = parser.parse_known_args()

if args.gpu in ["cpu", "-1"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU
elif "," in args.gpu:
    # 限制可见卡集合（如 '1,3'），随后脚本里用 cuda:0 即可
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
else:
    # 只暴露单张目标卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
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
import pickle

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

# -------------------- 线性层（循环实现） --------------------
import math
class LoopLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 tile_size: int = 0, unroll: int = 1,
                 use_transposed_weight: bool = False,
                 bias: bool = True, sparsify_thresh: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 简化参数存储
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None

        # 初始化
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            bound = 1 / math.sqrt(in_dim)
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        返回: [B, out_dim]
        """
        B, in_dim = x.shape
        out = x.new_zeros((B, self.out_dim))

        # 使用双重循环实现矩阵乘法 - 更清晰且符合约束
        for b in range(B):
            for i in range(self.out_dim):
                for j in range(in_dim):
                    out[b, i] += x[b, j] * self.W[i, j]

        if self.b is not None:
            out += self.b
        return out


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
model = nn.Sequential(
nn.Flatten(),
    LoopLinear(3072, 256),
    nn.ReLU(),
    LoopLinear(256, 10)
).to(device)

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
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

final_acc = evaluate(model, testloader, device)
print(f"Final Test Accuracy (subset): {final_acc:.2f}%")
print(f"best_train_acc (@subset): {best_train_acc:.2f}%")
print(f"best_test_acc (@subset): {best_test_acc:.2f}%")
# Final Test Accuracy (subset): 20.60%