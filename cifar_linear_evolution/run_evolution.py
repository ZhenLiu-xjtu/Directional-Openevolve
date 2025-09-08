import subprocess
import importlib.util
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入 tqdm
import numpy as np
import os, pickle, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入 tqdm
import subprocess
import importlib.util
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入 tqdm

# 配置参数（根据你的需求调整路径）
DATA_DIR = r'D:\dataset\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py'
SEED = 42
MAX_TRAIN = 200  # 训练子集样本数
MAX_TEST = 100  # 测试子集样本数
BATCH_SIZE = 128
EPOCHS = 300
LR = 0.01

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- 数据加载 --------------------
def load_cifar10_batch(filename):
    import pickle
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
        trX.append(x);
        trY.append(y)
    trX = torch.cat(trX, 0);
    trY = torch.cat(trY, 0)
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
test_ds = TensorDataset(test_X, test_Y)

trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -------------------- 模型定义 --------------------
class LinearLoopLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, -1)  # 扁平化输入
        out = x.new_zeros(B, self.out_features)
        for i in range(B):
            for j in range(self.out_features):
                out[i, j] = torch.dot(x[i], self.weights[j]) + self.bias[j]
        return out


# -------------------- 评测 --------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total


# -------------------- 加载最优程序 --------------------
def load_best_program(checkpoint_dir: str):
    """动态加载最新的 best_program.py"""
    best_py = os.path.join(checkpoint_dir, "best_program.py")
    if os.path.exists(best_py):
        spec = importlib.util.spec_from_file_location("best_program", best_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    else:
        return None


# -------------------- 训练与进化 --------------------
def train_with_evolution():
    # 初始化模型
    model = nn.Sequential(
        LinearLoopLayer(3072, 256),
        nn.ReLU(),
        LinearLoopLayer(256, 10)
    ).to(device)

    # 优化器和损失函数
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # 数据加载器
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        # 训练模型
        model.train()
        running_correct, running_total = 0, 0
        for x, y in tqdm(trainloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_total += y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()

        train_acc = 100.0 * running_correct / running_total
        test_acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # 每个 epoch 后进行进化，并更新模型
        print(f"Epoch {epoch}: Evolving model structure...")
        # 设置环境变量，强制 Python 使用 UTF-8 编码
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        subprocess.run([
            "python", "D:\openevolve\openevolve-main\openevolve-run.py",
            "D:\openevolve\openevolve-main\cifar_linear_evolution\initial_program.py",
            "D:\openevolve\openevolve-main\cifar_linear_evolution\evaluator.py",
            "--config", "D:\openevolve\openevolve-main\cifar_linear_evolution\config.yaml",
            "--iterations", "10"
        ])

        # 加载最优程序并替换模型
        checkpoint_dir = "D:\openevolve\openevolve-main\cifar_linear_evolution\openevolve_output\checkpoints\checkpoint_last"
        mod = load_best_program(checkpoint_dir)

        if mod is not None:
            # 如果找到了 best_program.py，就用进化后的模型
            model = mod.build_model().to(device)
            print(f"Epoch {epoch}: Updated model from best program.")
        else:
            # 否则，使用初始的线性模型继续训练
            print(f"Epoch {epoch}: Using original Linear model.")
            model = nn.Sequential(
                LinearLoopLayer(3072, 256),
                nn.ReLU(),
                LinearLoopLayer(256, 10)
            ).to(device)

    final_acc = evaluate(model, testloader, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")


# 调用训练函数
train_with_evolution()
