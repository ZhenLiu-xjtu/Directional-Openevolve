# evaluator.py
import os, sys, time, importlib.util, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 为了与你之前的数据设置一致，支持环境变量传入
DATA_DIR   = os.environ.get("CIFAR10_DIR", r"D:\dataset\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py")
SEED       = int(os.environ.get("SEED", "42"))
MAX_TRAIN  = int(os.environ.get("MAX_TRAIN", "2000"))
MAX_TEST   = int(os.environ.get("MAX_TEST", "1000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
EPOCHS     = int(os.environ.get("EVAL_EPOCHS", "3"))   # 评测只跑很短几轮，保证演化迭代快
LR         = float(os.environ.get("LR", "0.02"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

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

def _dyn_import(path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

@torch.no_grad()
def evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def evaluate(program_path: str):
    """
    OpenEvolve 默认会把“当前候选代码文件的路径”传给这个函数。
    返回一个 metrics 字典，包含 combined_score 作为主适应度。
    """
    # 1) 动态导入候选程序，构建模型
    prog = _dyn_import(program_path)
    model = prog.build_model().to(DEVICE)

    # 2) 数据
    trX, trY, teX, teY = load_cifar10_data(DATA_DIR, MAX_TRAIN, MAX_TEST)
    trainloader = DataLoader(TensorDataset(trX, trY), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    testloader  = DataLoader(TensorDataset(teX, teY), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3) 小步训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    t0 = time.time()
    model.train()
    for _ in range(EPOCHS):
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - t0

    # 4) 评测 + 目标：acc - α * log(params) - β * time
    val_acc = evaluate_acc(model, testloader)
    params  = float(count_params(model))
    score   = float(val_acc - 0.15 * np.log(params + 1.0) - 0.2 * elapsed)

    return {
        "val_acc": float(val_acc),
        "params":  params,
        "elapsed_sec": float(elapsed),
        "combined_score": score
    }

# 本地调试：python evaluator.py <path_to_initial_program.py>
if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    print(evaluate(p))
