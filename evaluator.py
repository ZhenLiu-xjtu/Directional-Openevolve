# evaluator.py
import os, sys, json, time, math, pickle, argparse, importlib.util, random
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------- 可配置项 ----------
DEFAULT_DATA_DIR = os.getenv("CIFAR10_DIR", "/data/lz/openevolve/dataset/cifar-10-batches-py")
SPLIT_DIR = "bench/splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].astype(np.float32) / 255.0
    labels = np.array(batch[b'labels'], dtype=np.int64)
    data = data.reshape(-1, 3, 32, 32)
    return torch.from_numpy(data), torch.from_numpy(labels)

def stratified_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idxs = []
    for c in np.unique(labels):
        cand = np.where(labels == c)[0]
        sel = rng.choice(cand, size=per_class, replace=False)
        idxs.append(sel)
    return np.concatenate(idxs)

def build_subset(data_dir: str, n_train_per_class=10, n_test_per_class=10, seed=42):
    # 聚合全量 CIFAR-10
    trX, trY, teX, teY = [], [], None, None
    for i in range(1, 6):
        x, y = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
        trX.append(x); trY.append(y)
    trX = torch.cat(trX, 0); trY = torch.cat(trY, 0)
    teX, teY = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    # 分层采样
    tr_idx = stratified_indices(trY.numpy(), n_train_per_class, seed)
    te_idx = stratified_indices(teY.numpy(), n_test_per_class, seed)
    # 存档（可复现）
    with open(os.path.join(SPLIT_DIR, f"train_100_seed{seed}.json"), "w") as f:
        json.dump(tr_idx.tolist(), f)
    with open(os.path.join(SPLIT_DIR, f"test_100_seed{seed}.json"), "w") as f:
        json.dump(te_idx.tolist(), f)
    return trX[tr_idx], trY[tr_idx], teX[te_idx], teY[te_idx]

def import_model_from_program(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # noqa
    # 约定程序内定义 Model 类
    assert hasattr(mod, "Model"), "Program must define a `Model` class."
    return mod.Model

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops_mlp(model: nn.Module, in_dim=3072) -> float:
    """简易 FLOPs 估计：FC multiply-add 算 2*in*out；仅统计自定义 FC 层。"""
    flops = 0.0
    for m in model.modules():
        if hasattr(m, "W") and isinstance(m.W, torch.nn.Parameter):
            out_f, in_f = m.W.shape
            flops += 2.0 * in_f * out_f
    return flops

@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total

def train_one(model, trainloader, testloader, device, epochs=5, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    # 计时（包含前后向）
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
    t1 = time.perf_counter()
    train_time = t1 - t0
    ips = len(trainloader.dataset) * epochs / max(train_time, 1e-9)
    acc = evaluate_acc(model, testloader, device)
    return acc, train_time, ips

def direction_reward(metrics: Dict[str, float], island: str) -> float:
    """
    方向驱动岛屿（DDI）奖励：
    - accuracy：更看重 Acc；对延迟惩罚弱
    - efficiency：更看重 IPS/低 Params/低 FLOPs
    - stability：鼓励 Acc 高且 std 低（这里只用单次，可置 0；多种子时可加）
    - novelty：鼓励探索（此处留空 0，由平台的网格命中率/新奇度计算）
    """
    acc = metrics["acc"]; latency = metrics["latency"]; params = metrics["params"]; flops = metrics["flops"]; ips = metrics["ips"]
    if island == "accuracy":
        return 0.6*acc - 0.2*math.log1p(latency)
    if island == "efficiency":
        return 0.8*ips - 0.5*math.log1p(params) - 0.5*math.log1p(flops)
    if island == "stability":
        return 0.5*acc  # 多种子时可加 -1.0*std
    if island == "novelty":
        return 0.0
    return 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--program", type=str, default="initial_program.py")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--device", type=str, default=os.getenv("DEVICE", "cpu"))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "5")))
    p.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "128")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    p.add_argument("--island", type=str, default=os.getenv("OE_ISLAND", "accuracy"))  # 由配置为各岛设定
    args = p.parse_args()

    # 固定线程让评测可重复（你也可放开以吃满 CPU）
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    set_global_seed(args.seed)
    train_X, train_Y, test_X, test_Y = build_subset(args.data_dir, 10, 10, args.seed)

    trainloader = DataLoader(TensorDataset(train_X, train_Y), batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader  = DataLoader(TensorDataset(test_X,  test_Y), batch_size=args.batch_size, shuffle=False, num_workers=0)

    Model = import_model_from_program(args.program)
    model = Model()  # 起点使用 initial_program.py 中默认结构

    device = torch.device(args.device)
    acc, train_time, ips = train_one(model, trainloader, testloader, device, epochs=args.epochs, lr=0.01)
    latency = train_time / (args.epochs + 1e-9)  # 每 epoch 的平均时间（s）
    params = float(count_params(model)) / 1e6    # 以百万计
    flops  = float(estimate_flops_mlp(model)) / 1e6

    # 基础多目标合成（与岛屿方向独立）
    base_score = acc - 0.5*math.log1p(latency) - 0.1*math.log1p(params)
    # DDI 方向增益
    delta = direction_reward({"acc":acc,"latency":latency,"params":params,"flops":flops,"ips":ips}, args.island)
    score = base_score + delta

    out = {
        "score": score,
        "base_score": base_score,
        "acc": acc,
        "latency": latency,
        "ips": ips,
        "params": params,
        "flops": flops,
        "seed": args.seed,
        "island": args.island,
        "program_path": os.path.abspath(args.program),
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
