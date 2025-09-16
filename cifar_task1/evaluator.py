# evaluator.py

from __future__ import annotations
import os
# ---- 线程与后端：避免 fork + autograd 冲突（强制单线程 & CPU）----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import math
import inspect
import importlib
import importlib.util
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import os
import time
import torch
import torch.nn as nn

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# -------------------- 运行时猴补丁：拦截违规 API --------------------
# def _monkey_patch_forbidden():
    # def _raise(*args, **kwargs):
    #     raise RuntimeError("Use of forbidden API detected (nn.Linear/matmul/einsum/@)")
    # nn.Linear = _raise  # type: ignore
    # torch.matmul = _raise  # type: ignore
    # torch.Tensor.__matmul__ = _raise  # type: ignore
    # torch.einsum = _raise  # type: ignore
    # # 新增：
    # torch.dot = _raise
    # torch.mm = _raise
    # torch.bmm = _raise
    # torch.mv = _raise
    # torch.addmm = _raise
    # torch.addmv = _raise

# -------------------- 静态扫描：忽略注释与字符串，只扫真实代码 --------------------
import io, tokenize, re




def _scan_source_forbidden(program_module):
    """
    静态源码扫描（仅在 OE_SCAN_FORBID=1 时启用）：
    检测候选程序中是否出现被禁止的算子调用。
    """
    if os.environ.get("OE_SCAN_FORBID", "0") != "1":
        return

    # 尝试获取源码
    src_chunks = []
    try:
        src_chunks.append(inspect.getsource(program_module))
    except Exception:
        # 遍历模块属性收集尽可能多的源码片段
        for name in dir(program_module):
            obj = getattr(program_module, name)
            try:
                src_chunks.append(inspect.getsource(obj))
            except Exception:
                pass
    src = "\n".join(src_chunks)

    # 需要禁止的 token（可按需增删）
    tokens = [
        r"\bnn\.Linear\b",
        r"\btorch\.matmul\b",
        r"\beinsum\b",
        r"@",
        r"\btorch\.mm\b",
        r"\btorch\.bmm\b",
        r"\btorch\.mv\b",
        r"\btorch\.dot\b",
        r"\btorch\.addmm\b",
        r"\btorch\.addmv\b",
    ]
    if not src:
        return
    pattern = re.compile("|".join(tokens))
    m = pattern.search(src)
    if m:
        hit = m.group(0)
        raise RuntimeError(f"Forbidden token detected in source: {hit}. "
                           f"Unset OE_SCAN_FORBID or remove forbidden calls.")


# -------------------- 训练/数据工具 --------------------
def _seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _get_cifar10_dataloaders(train_n: int = 100, test_n: int = 100, seed: int = 42, bs: int = 50):
    _seed_all(seed)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ])
    # 注意：root 指向“包含 cifar-10-batches-py 的父目录”
    root = os.environ.get("CIFAR10_ROOT", "/data/lz/openevolve/dataset")
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    # 固定子集索引
    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_set), generator=g)[:train_n].tolist()
    test_idx  = torch.randperm(len(test_set),  generator=g)[:test_n].tolist()

    # 避免 DataLoader 多进程：num_workers=0
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=bs, shuffle=True, num_workers=0)
    test_loader  = DataLoader(Subset(test_set,  test_idx),  batch_size=bs, shuffle=False, num_workers=0)
    return train_loader, test_loader

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())




REQUIRED_KEYS = ("in_dim","num_classes","hidden_dim","lowrank_rank","groups","sparsity")
def _validate_hparams(meta: dict):
    hp = (meta or {}).get("hyperparams", {}) or {}
    defaults = { "in_dim": 3*32*32,
                 "num_classes": 10,
                 "hidden_dim": 0,
                 "lowrank_rank": 0,
                 "groups": 1,
                 "sparsity": 1.0,
                 }
    for k, v in defaults.items():
        hp.setdefault(k, v)
    return hp
# def _validate_hparams(meta: dict):
#     hp = (meta or {}).get("hyperparams", {}) or {}
#     miss = [k for k in REQUIRED_KEYS if k not in hp]
#     if miss:
#         raise ValueError(f"hyperparams missing keys: {miss}. "
#                          "Candidates MUST return these in meta['hyperparams'].")
#     return hp

def _estimate_macs(meta: dict) -> int:
    """
    估算循环 Linear 的乘法次数（MACs），优先读取结构超参；向后兼容旧逻辑。
    约定：
      - 低秩分解（W≈U@V）：macs = in_dim * r + r * C, 其中 r=lowrank_rank
      - 分组线性（g 组）：macs ≈ in_dim * C / g
      - 稀疏（非零比例 ρ）：在上述基础上乘以 ρ（或 1-sparsity）
      - 两层 MLP：in_dim * H + H * C
      - 否则：in_dim * C
    """
    hp = _validate_hparams(meta)

    # hp = meta.get("hyperparams", {}) or {}
    in_dim = int(hp.get("in_dim", 3*32*32))
    C      = int(hp.get("num_classes", 10))
    H      = int(hp.get("hidden_dim", 0) or 0)

    # 优先识别“能显著改变算术量”的结构
    r = int(hp.get("lowrank_rank", 0) or 0)
    g = int(hp.get("groups", 1) or 1)
    sparsity = float(hp.get("sparsity", hp.get("nonzero_ratio", 1.0)))
    if sparsity <= 0.0: sparsity = 1.0
    sparsity = max(0.0, min(1.0, sparsity))

    if r > 0:
        macs = in_dim * r + r * C
    elif g > 1:
        macs = (in_dim * C) // max(1, g)
    elif H > 0:
        macs = in_dim * H + H * C
    else:
        macs = in_dim * C

    macs = int(max(1, macs * sparsity))
    return macs


def _measure_forward_median(model: nn.Module, device: str = "cpu",
                            batch: int = 256, warmup: int = 5, runs: int = 10) -> float:
    model.eval()
    x = torch.randn(batch, 3, 32, 32, device=device)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[len(times) // 2])

def _train_one_epoch(model: nn.Module,
                     train_loader,
                     device: str = "cpu",
                     lr: float = 1e-3,
                     weight_decay: float = 0.0,
                     max_batches: int = None) -> dict:
    """
    最小训练循环：1 个 epoch（或最多 max_batches 个 batch）。
    - 分类任务：CrossEntropyLoss
    - 优化器：Adam(lr, weight_decay)
    - 可选 AMP：设置环境变量 OE_USE_AMP=1 且 device=CUDA 时启用
    返回：{"train_loss": 平均损失, "batches": 实际训练的 batch 数}
    """
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    use_amp = device.startswith("cuda") and os.environ.get("OE_USE_AMP", "0") == "1"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_loss = 0.0
    total_items = 0
    batches = 0

    for i, (xb, yb) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item() * yb.size(0)
        total_items += yb.size(0)
        batches += 1

    avg_loss = total_loss / max(1, total_items)
    return {"train_loss": float(avg_loss), "batches": batches}
def _monkey_patch_forbidden():
    """
    运行期“禁用”高阶算子，强制在 for-loop 搜索空间内演化。
    建议仅在设置了 OE_FORBID_MM=1 时调用。
    被禁用：nn.Linear, torch.matmul, @, einsum, dot, mm, bmm, mv, addmm, addmv
    """
    def _raise(*args, **kwargs):
        raise RuntimeError("Use of forbidden API detected: nn.Linear/matmul/einsum/@/mm/dot/bmm/mv/addmm/addmv")

    nn.Linear = _raise          # type: ignore
    torch.matmul = _raise       # type: ignore
    torch.Tensor.__matmul__ = _raise  # type: ignore
    torch.einsum = _raise       # type: ignore
    torch.dot = _raise
    torch.mm = _raise
    torch.bmm = _raise
    torch.mv = _raise
    torch.addmm = _raise
    torch.addmv = _raise


# -------------------- 核心评测：固定 CPU；失败时自动降级为“无训练评测” --------------------
def _evaluate(program_module, device: str = "cpu") -> dict:
    if os.environ.get("OE_FORBID_MM", "0") == "1":
        _monkey_patch_forbidden()

    # 如果你有源码静态扫描函数，尽量保留；没有就跳过
    try:
        _scan_source_forbidden(program_module)
    except NameError:
        pass

    # 构建模型与 meta
    build_model = getattr(program_module, "build_model")
    model, meta = build_model()
    model.to(device)

    # 数据：扩大测试集，提升准确率分辨率
    train_loader, test_loader = _get_cifar10_dataloaders(
        train_n=100, test_n=100, seed=42, bs=50
    )

    # 训练（若你的评测包含训练过程，保持你原有逻辑；这里只写一个最小示例）
    _train_one_epoch(model, train_loader, device=device)

    # 准确率评测（与时延分开，避免 IO 抖动）
    model.eval()
    with torch.inference_mode():
        acc_sum = 0
        n = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            acc_sum += (pred == yb).sum().item()
            n += yb.numel()
    top1 = acc_sum / max(1, n)

    # 纯前向推理时延：固定输入，多次中位数
    infer_time = _measure_forward_median(model, device=device)

    # 资源与综合分数（保持你原有的公式，只展示典型设置）
    hp = _validate_hparams(meta)
    macs = _estimate_macs(hp)  # 你的实现若依赖 hp，这里 hp 已容错

    # 组合分数：可按需微调 alpha；建议先 0.005 与 0.010 做 A/B
    alpha = 0.005
    score = float(top1) - alpha * float(max(1e-9, (infer_time if infer_time > 0 else 1e-9)))

    return {
        "top1": float(top1),
        "infer_times_s": float(infer_time),
        "macs": float(macs),
        "score": float(score),
        "meta": meta,
    }


# -------------------- CLI 单测入口（不影响 OpenEvolve） --------------------
def main():
    module_path = os.environ.get("CANDIDATE_PATH", "initial_program.py")
    spec = importlib.util.spec_from_file_location("candidate_program", module_path)
    assert spec and spec.loader, f"cannot load program at {module_path}"
    program_module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_program"] = program_module
    spec.loader.exec_module(program_module)  # type: ignore
    result = _evaluate(program_module, device="cpu")
    print({"combined_score": result["score"], **result})

# -------------------- OpenEvolve 标准入口：自动矫正 __future__ 位置 --------------------
def evaluate(candidate_program_path: str) -> dict:

    import types

    # 读取并矫正 __future__ 位置
    with open(candidate_program_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
    others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
    fixed_src = ""
    if future:
        fixed_src += "\n".join(future) + "\n"
    fixed_src += "\n".join(others) + "\n"

    # 动态创建模块并执行
    module_name = "candidate_program"
    program_module = types.ModuleType(module_name)
    sys.modules[module_name] = program_module
    try:
        code_obj = compile(fixed_src, candidate_program_path, "exec")
        exec(code_obj, program_module.__dict__)
        result = _evaluate(program_module, device="cpu")
        return {"combined_score": result["score"], **result}
    except RuntimeError as e:
        # 对 “Autograd & Fork” 等已知问题兜底：跳过训练再评估一次
        if "Autograd" in str(e) and "Fork" in str(e):
            os.environ["OE_SKIP_TRAIN"] = "1"
            try:
                result = _evaluate(program_module, device="cpu")
                return {"combined_score": result["score"], **result}
            finally:
                os.environ.pop("OE_SKIP_TRAIN", None)
        # 其它异常：返回极低分，带上错误信息，保证演化不中断
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(evaluate(sys.argv[1]))
    else:
        main()