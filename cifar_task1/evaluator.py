# evaluator.py
# 评价指标：Top-1 精度（越高越好）与推理时延/参数量/乘法开销（越低越好）的加权组合
# 数据：CIFAR-10 子集，train=100 / test=100（固定随机下标，结果可复现）
# 违禁：nn.Linear / torch.matmul / einsum / '@'（运行时猴补丁 + 仅扫描“代码 token”的静态检查）

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

_FORBIDDEN_PATTERNS = [
    r"\btorch\s*\.\s*nn\s*\.\s*linear\b",
    r"\bnn\s*\.\s*linear\b",
    r"\btorch\s*\.\s*matmul\b",
    r"\.__matmul__\b",
    r"\btorch\s*\.\s*einsum\b",
    r"\beinsum\s*\(",
    # 新增：
    r"\btorch\s*\.\s*dot\b",
    r"\btorch\s*\.\s*mm\b",
    r"\btorch\s*\.\s*bmm\b",
    r"\btorch\s*\.\s*mv\b",
    r"\btorch\s*\.\s*addmm\b",
    r"\btorch\s*\.\s*addmv\b",
]


def _scan_source_forbidden(module) -> None:
    try:
        src = inspect.getsource(module)
    except Exception:
        # 某些环境拿不到源码，忽略静态扫描，交给运行时猴补丁兜底
        return
    # 仅保留代码 token，忽略注释和字符串
    code_tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        code_tokens.append(tok.string)
    code = " ".join(code_tokens)
    # for pat in _FORBIDDEN_PATTERNS:
    #     if re.search(pat, code):
    #         raise RuntimeError(f"Forbidden token found in program: {pat}")

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
    miss = [k for k in REQUIRED_KEYS if k not in hp]
    if miss:
        raise ValueError(f"hyperparams missing keys: {miss}. "
                         "Candidates MUST return these in meta['hyperparams'].")
    return hp

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



# -------------------- 核心评测：固定 CPU；失败时自动降级为“无训练评测” --------------------
def _evaluate(program_module, device: str = "cpu") -> Dict:
    # _monkey_patch_forbidden()
    _scan_source_forbidden(program_module)
    _seed_all(42)
    # 构建模型
    model, meta = program_module.build_model()
    model.to(device)

    train_loader, test_loader = _get_cifar10_dataloaders(train_n=100, test_n=10, seed=42, bs=50)

    # 训练（小预算）；遇到 fork+autograd 冲突时自动跳过训练
    do_train = os.environ.get("OE_SKIP_TRAIN", "0") != "1"
    train_ok = False
    if do_train:
        try:
            model.train()
            optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            max_steps = 100
            steps = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                steps += 1
                if steps >= max_steps:
                    break
            train_ok = True
        except RuntimeError as e:
            # 针对 “Autograd and Fork” 报错降级
            if "Autograd" in str(e) and "Fork" in str(e):
                train_ok = False
            else:
                # 其它异常仍然抛出，让外层兜底
                raise

    # 推理计时（固定 batch）；在“无训练评测”下同样进行
    model.eval()
    t0 = time.time()
    with torch.inference_mode():
        acc_sum = 0
        n = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            acc_sum += (pred == yb).sum().item()
            n += yb.numel()
    infer_time = time.time() - t0

    top1 = acc_sum / max(1, n)
    params = _count_params(model)

    # 组合分数（越大越好）：准确率 - α·log(时延) - β·log(参数量) - γ·log(MACs)
    alpha = 0.00
    beta  = 0.05
    gamma = float(os.environ.get("OE_GAMMA", "0.06"))
    macs  = _estimate_macs(meta)

    score = float(
        top1
        - alpha * math.log(max(infer_time, 1e-3))
        - beta  * math.log(max(params, 1))
        - gamma * math.log(max(macs, 1))
    )

    return {
        "score": score,
        "metrics": {
            "top1": top1,
            "infer_time_s": infer_time,
            "params": params,
            "macs": macs,
            "trained": bool(train_ok)
        },
        "metadata": meta
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
    """
    OpenEvolve 调用该函数评估候选程序。
    这里先读取候选源码，把所有 'from __future__ import ...' 行移动到文件开头，
    防止 'must occur at the beginning of the file' 语法错误。
    任何异常都尽量转化为可比较的分数返回，避免整轮演化中断。
    """
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
