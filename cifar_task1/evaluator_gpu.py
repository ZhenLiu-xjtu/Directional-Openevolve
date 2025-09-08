# evaluator.py
# CIFAR-10 子集评测：Top-1 精度（高优）与推理时延/参数量（低优）的组合
# 禁止 nn.Linear / matmul / einsum / @

from __future__ import annotations
import os, sys, time, math, inspect, importlib.util
from typing import Dict
import io, tokenize, re

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

# ---- 线程限制，避免 fork+autograd 冲突 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ---- GPU 设置（固定用第7号卡） ----
DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"
USE_AMP = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---- 禁止 API ----
def _monkey_patch_forbidden():
    def _raise(*args, **kwargs):
        raise RuntimeError("Use of forbidden API detected (nn.Linear/matmul/einsum/@)")
    nn.Linear = _raise  # type: ignore
    torch.matmul = _raise  # type: ignore
    torch.Tensor.__matmul__ = _raise  # type: ignore
    torch.einsum = _raise  # type: ignore

_FORBIDDEN_PATTERNS = [
    r"\btorch\s*\.\s*nn\s*\.\s*linear\b",
    r"\bnn\s*\.\s*linear\b",
    r"\btorch\s*\.\s*matmul\b",
    r"\.__matmul__\b",
    r"\btorch\s*\.\s*einsum\b",
    r"\beinsum\s*\(",
]

def _scan_source_forbidden(module) -> None:
    try:
        src = inspect.getsource(module)
    except Exception:
        return
    code_tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        code_tokens.append(tok.string)
    code = " ".join(code_tokens)
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            raise RuntimeError(f"Forbidden token found in program: {pat}")

# ---- 数据 ----
def _seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _get_cifar10_dataloaders(train_n=100, test_n=100, seed=42, bs=50):
    _seed_all(seed)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
    ])
    root = os.environ.get("CIFAR10_ROOT", "/data/lz/openevolve/dataset")
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_set), generator=g)[:train_n].tolist()
    test_idx  = torch.randperm(len(test_set), generator=g)[:test_n].tolist()
    pin = DEVICE.startswith("cuda")
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(Subset(test_set,  test_idx),  batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, test_loader

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ---- 评测 ----
def _evaluate(program_module, device: str = DEVICE) -> Dict:
    _monkey_patch_forbidden()
    _scan_source_forbidden(program_module)

    model, meta = program_module.build_model()
    model.to(device)
    train_loader, test_loader = _get_cifar10_dataloaders()

    # 训练
    train_ok = False
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.startswith("cuda"))
    try:
        model.train()
        optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
        criterion = nn.CrossEntropyLoss()
        steps = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if USE_AMP and device.startswith("cuda"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(xb); loss = criterion(logits, yb)
                scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
            else:
                logits = model(xb); loss = criterion(logits, yb)
                loss.backward(); optim.step()
            steps += 1
            if steps >= 100: break
        train_ok = True
    except RuntimeError as e:
        if "Autograd" in str(e) and "Fork" in str(e):
            train_ok = False
        else:
            raise

    # 推理
    model.eval(); t0 = time.time()
    with torch.inference_mode():
        acc_sum = 0; n = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if USE_AMP and device.startswith("cuda"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(xb)
            else:
                logits = model(xb)
            pred = logits.argmax(dim=1)
            acc_sum += (pred == yb).sum().item(); n += yb.numel()
    infer_time = time.time() - t0

    top1 = acc_sum / max(1, n); params = _count_params(model)
    alpha, beta = 0.20, 0.05
    score = float(top1 - alpha*math.log(max(infer_time,1e-3)) - beta*math.log(max(params,1)))
    return {"score": score, "metrics": {"top1": top1, "infer_time_s": infer_time, "params": params, "trained": bool(train_ok)}, "metadata": meta}

# ---- CLI 测试入口 ----
def main():
    module_path = os.environ.get("CANDIDATE_PATH", "initial_program.py")
    spec = importlib.util.spec_from_file_location("candidate_program", module_path)
    assert spec and spec.loader
    program_module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_program"] = program_module
    spec.loader.exec_module(program_module)  # type: ignore
    result = _evaluate(program_module)
    print({"combined_score": result["score"], **result})

# ---- OpenEvolve 入口 ----
def evaluate(candidate_program_path: str) -> dict:
    import types
    with open(candidate_program_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
    others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
    fixed_src = ("\n".join(future)+"\n" if future else "") + "\n".join(others) + "\n"

    module_name = "candidate_program"
    program_module = types.ModuleType(module_name)
    sys.modules[module_name] = program_module
    try:
        code_obj = compile(fixed_src, candidate_program_path, "exec")
        exec(code_obj, program_module.__dict__)
        result = _evaluate(program_module)
        return {"combined_score": result["score"], **result}
    except RuntimeError as e:
        if "Autograd" in str(e) and "Fork" in str(e):
            os.environ["OE_SKIP_TRAIN"]="1"
            try:
                result=_evaluate(program_module)
                return {"combined_score": result["score"], **result}
            finally:
                os.environ.pop("OE_SKIP_TRAIN",None)
        return {"combined_score": -1e9,"metrics":{"error":1.0},"message":f"{type(e).__name__}: {e}"}
    except Exception as e:
        return {"combined_score": -1e9,"metrics":{"error":1.0},"message":f"{type(e).__name__}: {e}"}

if __name__=="__main__":
    if len(sys.argv)==2: print(evaluate(sys.argv[1]))
    else: main()
