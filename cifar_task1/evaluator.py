# evaluator.py — CIFAR-10 evaluator with worker-safe training, rich diagnostics
from __future__ import annotations
import os

# Threads (can be overridden by OE_NUM_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OE_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("OE_NUM_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("OE_NUM_THREADS", "1"))

import sys
import time
import math
import inspect
import importlib.util
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import io, tokenize, re

try:
    torch.set_num_threads(max(1, int(os.environ.get("OE_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(max(1, int(os.environ.get("OE_NUM_INTEROP_THREADS", "1"))))
except Exception:
    pass

# -------------------- helpers --------------------
def _pick_device() -> str:
    want = os.environ.get("OE_DEVICE", "").lower()
    if want in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    if want in ("mps", "metal") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _synchronize(dev: str) -> None:
    try:
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()  # type: ignore[attr-defined]
    except Exception:
        pass

def _in_subprocess() -> bool:
    try:
        import multiprocessing as mp
        return mp.current_process().name != "MainProcess"
    except Exception:
        return False

# -------------------- runtime guards: forbid ops --------------------
def _monkey_patch_forbidden():
    def _raise(*args, **kwargs):
        raise RuntimeError("Use of forbidden API detected (nn.Linear/matmul/einsum/@/dot/mm/mv/bmm/addmm/addmv)")
    nn.Linear = _raise  # type: ignore
    torch.matmul = _raise  # type: ignore
    torch.Tensor.__matmul__ = _raise  # type: ignore
    torch.einsum = _raise  # type: ignore
    torch.dot = _raise
    torch.mm = _raise
    torch.mv = _raise
    torch.bmm = _raise
    torch.addmm = _raise
    torch.addmv = _raise

# -------------------- static scan (ignore comments/strings) --------------------
_FORBIDDEN_PATTERNS = [
    r"\btorch\s*\.\s*nn\s*\.\s*linear\b",
    r"\bnn\s*\.\s*linear\b",
    r"\btorch\s*\.\s*matmul\b",
    r"\.__matmul__\b",
    r"\btorch\s*\.\s*einsum\b",
    r"\beinsum\s*\(",
    r"\btorch\s*\.\s*dot\b",
    r"\btorch\s*\.\s*mm\b",
    r"\btorch\s*\.\s*mv\b",
    r"\btorch\s*\.\s*bmm\b",
    r"\btorch\s*\.\s*addmm\b",
    r"\btorch\s*\.\s*addmv\b",
]
def _scan_source_forbidden(module) -> None:
    try:
        src = inspect.getsource(module)
    except Exception:
        return
    toks = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        toks.append(tok.string)
    code = " ".join(toks)
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            raise RuntimeError(f"Forbidden token found in program: {pat}")

# -------------------- data / loaders --------------------
def _seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _auto_num_workers(device: str) -> int:
    w_env = os.environ.get("OE_DATALOADER_WORKERS")
    if w_env is not None:
        try: return max(0, int(w_env))
        except Exception: return 0
    if _in_subprocess():
        return 0  # worker 内一律 0，避免 Autograd+Fork
    if device != "cpu":
        return 0
    return 0  # 主进程 CPU 如需多进程再自行调大

def _get_cifar10_dataloaders(train_n=None, test_n=None, seed=42, bs=None, device="cpu"):
    _seed_all(seed)
    train_n = int(os.environ.get("OE_TRAIN_SUBSET", train_n if train_n is not None else 100))
    test_n  = int(os.environ.get("OE_EVAL_SUBSET",  test_n  if test_n  is not None else 10))
    bs      = int(os.environ.get("OE_BATCH_SIZE",   bs      if bs      is not None else 50))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
    ])
    root = os.environ.get("CIFAR10_ROOT", "/data/lz/openevolve/dataset")
    train_set = datasets.CIFAR10(root=root, train=True,  download=True, transform=tfm)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_set), generator=g)[:train_n].tolist()
    test_idx  = torch.randperm(len(test_set),  generator=g)[:test_n].tolist()

    workers = _auto_num_workers(device)
    pin = (device == "cuda")
    prefetch = int(os.environ.get("OE_PREFETCH_FACTOR", "2"))
    persistent = (workers > 0)

    train_loader = DataLoader(
        Subset(train_set, train_idx), batch_size=bs, shuffle=True,
        num_workers=workers, persistent_workers=persistent,
        pin_memory=pin, prefetch_factor=prefetch if workers > 0 else None,
    )
    test_loader = DataLoader(
        Subset(test_set, test_idx), batch_size=bs, shuffle=False,
        num_workers=workers, persistent_workers=persistent,
        pin_memory=pin, prefetch_factor=prefetch if workers > 0 else None,
    )
    return train_loader, test_loader, workers

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def _estimate_macs(meta: dict) -> int:
    hp = meta.get("hyperparams", {})
    in_dim = int(hp.get("in_dim", 3*32*32))
    C      = int(hp.get("num_classes", 10))
    H      = int(hp.get("hidden_dim", 0) or 0)
    return int(in_dim * H + H * C if H > 0 else in_dim * C)

# -------------------- core eval --------------------
def _evaluate(program_module, device: str | None = None) -> Dict:
    _monkey_patch_forbidden()
    _scan_source_forbidden(program_module)

    # 确保不被外部环境跳过训练
    os.environ.pop("OE_SKIP_TRAIN", None)

    device = device or _pick_device()
    model, meta = program_module.build_model()
    model.to(device)

    train_loader, test_loader, workers = _get_cifar10_dataloaders(device=device)

    max_steps = int(os.environ.get("OE_MAX_STEPS", "20"))
    max_train_batches = int(os.environ.get("OE_MAX_TRAIN_BATCHES", "0"))
    do_train = os.environ.get("OE_SKIP_TRAIN", "0") != "1"

    train_ok = False
    train_exc_summary = None

    if do_train:
        try:
            # —— 训练前预检：可微 + 形状 + 数值（把越界/断图提前识别为“训练异常”，而不是致命错误）——
            for p in model.parameters():
                p.requires_grad_(True)

            _chk_iter = iter(train_loader)
            try:
                xb_chk, yb_chk = next(_chk_iter)
            except StopIteration:
                xb_chk = yb_chk = None

            if xb_chk is not None:
                xb_chk = xb_chk.to(device, non_blocking=(device == "cuda"))[:1]
                yb_chk = yb_chk.to(device, non_blocking=(device == "cuda"))[:1]

                logits_chk = model(xb_chk)  # 这里若有 index 10 越界，会直接抛异常到本 try

                # 形状检查：必须 [N, num_classes]
                C = int((meta.get("hyperparams") or {}).get("num_classes", 10))
                if not (isinstance(logits_chk, torch.Tensor) and logits_chk.dim() == 2 and logits_chk.size(1) == C):
                    raise RuntimeError(f"Invalid logits shape: expected [N,{C}], got {tuple(logits_chk.shape)}")

                # 数值稳定性
                if torch.isnan(logits_chk).any() or torch.isinf(logits_chk).any():
                    raise RuntimeError("NaN/Inf in logits")

                # 可微性
                if not logits_chk.requires_grad:
                    raise RuntimeError("Non-differentiable forward: logits have no grad_fn (detach/no_grad?)")

            # —— 真正训练 ——
            model.train()
            optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
            criterion = nn.CrossEntropyLoss()
            steps = 0
            for bi, (xb, yb) in enumerate(train_loader, start=1):
                xb = xb.to(device, non_blocking=(device == "cuda"))
                yb = yb.to(device, non_blocking=(device == "cuda"))
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                if not (isinstance(loss, torch.Tensor) and loss.requires_grad):
                    raise RuntimeError("Loss not requiring grad (likely .item()/.detach()/no_grad used)")

                loss.backward()
                optim.step()
                steps += 1
                if max_train_batches > 0 and bi >= max_train_batches: break
                if steps >= max_steps: break
            _synchronize(device)
            train_ok = True

        except Exception as e:
            # 打到 stderr，并把摘要回传到 metrics
            if os.environ.get("OE_PRINT_TRAIN_EXC", "0") == "1":
                print("TRAIN EXC (downgrade to no-train):", repr(e), file=sys.stderr, flush=True)
            train_ok = False
            train_exc_summary = f"{type(e).__name__}: {str(e).splitlines()[0]}"

    # ---------------- Inference ----------------
    model.eval()
    _synchronize(device)
    t0 = time.time()
    acc_sum = 0; n = 0
    try:
        with torch.inference_mode():
            max_eval_batches = int(os.environ.get("OE_MAX_EVAL_BATCHES", "0"))
            for bi, (xb, yb) in enumerate(test_loader, start=1):
                xb = xb.to(device, non_blocking=(device == "cuda"))
                yb = yb.to(device, non_blocking=(device == "cuda"))
                logits = model(xb)  # 若前向依旧越界/报错，这里会触发 except
                pred = logits.argmax(dim=1)
                acc_sum += (pred == yb).sum().item()
                n += yb.numel()
                if max_eval_batches > 0 and bi >= max_eval_batches:
                    break
    except Exception as e:
        # 前向异常（例如 index 10 越界）—— 优雅降级，返回惩罚分，并带上 train_exc 线索
        err_msg = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        metrics = {
            "error": 1.0,
            "trained": bool(train_ok),
            "device": device,
            "dataloader_workers": workers,
        }
        if train_exc_summary:
            metrics["train_exc"] = train_exc_summary
        return {"score": -1e9, "metrics": metrics, "message": err_msg, "metadata": meta}

    _synchronize(device)
    infer_time = time.time() - t0

    # 可选：超慢候选硬阈值（设置 OE_MAX_INFER_TIME>0 启用）
    MAX_INFER_S = float(os.environ.get("OE_MAX_INFER_TIME", "0"))
    if MAX_INFER_S > 0 and infer_time > MAX_INFER_S:
        metrics = {
            "error": 1.0,
            "infer_time_s": infer_time,
            "trained": bool(train_ok),
            "device": device,
            "dataloader_workers": workers,
        }
        if train_exc_summary:
            metrics["train_exc"] = train_exc_summary
        return {"score": -1e9, "metrics": metrics, "message": f"Timeout: infer_time {infer_time:.3f}s > {MAX_INFER_S}s", "metadata": meta}

    top1 = acc_sum / max(1, n)
    params = _count_params(model)
    macs  = _estimate_macs(meta)

    alpha = 0.20; beta = 0.05; gamma = float(os.environ.get("OE_GAMMA", "0.02"))
    score = float(
        top1
        - alpha * math.log(max(infer_time, 1e-3))
        - beta  * math.log(max(params, 1))
        - gamma * math.log(max(macs, 1))
    )

    metrics = {
        "top1": top1,
        "infer_time_s": infer_time,
        "params": params,
        "macs": macs,
        "trained": bool(train_ok),
        "device": device,
        "dataloader_workers": workers,
    }
    if not train_ok and train_exc_summary:
        metrics["train_exc"] = train_exc_summary  # ✅ 回传异常摘要

    return {"score": score, "metrics": metrics, "metadata": meta}

# -------------------- CLI quick test --------------------
def main():
    module_path = os.environ.get("CANDIDATE_PATH", "initial_program.py")
    spec = importlib.util.spec_from_file_location("candidate_program", module_path)
    assert spec and spec.loader, f"cannot load program at {module_path}"
    program_module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_program"] = program_module
    spec.loader.exec_module(program_module)  # type: ignore
    result = _evaluate(program_module, device=_pick_device())
    print({"combined_score": result["score"], **result})

# -------------------- OpenEvolve entry --------------------
def evaluate(candidate_program_path: str) -> dict:
    """
    Load candidate safely (fix __future__ placement), evaluate, and return comparable score.
    Any exception returns a very low score and message instead of crashing evolution.
    """
    import types
    with open(candidate_program_path, "r", encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    future = [ln for ln in lines if ln.strip().startswith("from __future__ import")]
    others = [ln for ln in lines if not ln.strip().startswith("from __future__ import")]
    fixed_src = ""
    if future: fixed_src += "\n".join(future) + "\n"
    fixed_src += "\n".join(others) + "\n"

    module_name = "candidate_program"
    program_module = types.ModuleType(module_name)
    sys.modules[module_name] = program_module
    try:
        code_obj = compile(fixed_src, candidate_program_path, "exec")
        exec(code_obj, program_module.__dict__)
        result = _evaluate(program_module, device=_pick_device())
        return {"combined_score": result["score"], **result}
    except Exception as e:
        return {"combined_score": -1e9, "metrics": {"error": 1.0}, "message": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(evaluate(sys.argv[1]))
    else:
        main()
