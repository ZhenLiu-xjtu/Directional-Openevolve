# reserve_gpu_mem.py
# 占用指定 GPU 的显存（默认 19GB），不进行计算；按块分配并保持进程常驻。
# 依赖：PyTorch（torch）

import argparse, time, signal, sys
import torch

def fmt_bytes(b: int) -> str:
    return f"{b / (1024 ** 3):.2f} GB"

def main():
    parser = argparse.ArgumentParser(description="Reserve GPU memory without computation.")
    parser.add_argument("--gb", type=float, default=19.0, help="Target GPU memory to reserve (in GB). Default: 19")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index after CUDA_VISIBLE_DEVICES remap. Default: 0")
    parser.add_argument("--chunk-gb", type=float, default=0.5, help="Allocation chunk size (GB). Default: 0.5")
    parser.add_argument("--safety-gb", type=float, default=0.5, help="Safety headroom to leave unused (GB). Default: 0.5")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用：请在有 NVIDIA GPU 的环境中运行。")
        sys.exit(1)

    torch.cuda.set_device(args.device)
    dev = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(dev)
    total = props.total_memory
    total_gb = total / (1024 ** 3)

    target_bytes_raw = int(args.gb * (1024 ** 3))
    safety_bytes = int(max(args.safety_gb, 0) * (1024 ** 3))
    # 若用户目标过大，给出提示（不强行修改目标，尽量占用到接近目标）
    if target_bytes_raw + safety_bytes > total:
        print(f"[提示] 目标({args.gb:.2f}GB) + 预留({args.safety_gb:.2f}GB) 超过设备可用总量（约 {total_gb:.2f}GB）。"
              f" 将尽力分配至接近目标，但可能到达物理上限后停止。")

    # 预热上下文，避免首次 tiny alloc 时的统计抖动
    _ = torch.empty(1, device=dev)

    tensors = []
    chunk_bytes = max(int(args.chunk_gb * (1024 ** 3)), 1)  # 至少 1B
    min_chunk_bytes = 1 * 1024 * 1024  # 1MB 下限，避免无限缩小
    print(f"设备: {props.name} | 总显存: {fmt_bytes(total)} | 目标占用: ~{args.gb:.2f}GB | 初始块大小: {fmt_bytes(chunk_bytes)}")

    def cur_reserved() -> int:
        return torch.cuda.memory_reserved(dev)
    def cur_alloc() -> int:
        return torch.cuda.memory_allocated(dev)

    # 分配循环：直到 reserved 达到接近目标（或不能再分配）
    try:
        while True:
            reserved = cur_reserved()
            allocated = cur_alloc()

            # 结束条件：已达到或非常接近目标（考虑安全余量）
            if reserved >= target_bytes_raw:
                print(f"[完成] reserved={fmt_bytes(reserved)} allocated={fmt_bytes(allocated)} "
                      f"(chunks={len(tensors)}). 正在保持占用，Ctrl+C 释放。")
                break

            # 若即将越过物理上限（留安全余量），也停止
            if reserved + chunk_bytes + safety_bytes > total:
                print(f"[到达上限] 继续分配可能导致 OOM。当前 reserved={fmt_bytes(reserved)}，已尽力接近目标。")
                break

            try:
                # 使用 uint8 一字节元素，按字节数精确控制
                tensors.append(torch.empty(chunk_bytes, dtype=torch.uint8, device=dev))
                reserved = cur_reserved()
                allocated = cur_alloc()
                print(f"[分配] +{fmt_bytes(chunk_bytes)} -> reserved={fmt_bytes(reserved)}, "
                      f"allocated={fmt_bytes(allocated)}, chunks={len(tensors)}")
            except RuntimeError as e:
                msg = str(e)
                if "CUDA out of memory" in msg or "out of memory" in msg:
                    # 块太大，减半重试
                    if chunk_bytes > min_chunk_bytes:
                        chunk_bytes = max(chunk_bytes // 2, min_chunk_bytes)
                        print(f"[OOM] 缩小块大小并重试：{fmt_bytes(chunk_bytes)}")
                        continue
                    else:
                        print("[停止] 已到最小块大小仍 OOM，无法进一步分配。")
                        break
                else:
                    raise

        print("占用保持中… 按 Ctrl+C 释放显存并退出。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass

    finally:
        # 释放引用，交还显存
        tensors.clear()
        torch.cuda.empty_cache()
        print("已释放显存，退出。")

if __name__ == "__main__":
    main()
