import torch
import time
import argparse
from datetime import datetime
import signal
import sys

# 处理Ctrl+C中断
def signal_handler(sig, frame):
    print("\n收到终止信号，正在退出...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def gpu_worker(gpu_id, memory_size):
    try:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print(f"GPU {gpu_id} 不可用，使用CPU")
            return
        
        # 计算张量大小（同前）
        elements_per_mb = 262144
        total_elements = int(memory_size * elements_per_mb)
        matrix_size = int(total_elements **0.5)
        while matrix_size * matrix_size > total_elements:
            matrix_size -= 1
        
        # 在GPU上创建随机矩阵
        print(f"GPU {gpu_id}: 创建 {matrix_size}x{matrix_size} 矩阵，约占用 {memory_size}MB 显存")
        mat = torch.randn(matrix_size, matrix_size, device=device)
        
        # 记录开始时间
        start_time = time.time()
        iterations = 0
        
        # 无限循环进行低强度运算（关键修改：添加torch.no_grad()）
        with torch.no_grad():  # 禁用自动梯度计算
            while True:
                # 简单的矩阵运算
                mat = (mat @ mat) * 0.001 + mat * 0.999
                iterations += 1
                
                if iterations % 2 == 0:
                    time.sleep(0.5)
                    if iterations % 10000 == 0:
                        elapsed = time.time() - start_time
                        hours, remainder = divmod(int(elapsed), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id}: "
                            f"已运行 {hours}h{minutes}m{seconds}s, 迭代 {iterations}次")
    
    except Exception as e:
        print(f"GPU {gpu_id} 发生错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='持续占用指定大小的GPU显存并进行低强度运算（直到手动终止）')
    parser.add_argument('--memory', type=int, default=5134, help='每个GPU要占用的显存大小(MB)')
    parser.add_argument('--gpus', type=str, default='6,7', help='指定使用的GPU，用逗号分隔，如"0,1"，默认使用所有可用GPU')
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("没有可用的GPU，程序退出")
        return

    # 确定要使用的GPU
    if args.gpus == 'all':
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(id) for id in args.gpus.split(',')]
    
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"将使用的GPU: {gpu_ids}")
    print(f"每个GPU计划占用显存: {args.memory}MB")
    print("程序将持续运行，按Ctrl+C终止...")
    
    # 为每个GPU启动一个线程
    import threading
    threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, args.memory)
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # 延迟启动，避免资源竞争
    
    # 等待所有线程完成（实际上会一直运行直到中断）
    for thread in threads:
        thread.join()
    
    print("所有GPU任务已终止")

if __name__ == "__main__":
    main()
    