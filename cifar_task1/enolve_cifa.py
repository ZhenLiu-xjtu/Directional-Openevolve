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


import os, pickle, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入 tqdm
import random  # 导入 random 模块

# -------------------- 配置 --------------------
DATA_DIR   = r'/data/lz/openevolve/dataset/cifar-10-batches-py'
SEED       = 42
MAX_TRAIN  = 2000   # 训练子集样本数（如 2000），None 表示用全量
MAX_TEST   = 1000   # 测试子集样本数（如 1000），None 表示用全量
BATCH_SIZE = 128
EPOCHS     = 300
LR         = 0.01
SAVE_DIR   = './saved_models'  # 保存最优模型的路径

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存目录
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -------------------- 数据加载 --------------------
def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].astype(np.float32) / 255.0       # [N, 3072] → 归一化
    labels = np.array(batch[b'labels'], dtype=np.int64)    # [N]
    data = data.reshape(-1, 3, 32, 32)                     # [N, 3, 32, 32]
    return torch.from_numpy(data), torch.from_numpy(labels)

def load_cifar10_data(data_dir, max_train=None, max_test=None):
    # 训练集
    trX, trY = [], []
    for i in range(1, 6):
        x, y = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}"))
        trX.append(x); trY.append(y)
    trX = torch.cat(trX, 0); trY = torch.cat(trY, 0)

    # 测试集
    teX, teY = load_cifar10_batch(os.path.join(data_dir, "test_batch"))

    # 随机下采样子集（不分层，够用来快速验证）
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

trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
testloader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------- 线性层（循环实现） --------------------
class LinearLoopLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias    = nn.Parameter(torch.randn(out_features))

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

# -------------------- 进化算法 --------------------

class Island:
    def __init__(self, model, mutate_prob=0.1):
        self.model = model
        self.mutate_prob = mutate_prob
        self.best_model = model  # 记录当前最优模型

    def mutate(self):
        # 变异策略：例如修改权重初始化方式
        mutation_type = random.choice(["weight_init", "sparsity", "low_rank", "vectorization"])
        if mutation_type == "weight_init":
            self.model.weights.data = torch.randn_like(self.model.weights.data)  # 重新初始化权重
        elif mutation_type == "sparsity":
            self.model.weights.data *= torch.rand_like(self.model.weights.data) < 0.5  # 随机稀疏化
        elif mutation_type == "low_rank":
            U, S, V = torch.svd(self.model.weights.data)
            self.model.weights.data = U[:, :self.model.out_features] @ torch.diag(S[:self.model.out_features]) @ V.t()  # 低秩近似
        elif mutation_type == "vectorization":
            self.model.forward = self.vectorized_forward

    def vectorized_forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平输入
        return torch.matmul(x, self.model.weights.t()) + self.model.bias  # 向量化的计算

    def save_best_model(self, epoch):
        # 保存最优模型（weights 和 bias）
        best_model_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), best_model_path)
        print(f"Saved best model for epoch {epoch} to {best_model_path}")

# -------------------- 群岛算法的进化 --------------------
class EvolutionarySearch:
    def __init__(self, population_size, mutation_rate, generations, model, trainloader, testloader, criterion, optimizer):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.islands = [Island(model, mutation_rate) for _ in range(population_size)]  # 初始岛群

    def evaluate(self):
        # 直接评估当前岛屿的模型
        accuracy = evaluate(self.model, self.testloader, device)
        return accuracy

    def select(self):
        # 选择适应度最好的岛屿
        scores = [self.evaluate() for island in self.islands]
        best_islands = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.population_size // 2]
        return [self.islands[i] for i in best_islands]

    def evolve(self):
        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")

            # 选择适应度最好的个体
            selected_islands = self.select()

            # 变异（仅对选中的个体进行变异）
            for island in selected_islands:
                if random.random() < self.mutation_rate:
                    island.mutate()

            # 将选中的个体更新到种群中
            self.islands = selected_islands + [Island(self.model) for _ in range(self.population_size - len(selected_islands))]

            # 评估当前最好的个体并保存
            best_island = self.islands[0]
            best_island.save_best_model(generation + 1)  # 保存最优模型
            accuracy = self.evaluate()
            print(f"Best Accuracy: {accuracy:.2f}%")

            # 在每代显示进度条
            tqdm.write(f"Generation {generation + 1}/{self.generations} completed. Best Accuracy: {accuracy:.2f}%")
            return accuracy

# -------------------- 训练 --------------------
model = nn.Sequential(
    LinearLoopLayer(3072, 256),
    nn.ReLU(),
    LinearLoopLayer(256, 10)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建进化搜索实例
evolution = EvolutionarySearch(population_size=10, mutation_rate=0.1, generations=10, model=model, trainloader=trainloader, testloader=testloader, criterion=criterion, optimizer=optimizer)

# 训练并进行群岛进化
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_correct, running_total = 0, 0
    # 使用 tqdm 显示进度条
    for x, y in tqdm(trainloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch"):
        x, y = x.to(device), y.to(device)
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
    print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # 在每个 epoch 后进行群岛进化
    evolution.evolve()

final_acc = evaluate(model, testloader, device)
print(f"Final Test Accuracy (subset): {final_acc:.2f}%")
