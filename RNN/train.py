import time

import torch
import torch.nn as nn

from getdata import load_data
from process import data_process
from RNN import RNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回 (平均 loss, 准确率)。"""
    model.train()  # 切到训练模式：Dropout 生效、BatchNorm 用当前 batch 统计量

    # total_loss 累加的是"每个样本的 loss × 样本数"，最后除以总样本数才是平均 loss
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)  # 数据和标签放到同一设备(cpu/mps/cuda)

        optimizer.zero_grad()              # 清空上一步残留的梯度，否则梯度会累加
        logits = model(x)                  # 前向：得到 (batch, 10) 的原始分数
        loss = criterion(logits, y)        # CrossEntropyLoss 内部已做 softmax，传 logits
        loss.backward()                    # 反向传播：计算每个参数的梯度
        optimizer.step()                   # 用梯度更新参数

        # loss.item() 是当前 batch 的平均 loss，乘 batch 大小还原成总和
        total_loss += loss.item() * x.size(0)
        # argmax(1) 取每行最大分数的下标作为预测类别，与真实标签比较后求和
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)                 # 累加已处理的样本数

    return total_loss / total, correct / total  # 平均 loss、准确率


@torch.no_grad()  # 评估不需要梯度，关闭计算图可省显存、加速
def evaluate(model, loader, criterion, device):
    """评估模型，返回 (平均 loss, 准确率)。"""
    model.eval()  # 切到评估模式：Dropout 关闭、BatchNorm 用滑动统计量

    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        # 评估时不能 backward，只统计 loss 和准确率
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    # ---- 1. 选择设备：优先 GPU，其次 Apple 芯片的 MPS，最后 CPU ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # ---- 2. 读数据：5 个训练 batch 合并成 all_train_x，测试集单独保留 ----
    all_train_x, all_train_y, test_x, test_y = load_data()
    # all_train_x: (50000, 3072) uint8, all_train_y: (50000,)

    # ---- 3. 使用全部 50000 张训练图像，不再切分验证集 ----
    # 均值和标准差只根据全部训练集计算，测试集只用于最终评估。
    train_loader, _, test_loader = data_process(
        all_train_x, all_train_y,
        test_x=test_x, test_y=test_y, batch_size=128)

    # ---- 5. 建模型、损失函数、优化器 ----
    model = RNN().to(device)  # 模型参数也要放到同一设备
    criterion = nn.CrossEntropyLoss()  # 多分类标准损失
    optimizer = torch.optim.Adam(       # Adam 优化器
        model.parameters(), lr=1e-3,    # 学习率 1e-3
        weight_decay=1e-4)              # L2 正则，抑制过拟合

    # ---- 6. 训练循环 ----
    for epoch in range(1, 21):
        t0 = time.time()

        # 训练一轮
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)

        print(f"epoch {epoch:2d} | train {tr_loss:.4f}/{tr_acc:.4f} "
              f"| {time.time()-t0:.1f}s")

    # ---- 7. 用训练完成后的模型做最终测试 ----
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"test {test_loss:.4f}/{test_acc:.4f}")


if __name__ == "__main__":
    main()
