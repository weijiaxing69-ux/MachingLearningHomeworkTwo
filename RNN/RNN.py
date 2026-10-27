import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 每张图片看成长度为 32 的序列
        # 每一行有 3 * 32 = 96 个特征
        self.rnn = nn.GRU(
            input_size=96,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # x 原始形状:
        # (batch_size, 3, 32, 32)

        batch_size = x.size(0)

        # 变成:
        # (batch_size, 32, 3 * 32)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, 32, 96)

        # output: (batch_size, 32, 128)
        output, hidden = self.rnn(x)

        # 取最后一个时间步
        last_output = output[:, -1, :]

        logits = self.classifier(last_output)
        return logits