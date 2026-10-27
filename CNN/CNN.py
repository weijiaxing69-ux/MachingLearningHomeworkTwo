import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        #TODO 卷积 -> Relu -> 池化 -> 全连接
        super().__init__()
        #TODO 现在开始布置卷积层
        self.feature=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            #TODO 随机丢弃一些数据 增强泛化能力
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        #TODO 对数据进行卷积池化 然后返回数据
        x=self.feature(x)
        x=self.classifier(x)
        return x

