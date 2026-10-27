import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def normalize(x,mean,std):
    x=x.reshape(-1,3,32,32).astype(np.float32) / 255.0
    x=(x-mean[None,:,None,None]) / std[None,:,None,None]
    return x
#TODO 将数据 归一化-> tensor -> DataLoader

def data_process(train_x, train_y, val_x=None, val_y=None,
                 test_x=None, test_y=None, batch_size=128):
    """标准化数据并创建训练、验证、测试 DataLoader。

    val_x 和 val_y 可以省略，适合使用全部训练数据进行训练的实验。
    """

    if test_x is None or test_y is None:
        raise ValueError("test_x 和 test_y 不能为空")

    #TODO 首先是标准化 并且计算出平均值
    train_x_f=train_x.reshape(-1,3,32,32).astype(np.float32)/255.0
    mean=train_x_f.mean(axis=(0,2,3))
    std=train_x_f.std(axis=(0,2,3))
    std=np.maximum(std, 1e-8)

    #TODO numpy->tensor
    def to_tensor(x,y):
        x=torch.from_numpy(normalize(x,mean,std))
        y=torch.from_numpy(y).long()
        return TensorDataset(x,y)

    train_ds=to_tensor(train_x,train_y)
    test_ds=to_tensor(test_x,test_y)

    #TODO DataLoader

    train_dataloader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_dataloader = None
    if val_x is not None and val_y is not None:
        val_ds = to_tensor(val_x, val_y)
        val_dataloader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    test_dataloader=DataLoader(test_ds,batch_size=batch_size,shuffle=False)

    return  train_dataloader,val_dataloader,test_dataloader




