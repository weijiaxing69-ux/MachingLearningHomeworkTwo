import os
import pickle
import sys

import numpy as np

# ---- 让脚本无论从哪个目录启动，都能 import 到项目根目录下的 config 包 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config

TRAIN_FILES = ["data_batch_1", "data_batch_2", "data_batch_3",
               "data_batch_4", "data_batch_5"]


def load_batch(filename):
    """读取单个 pickle 文件，返回 (data, labels)。

    data:   uint8, (10000, 3072)，每行一张图，通道优先拉平
    labels: int64, (10000,)
    """
    with open(os.path.join(config.data_path, filename), "rb") as fh:
        batch = pickle.load(fh, encoding="bytes")  # key 是 bytes
    return batch[b"data"], np.array(batch[b"labels"], dtype=np.int64)


def load_data():
    """把 5 个训练 batch 合并成一个，再单独读测试集。

    返回 (X_train, y_train, X_test, y_test)：
    X_train: uint8, (50000, 3072)
    y_train: int64, (50000,)
    X_test:  uint8, (10000, 3072)
    y_test:  int64, (10000,)
    """
    parts = [load_batch(f) for f in TRAIN_FILES]
    X_train = np.vstack([p[0] for p in parts])       # (5*10000, 3072)
    y_train = np.concatenate([p[1] for p in parts])  # (5*10000,)
    X_test, y_test = load_batch("test_batch")
    return X_train, y_train, X_test, y_test


def split_train_val(X, y, val_size=5000, seed=42):
    """从合并后的训练集里随机切出验证集。
    
    返回 (X_train, y_train, X_val, y_val)。
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))  # 打乱下标
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
