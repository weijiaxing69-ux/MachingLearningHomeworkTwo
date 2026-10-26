import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 让脚本无论从哪个目录启动, 都能 import 到项目根目录下的模块。
# sys.path[0] 默认是脚本所在目录(全连接神经网络/), 这里补上项目根目录。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

