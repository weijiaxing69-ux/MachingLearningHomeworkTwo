#TODO 在这里面来进行对于数据shape的判断

import pickle, os
import numpy as np
import config.config as config
d = config.data_path

for f in ["data_batch_1", "data_batch_2", "data_batch_3",
          "data_batch_4", "data_batch_5", "test_batch"]:
    with open(os.path.join(d, f), "rb") as fh:
        b = pickle.load(fh, encoding="bytes")   # 注意 encoding='bytes'
    print(f, {k: getattr(v, "shape", len(v)) for k, v in b.items()})


# 2. 每个 batch 的 shape
# 每个 batch 是 pickle 序列化的 dict，key 是 bytes（所以必须用 b'data' 这种写法）：
#
# key	类型	shape / 长度	说明
# data	ndarray uint8	(10000, 3072)	每行一张图，3072 = 3×32×32
# labels	list[int]	10000	取值 0–9
# filenames	list[bytes]	10000	原始文件名
# batch_label	bytes	21	如 training batch 1 of 5
# batches.meta：num_cases_per_batch=10000，num_vis=3072，label_names 10 个类别
# （airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck）。
#
# 数据取值 min=0, max=255，uint8。
#
# 整图关键 shape：(10000, 3072) → 单张 3072 按 通道优先 排布（前 1024 = R，中 1024 = G，后 1024 = B）→ reshape(3, 32, 32)。
# 合并 5 个训练 batch 后是 (50000, 3072)，测试集 (10000, 3072)。