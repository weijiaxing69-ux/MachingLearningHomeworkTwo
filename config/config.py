import os

# 以本文件为锚点定位项目根目录（config/ 的上一级），
# 这样无论从哪个目录启动脚本，都能找到数据目录。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(PROJECT_ROOT, "cifar-10-batches-py")
