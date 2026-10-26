import os
import pickle
import sys

import numpy as np

# ---- 让脚本无论从哪个目录启动，都能 import 到项目根目录下的 config 包 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config

def load_data():
    

