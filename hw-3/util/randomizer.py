import torch
import numpy as np


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异，避免这种结果波动
    torch.backends.cudnn.deterministic = True
    # 设置 torch.backends.cudnn.benchmark=True
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
