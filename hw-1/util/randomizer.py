import torch
import numpy as np
from torch.utils.data import random_split


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


def train_valid_split(dataset, train_ratio, seed):
    train_size = int(len(dataset) * train_ratio)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size],
                                        generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set
