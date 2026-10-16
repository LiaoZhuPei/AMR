# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


def per_sample_normalize(signal_data):
    """
    对单个样本进行归一化，将其缩放到 [-1, 1] 区间。
    输入 shape: (2, N)
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    # 归一化到 [-1, 1]
    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data


class IQDataset(Dataset):
    def __init__(self, iq_data, labels, apply_normalization=True):
        """
        Args:
            iq_data (numpy.ndarray): 原始 I/Q 数据, shape (N, 2, Signal_Length).
            labels (numpy.ndarray): 标签数据.
            apply_normalization (bool): 是否应用逐样本归一化.
        """
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.apply_normalization = apply_normalization

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]

        if self.apply_normalization:
            sample = per_sample_normalize(sample)

        # 转换为 Tensor
        sample_tensor = torch.from_numpy(sample).float()

        # 调整维度以匹配 LSTM 输入 (N, L, C) -> (Signal_Length, 2)
        sample_tensor = sample_tensor.permute(1, 0)

        return sample_tensor, self.labels[idx]