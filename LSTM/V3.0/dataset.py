# dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def _per_sample_normalize_single(signal_data):
    """私有辅助函数：对单个样本进行归一化 (此函数保持不变)"""
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    return 2 * (signal_data - min_val) / range_val - 1

# #############################################################################
# --- 修改点 3: 创建一个新的Dataset类以适配CNN的输入 ---
# #############################################################################
class IQImageDataset(Dataset):
    """
    为2D CNN准备数据的Dataset。
    输出形状为 (1, 2, 1024)，符合 (C, H, W) 格式。
    """
    def __init__(self, iq_data, labels, normalize=True):
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. 获取原始的单个样本 (shape: 2, 1024)
        sample = self.raw_data[idx]

        # 2. 如果需要，应用逐样本归一化
        if self.normalize:
            sample = _per_sample_normalize_single(sample)

        # 3. 转换为 PyTorch Tensor
        sample_tensor = torch.from_numpy(sample).float() # Shape: (2, 1024)

        # 4. 关键步骤：增加一个“通道”维度以匹配Conv2d的输入
        # 将 (H, W) -> (C, H, W)
        sample_tensor = sample_tensor.unsqueeze(0) # Shape: (1, 2, 1024)

        return sample_tensor, self.labels[idx]

def get_dataloaders(config):
    """
    主函数：加载所有数据，创建Dataloaders并返回
    """
    print("--- Loading and Preparing Data ---")
    
    # ... (数据加载部分完全不变) ...
    train_data_flat = pd.read_csv(config.DATA_PATHS['train_data'], header=None).values
    num_train_samples = train_data_flat.shape[0]
    train_raw = train_data_flat.reshape(num_train_samples, config.DATA_PARAMS['iq_channels'], config.DATA_PARAMS['signal_length'])
    train_labels_idx = pd.read_csv(config.DATA_PATHS['train_labels'], header=None).values.flatten()
    test_data_flat = pd.read_csv(config.DATA_PATHS['test_data'], header=None).values
    num_test_samples = test_data_flat.shape[0]
    test_raw = test_data_flat.reshape(num_test_samples, config.DATA_PARAMS['iq_channels'], config.DATA_PARAMS['signal_length'])
    test_labels_idx = pd.read_csv(config.DATA_PATHS['test_labels'], header=None).values.flatten()
    Z_test = pd.read_csv(config.DATA_PATHS['test_snr'], header=None).values
    Y_test_onehot = np.eye(len(config.CLASSES), dtype='uint8')[test_labels_idx]
    X_train, X_val, Y_train, Y_val = train_test_split(
        train_raw, train_labels_idx, test_size=config.DATA_PARAMS['val_split_ratio'], random_state=42, stratify=train_labels_idx
    )
    print("Data loaded and split successfully.")

    # #############################################################################
    # --- 修改点 4: 使用新的 IQImageDataset 类 ---
    # #############################################################################
    # 创建Dataset实例
    train_dataset = IQImageDataset(X_train, Y_train, normalize=True)
    val_dataset = IQImageDataset(X_val, Y_val, normalize=True)
    test_dataset = IQImageDataset(test_raw, test_labels_idx, normalize=True)
    
    # 创建DataLoader实例 (这部分不变)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_PARAMS['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN_PARAMS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.TRAIN_PARAMS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    print("Dataloaders for CNN created successfully.")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'Y_test_onehot': Y_test_onehot,
        'Z_test': Z_test
    }