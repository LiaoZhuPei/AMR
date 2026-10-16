# dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def _per_sample_normalize_single(signal_data):
    """私有辅助函数：对单个样本进行归一化"""
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    return 2 * (signal_data - min_val) / range_val - 1

class IQFeatureDataset(Dataset):
    """自定义Dataset类，负责特征工程"""
    def __init__(self, iq_data, labels, normalize_iq_amp=True):
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.normalize_iq_amp = normalize_iq_amp

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        iq_sample_raw = self.raw_data[idx]
        
        i_data_raw = iq_sample_raw[0, :]
        q_data_raw = iq_sample_raw[1, :]

        amplitude_raw = np.sqrt(i_data_raw**2 + q_data_raw**2)
        phase = np.arctan2(q_data_raw, i_data_raw)
        
        delta_phase = np.diff(phase, prepend=phase[0]) 
        delta_phase[delta_phase > np.pi] -= 2 * np.pi
        delta_phase[delta_phase < -np.pi] += 2 * np.pi
        
        if self.normalize_iq_amp:
            iq_normalized = _per_sample_normalize_single(iq_sample_raw)
            i_data_norm = iq_normalized[0, :]
            q_data_norm = iq_normalized[1, :]
            amplitude_norm = _per_sample_normalize_single(amplitude_raw)
        else:
            i_data_norm, q_data_norm, amplitude_norm = i_data_raw, q_data_raw, amplitude_raw

        features = np.stack([
            i_data_norm, q_data_norm, amplitude_norm, phase, delta_phase
        ], axis=0)
        
        features_tensor = torch.from_numpy(features).float().permute(1, 0)
        return features_tensor, self.labels[idx]

def get_dataloaders(config):
    """
    主函数：加载所有数据，创建Dataloaders并返回
    """
    print("--- Loading and Preparing Data ---")
    
    # 加载训练数据
    train_data_flat = pd.read_csv(config.DATA_PATHS['train_data'], header=None).values
    num_train_samples = train_data_flat.shape[0]
    train_raw = train_data_flat.reshape(num_train_samples, config.DATA_PARAMS['iq_channels'], config.DATA_PARAMS['signal_length'])
    train_labels_idx = pd.read_csv(config.DATA_PATHS['train_labels'], header=None).values.flatten()
    print(f"Loaded {num_train_samples} training samples.")

    # 加载测试数据
    test_data_flat = pd.read_csv(config.DATA_PATHS['test_data'], header=None).values
    num_test_samples = test_data_flat.shape[0]
    test_raw = test_data_flat.reshape(num_test_samples, config.DATA_PARAMS['iq_channels'], config.DATA_PARAMS['signal_length'])
    test_labels_idx = pd.read_csv(config.DATA_PATHS['test_labels'], header=None).values.flatten()
    Z_test = pd.read_csv(config.DATA_PATHS['test_snr'], header=None).values
    Y_test_onehot = np.eye(len(config.CLASSES), dtype='uint8')[test_labels_idx]
    print(f"Loaded {num_test_samples} test samples.")

    # 划分训练集和验证集
    X_train, X_val, Y_train, Y_val = train_test_split(
        train_raw, train_labels_idx, test_size=config.DATA_PARAMS['val_split_ratio'], random_state=42, stratify=train_labels_idx
    )
    print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # 创建Dataset实例
    train_dataset = IQFeatureDataset(X_train, Y_train, normalize_iq_amp=True)
    val_dataset = IQFeatureDataset(X_val, Y_val, normalize_iq_amp=True)
    test_dataset = IQFeatureDataset(test_raw, test_labels_idx, normalize_iq_amp=True)
    
    # 创建DataLoader实例
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_PARAMS['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN_PARAMS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.TRAIN_PARAMS['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    print("Dataloaders created successfully.")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'Y_test_onehot': Y_test_onehot,
        'Z_test': Z_test
    }