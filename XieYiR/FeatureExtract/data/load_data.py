# data/load_data.py
import numpy as np
import pandas as pd
from config import Config

class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = None
        self.tags = None
        
    def load(self):
        """加载原始数据"""
        print("[INFO] 加载数据...")
        self.data = np.load(self.cfg.DATA_PATH, allow_pickle=True)  # (172800, 2048) complex
        self.tags = pd.read_csv(self.cfg.TAGS_PATH, sep=',', skipinitialspace=True)
        
        print(f"[INFO] 数据形状: {self.data.shape}")
        print(f"[INFO] 标签数量: {len(self.tags)}")
        print(f"[INFO] 模式类别: {self.tags['mode'].unique()}")
        print(f"[INFO] SNR范围: {sorted(self.tags['snr'].unique())}")
        return self
    
    def get_target_data(self, snr_list=None):
        """获取目标类别（mt63_1000）数据"""
        mask = self.tags['mode'] == self.cfg.TARGET_MODE
        if snr_list is not None:
            mask &= self.tags['snr'].isin(snr_list)
        
        indices = self.tags[mask].index.tolist()
        signals = self.data[indices]  # complex (N, 2048)
        print(f"[INFO] 目标类别样本数: {len(indices)}, SNR筛选: {snr_list}")
        return signals, indices
    
    def get_negative_data(self, snr_list=None, exclude_mode=None):
        """获取非目标类别数据（负样本/未知信号）"""
        exclude = exclude_mode or self.cfg.TARGET_MODE
        mask = self.tags['mode'] != exclude
        if snr_list is not None:
            mask &= self.tags['snr'].isin(snr_list)
        
        indices = self.tags[mask].index.tolist()
        signals = self.data[indices]
        labels  = self.tags.loc[indices, 'mode'].tolist()
        print(f"[INFO] 负样本数量: {len(indices)}")
        return signals, labels, indices
    
    def get_split(self, train_snr=None, test_snr=None):
        """
        构建训练/测试集
        训练集: 仅包含目标类别信号（单分类学习）
        测试集: 目标类别 + 所有非目标类别
        """
        train_snr = train_snr or self.cfg.TRAIN_SNR
        test_snr  = test_snr  or self.cfg.TEST_SNR
        
        # 训练集：仅目标类别
        X_train, _ = self.get_target_data(snr_list=train_snr)
        y_train = np.ones(len(X_train), dtype=int)   # +1 = Link11
        
        # 测试集：目标类别 + 负样本
        X_test_pos, _ = self.get_target_data(snr_list=test_snr)
        y_test_pos = np.ones(len(X_test_pos), dtype=int)
        
        X_test_neg, neg_modes, _ = self.get_negative_data(snr_list=test_snr)
        y_test_neg = -np.ones(len(X_test_neg), dtype=int)  # -1 = 非Link11
        
        X_test = np.vstack([X_test_pos, X_test_neg])
        y_test = np.concatenate([y_test_pos, y_test_neg])
        meta_modes = ([self.cfg.TARGET_MODE] * len(X_test_pos)) + neg_modes
        
        print(f"\n[INFO] 训练集大小: {len(X_train)}")
        print(f"[INFO] 测试集大小: {len(X_test)} "
              f"(正: {len(X_test_pos)}, 负: {len(X_test_neg)})")
        
        return X_train, y_train, X_test, y_test, meta_modes