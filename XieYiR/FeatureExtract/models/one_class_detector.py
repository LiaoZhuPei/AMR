# models/one_class_detector.py
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
from config import Config

class OneClassDetector:
    """
    开放集检测器：基于One-Class SVM + Isolation Forest集成
    
    训练策略:
    - 仅使用目标类别（mt63_1000）训练
    - 学习目标类别的特征分布边界
    - 对未知信号给出异常得分
    
    决策策略:
    - 两个模型得分标准化后加权融合
    - 基于训练集得分分布确定最优阈值
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # 预处理
        self.scaler = StandardScaler()
        self.pca    = PCA(n_components=0.95, random_state=cfg.SEED)  # 保留95%方差
        
        # 单分类模型
        self.ocsvm = OneClassSVM(
            kernel=cfg.OCSVM_KERNEL,
            nu=cfg.OCSVM_NU,
            gamma=cfg.OCSVM_GAMMA
        )
        self.iforest = IsolationForest(
            n_estimators=cfg.IF_N_ESTIMATORS,
            contamination=cfg.IF_CONTAMINATION,
            random_state=cfg.SEED,
            n_jobs=-1
        )
        
        # 阈值（在训练后确定）
        self.threshold_ocsvm   = None
        self.threshold_iforest = None
        self.threshold_fusion  = None
        
        # 训练集得分分布（用于归一化）
        self.train_scores_ocsvm   = None
        self.train_scores_iforest = None
        
        self.is_fitted = False
    
    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """标准化 + PCA降维"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            X_pca    = self.pca.fit_transform(X_scaled)
            print(f"[INFO] PCA降维: {X.shape[1]} → {X_pca.shape[1]} 维")
        else:
            X_scaled = self.scaler.transform(X)
            X_pca    = self.pca.transform(X_scaled)
        return X_pca
    
    def fit(self, X_train: np.ndarray):
        """
        训练单分类模型
        X_train: (N, n_features) 仅包含目标类别特征
        """
        print("\n[TRAIN] 预处理...")
        X_proc = self._preprocess(X_train, fit=True)
        
        print(f"[TRAIN] 训练 One-Class SVM (nu={self.cfg.OCSVM_NU})...")
        self.ocsvm.fit(X_proc)
        
        print(f"[TRAIN] 训练 Isolation Forest "
              f"(n_est={self.cfg.IF_N_ESTIMATORS})...")
        self.iforest.fit(X_proc)
        
        # 计算训练集得分，用于确定阈值
        print("[TRAIN] 计算训练集异常得分...")
        self.train_scores_ocsvm   = self.ocsvm.score_samples(X_proc)
        self.train_scores_iforest = self.iforest.score_samples(X_proc)
        
        # 使用百分位数确定阈值（允许少量误报）
        p = self.cfg.THRESHOLD_PERCENTILE
        self.threshold_ocsvm   = np.percentile(self.train_scores_ocsvm, p)
        self.threshold_iforest = np.percentile(self.train_scores_iforest, p)
        
        print(f"[TRAIN] OCSVM 阈值 (p={p}): {self.threshold_ocsvm:.4f}")
        print(f"[TRAIN] IForest 阈值 (p={p}): {self.threshold_iforest:.4f}")
        
        self.is_fitted = True
        return self
    
    def _normalize_scores(self, scores: np.ndarray, 
                           train_scores: np.ndarray) -> np.ndarray:
        """
        将得分归一化到[0,1]
        基于训练集得分的min-max归一化
        得分越高 → 越像目标类别
        """
        s_min = train_scores.min()
        s_max = train_scores.max()
        return (scores - s_min) / (s_max - s_min + 1e-12)
    
    def score(self, X: np.ndarray) -> dict:
        """
        计算异常得分
        返回各模型得分及融合得分（得分越高 = 越像Link11）
        """
        assert self.is_fitted, "模型未训练！"
        X_proc = self._preprocess(X, fit=False)
        
        raw_ocsvm   = self.ocsvm.score_samples(X_proc)
        raw_iforest = self.iforest.score_samples(X_proc)
        
        # 归一化得分
        norm_ocsvm   = self._normalize_scores(raw_ocsvm,   
                                               self.train_scores_ocsvm)
        norm_iforest = self._normalize_scores(raw_iforest, 
                                               self.train_scores_iforest)
        
        # 加权融合（可调整权重）
        fusion = 0.7 * norm_ocsvm + 0.3 * norm_iforest
        
        return {
            "ocsvm_raw":   raw_ocsvm,
            "iforest_raw": raw_iforest,
            "ocsvm_norm":  norm_ocsvm,
            "iforest_norm": norm_iforest,
            "fusion":      fusion
        }
    
    def predict(self, X: np.ndarray, 
                method: str = "fusion") -> np.ndarray:
        """
        预测是否为目标类别
        返回: +1 (Link11) / -1 (Unknown)
        method: "ocsvm" | "iforest" | "fusion"
        """
        scores_dict = self.score(X)
        X_proc = self._preprocess(X, fit=False)
        
        if method == "ocsvm":
            raw = scores_dict["ocsvm_raw"]
            threshold = self.threshold_ocsvm
            preds = np.where(raw >= threshold, 1, -1)
            
        elif method == "iforest":
            raw = scores_dict["iforest_raw"]
            threshold = self.threshold_iforest
            preds = np.where(raw >= threshold, 1, -1)
            
        elif method == "fusion":
            # 融合得分的阈值（归一化后，训练集的5th percentile）
            train_ocsvm_norm   = self._normalize_scores(
                self.train_scores_ocsvm, self.train_scores_ocsvm)
            train_iforest_norm = self._normalize_scores(
                self.train_scores_iforest, self.train_scores_iforest)
            train_fusion = 0.7 * train_ocsvm_norm + 0.3 * train_iforest_norm
            
            threshold = np.percentile(train_fusion, self.cfg.THRESHOLD_PERCENTILE)
            fusion    = scores_dict["fusion"]
            preds     = np.where(fusion >= threshold, 1, -1)
            
            if self.threshold_fusion is None:
                self.threshold_fusion = threshold
        else:
            raise ValueError(f"未知方法: {method}")
        
        return preds
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回归一化融合得分作为伪概率"""
        return self.score(X)["fusion"]
    
    def save(self, path: str = None):
        """保存模型"""
        path = path or self.cfg.MODEL_SAVE_PATH
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "detector.pkl"))
        print(f"[INFO] 模型已保存至 {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        return joblib.load(os.path.join(path, "detector.pkl"))