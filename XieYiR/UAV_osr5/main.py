"""
SDAE-LOF 无人机射频信号异常检测
========================================
第一级：SDAE(去噪自编码器) + LOF(局部离群因子) 做异常检测
第二级（可选扩展）：对被判定为无人机的样本做型号分类
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import joblib


# =========================================================
# 0. 全局配置
# =========================================================
class Config:
    DATA_ROOT = "./data_root"          # 数据集根目录
    INPUT_DIM = 1024                   # 信号维度
    LATENT_DIM = 32                    # 隐表征维度
    HIDDEN_DIMS = [512, 128]           # 编码器隐藏层（解码器对称使用）
    SNR_DB = 30                       # AWGN 信噪比(dB)，可调
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3
    LOF_K = 20                         # LOF 近邻数 k
    LOF_METRIC = "minkowski"           # 距离度量方式
    TEST_SIZE = 0.2                    # 正常样本中划出多少做测试
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 三类信号对应的文件夹
    WIFI_DIRS = ["TPLINK_TL_WR940N", "CISCO_LINKSYS_E3200"]
    BT_DIRS = ["APPLE_IPAD3", "APPLE_IPHONE6S", "APPLE_IPHONE7",
               "FITBIT_CHARGE3", "MOTOROLA"]
    DRONE_DIRS = ["DJI_M600", "BEEBEERUN", "DJI_INSPIRE",
                  "DJI_MAVICPRO", "DJI_PHANTOM"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. 数据加载  (丢弃第1025列，用文件夹名作为类别标签)
# =========================================================
def load_folder_csv(folder_path):
    """读取单个设备文件夹下所有csv，去掉最后一列(逐样本字符串标签)"""
    all_rows = []
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if len(csv_files) == 0:
        print(f"[警告] 文件夹 {folder_path} 未找到 csv 文件")
    for f in csv_files:
        # 若csv包含表头，请改为 header=0
        df = pd.read_csv(f, header=None)
        data = df.iloc[:, :-1].values.astype(np.float32)   # 丢弃第1025列
        all_rows.append(data)
    if len(all_rows) == 0:
        return np.empty((0, 1024), dtype=np.float32)
    return np.concatenate(all_rows, axis=0)


def load_dataset(data_root, dir_list):
    """加载给定设备文件夹列表，合并返回 (X, y_device)"""
    X_list, label_list = [], []
    for d in dir_list:
        folder = os.path.join(data_root, d)
        X = load_folder_csv(folder)
        if X.shape[0] == 0:
            continue
        X_list.append(X)
        label_list.extend([d] * X.shape[0])   # 文件夹名 = 类别标签
    if len(X_list) == 0:
        return np.empty((0, 1024), dtype=np.float32), np.array([])
    return np.concatenate(X_list, axis=0), np.array(label_list)


# =========================================================
# 2. AWGN 噪声 & 全局 min-max 归一化 (对应公式(2))
# =========================================================
def add_awgn(signal, snr_db):
    """按指定信噪比对信号逐样本添加高斯白噪声"""
    sig_power = np.mean(signal ** 2, axis=1, keepdims=True)
    sig_power = np.maximum(sig_power, 1e-12)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, 1, size=signal.shape) * np.sqrt(noise_power)
    return signal + noise


class GlobalMinMaxScaler:
    """公式(2)：使用训练集整体的标量 min / max 做归一化"""
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        self.min_ = float(np.min(X))
        self.max_ = float(np.max(X))
        return self

    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_ + 1e-12)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# =========================================================
# 3. SDAE 网络结构 (2.1节)
# =========================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-1], latent_dim), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [latent_dim] + hidden_dims[::-1]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-1], output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class SDAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def forward(self, x_noisy):
        h = self.encoder(x_noisy)
        u = self.decoder(h)
        return u, h


# =========================================================
# 4. SDAE 训练  (阶段一)
# =========================================================
def train_sdae(model, clean_train, cfg):
    """
    clean_train: 原始干净训练数据 (仅 WiFi+蓝牙, 未加噪, 未归一化)
    每个 epoch 重新随机采样一次 AWGN，实现"随机叠加噪声"的效果
    """
    scaler = GlobalMinMaxScaler().fit(clean_train)
    m_clean = scaler.transform(clean_train)                 # 重构目标，训练全程不变
    clean_tensor = torch.tensor(m_clean, dtype=torch.float32)

    model.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(cfg.EPOCHS):
        # 每个epoch重新生成带噪样本 x́
        noisy_train = add_awgn(clean_train, cfg.SNR_DB)
        m_noisy = np.clip(scaler.transform(noisy_train), 0.0, 1.0)
        noisy_tensor = torch.tensor(m_noisy, dtype=torch.float32)

        dataset = TensorDataset(noisy_tensor, clean_tensor)
        loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

        epoch_loss = 0.0
        for noisy_batch, clean_batch in loader:
            noisy_batch = noisy_batch.to(cfg.DEVICE)
            clean_batch = clean_batch.to(cfg.DEVICE)

            optimizer.zero_grad()
            recon, _ = model(noisy_batch)
            loss = criterion(recon, clean_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * noisy_batch.size(0)

        epoch_loss /= len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[SDAE] Epoch {epoch+1}/{cfg.EPOCHS}  重构损失: {epoch_loss:.6f}")

    return model, scaler


# =========================================================
# 5. LOF 训练  (阶段二)
# =========================================================
def encode_data(encoder, X, scaler, cfg):
    """使用训练好的编码器压缩数据 (无噪声输入)"""
    m = np.clip(scaler.transform(X), 0.0, 1.0)
    encoder.eval()
    with torch.no_grad():
        h = encoder(torch.tensor(m, dtype=torch.float32).to(cfg.DEVICE))
    return h.cpu().numpy()


def train_lof(encoder, clean_train, scaler, cfg):
    """
    使用编码器压缩【无噪声】训练数据，得到32维特征，训练 LOF (novelty=True 以支持对新样本 predict)
    """
    h_train = encode_data(encoder, clean_train, scaler, cfg)

    lof = LocalOutlierFactor(
        n_neighbors=cfg.LOF_K,
        metric=cfg.LOF_METRIC,
        novelty=True,          # 允许对训练集之外的新样本做 predict
        contamination="auto",
    )
    lof.fit(h_train)
    return lof, h_train


def detect_anomaly(lof, h):
    """
    pred:  1 = 正常(干扰信号,inlier)，-1 = 异常(无人机,outlier)  —— sklearn 约定
    score: 值越大表示越可能是异常（近似对应 LOF(p) 越大于1）
    """
    pred = lof.predict(h)                 # -1 / 1
    score = -lof.score_samples(h)         # score_samples 越小越异常，取负后越大越异常
    return pred, score


# =========================================================
# 6. 评估
# =========================================================
def evaluate(y_true_binary, pred, scores):
    """
    y_true_binary / pred : 1=正常(inlier)，-1=无人机异常(outlier)
    """
    y_true01 = (y_true_binary == -1).astype(int)   # 异常=1，正常=0
    y_pred01 = (pred == -1).astype(int)

    acc = accuracy_score(y_true01, y_pred01)
    prec = precision_score(y_true01, y_pred01, zero_division=0)
    rec = recall_score(y_true01, y_pred01, zero_division=0)
    f1 = f1_score(y_true01, y_pred01, zero_division=0)
    try:
        auc = roc_auc_score(y_true01, scores)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true01, y_pred01)

    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("混淆矩阵 [0=正常干扰, 1=无人机异常] (行=真实, 列=预测)")
    print(cm)
    print("=" * 50)

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "confusion_matrix": cm}


# =========================================================
# 7. (可选扩展) 第二级：无人机型号分类
# =========================================================
def train_drone_classifier(encoder, X_drone, y_drone, scaler, cfg):
    """
    仅对被判定为异常(无人机)的样本使用；用编码后的32维特征训练一个简单分类器
    （具体网络结构文档未给出，这里用 RandomForest 作为示例，可替换为 MLP 等）
    """
    h_drone = encode_data(encoder, X_drone, scaler, cfg)
    h_train, h_test, y_train, y_test = train_test_split(
        h_drone, y_drone, test_size=0.2, random_state=cfg.RANDOM_STATE, stratify=y_drone
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=cfg.RANDOM_STATE)
    clf.fit(h_train, y_train)
    acc = clf.score(h_test, y_test)
    print(f"[第二级-型号分类] 测试集准确率: {acc:.4f}")
    return clf


# =========================================================
# 8. 主流程
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.RANDOM_STATE)

    print("加载 WiFi 数据 ...")
    X_wifi, y_wifi = load_dataset(cfg.DATA_ROOT, cfg.WIFI_DIRS)
    print(f"  WiFi 样本数: {X_wifi.shape[0]}")

    print("加载 蓝牙 数据 ...")
    X_bt, y_bt = load_dataset(cfg.DATA_ROOT, cfg.BT_DIRS)
    print(f"  蓝牙样本数: {X_bt.shape[0]}")

    print("加载 无人机 数据 ...")
    X_drone, y_drone = load_dataset(cfg.DATA_ROOT, cfg.DRONE_DIRS)
    print(f"  无人机样本数: {X_drone.shape[0]}")

    # 正常样本 = WiFi + 蓝牙（唯一可用于训练的数据）
    X_normal = np.concatenate([X_wifi, X_bt], axis=0)

    # 正常样本划分 训练 / 测试
    X_norm_train, X_norm_test = train_test_split(
        X_normal, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, shuffle=True
    )
    print(f"正常训练样本: {X_norm_train.shape[0]}, 正常测试样本: {X_norm_test.shape[0]}")
    print(f"无人机(全部作为测试期异常样本): {X_drone.shape[0]}")

    # ---------- 阶段一: 训练 SDAE ----------
    model = SDAE(cfg.INPUT_DIM, cfg.HIDDEN_DIMS, cfg.LATENT_DIM)
    model, scaler = train_sdae(model, X_norm_train, cfg)

    # ---------- 阶段二: 训练 LOF ----------
    lof, h_train = train_lof(model.encoder, X_norm_train, scaler, cfg)

    # ---------- 测试推理: 正常测试集 + 全部无人机 ----------
    X_test_all = np.concatenate([X_norm_test, X_drone], axis=0)
    y_test_binary = np.concatenate([
        np.ones(X_norm_test.shape[0]),     # 正常 -> 1 (inlier)
        -np.ones(X_drone.shape[0]),        # 无人机 -> -1 (outlier)
    ])

    h_test = encode_data(model.encoder, X_test_all, scaler, cfg)
    pred, scores = detect_anomaly(lof, h_test)
    metrics = evaluate(y_test_binary, pred, scores)

    # ---------- (可选) 第二级：对被判定为无人机的样本进一步分类 ----------
    if X_drone.shape[0] > 0:
        drone_clf = train_drone_classifier(model.encoder, X_drone, y_drone, scaler, cfg)
        joblib.dump(drone_clf, "drone_classifier.pkl")

    # ---------- 保存模型 ----------
    torch.save(model.state_dict(), "sdae_model.pth")
    joblib.dump({"lof": lof, "scaler": scaler}, "lof_scaler.pkl")

    return model, lof, scaler, metrics


if __name__ == "__main__":
    main()