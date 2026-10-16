# train.py (完整代码，直接替换)

import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

# 从其他模块导入必要的类和函数
# 注意：我们不再需要 dataset.py 和 sklearn
import model as model_module
import utils


def create_dummy_data(num_samples, seq_len, num_classes=2, iq_channels=2):
    """创建一个简单的正弦/余弦虚拟数据集"""
    print(f"Creating a dummy dataset with {num_samples} samples...")
    X = np.zeros((num_samples, iq_channels, seq_len))
    Y = np.zeros(num_samples, dtype=int)

    time = np.linspace(0, 10 * np.pi, seq_len)

    for i in range(num_samples):
        label = i % num_classes
        Y[i] = label
        # 随机化相位和幅度以增加一点点难度
        phase = np.random.rand() * np.pi
        amplitude = 1.0 + np.random.rand() * 0.1

        if label == 0:  # 类别0: 正弦波
            X[i, 0, :] = amplitude * np.sin(time + phase)
            X[i, 1, :] = amplitude * np.sin(time + phase)  # I和Q相同
        elif label == 1:  # 类别1: 余弦波
            X[i, 0, :] = amplitude * np.cos(time + phase)
            X[i, 1, :] = amplitude * np.cos(time + phase)  # I和Q相同
        # 可以为更多类别添加其他简单波形...

    # 手动 permute 以匹配 LSTM 输入
    X_permuted = np.transpose(X, (0, 2, 1))  # Shape: (N, L, C)
    return torch.from_numpy(X_permuted).float(), torch.from_numpy(Y).long()


def main(config):
    """主函数，执行整个训练和评估流程"""
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config['seed'])

    # --- 关键修改：使用虚拟数据 ---
    # 我们不再加载真实数据，而是创建虚拟数据
    num_classes_dummy = 2  # 我们只创建2个简单的类别
    X_dummy, Y_dummy = create_dummy_data(num_samples=1280, seq_len=500, num_classes=num_classes_dummy)

    # 直接用TensorDataset创建数据集和加载器
    dummy_dataset = TensorDataset(X_dummy, Y_dummy)
    dummy_loader = DataLoader(dummy_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # --- 模型、损失函数、优化器定义 ---
    # 注意：num_classes 现在是虚拟数据的类别数
    model = model_module.build_model(config['model'], num_classes_dummy).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['optimizer']['lr'])

    # --- 在虚拟数据上进行训练测试 ---
    print("\n--- STARTING TRAINING ON DUMMY DATA ---")
    model.train()
    for epoch in range(20):  # 只训练20个epoch就足够看出趋势
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for inputs, labels in dummy_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"Epoch {epoch + 1:02d}, Loss: {avg_loss:.6f}, Accuracy: {avg_acc:.4f}")

    print("--- DUMMY DATA TRAINING FINISHED ---\n")

    if avg_acc > 0.95:
        print(">>> SUCCESS: Model successfully learned on the dummy data.")
        print(
            ">>> This strongly suggests the issue lies within your REAL DATA or its specific characteristics at length 500.")
    else:
        print(">>> FAILURE: Model FAILED to learn on the simple dummy data.")
        print(">>> This suggests a hidden bug in the model architecture (model.py) or the training loop itself.")

    sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PyTorch model based on a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)