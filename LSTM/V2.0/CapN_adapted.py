# 文件名: CapN.py (或 CapN_adapted.py)

import torch
import torch.nn as nn

class CapsuleNetworkAdapted(nn.Module):
    def __init__(self, input_channels=2, input_length=1024, num_classes=12, dropout_rate=0.5):
        """
        为短信号 (长度1024) 优化的胶囊网络。
        - 采用更温和的降采样策略 (小卷积核 + Max Pooling)。
        - 内置 Dropout 以对抗过拟合。
        """
        super(CapsuleNetworkAdapted, self).__init__()
        self.num_classes = num_classes

        # --- 1. 特征提取前端 (更温和的降采样) ---
        # 使用两级 Conv-Pool 结构逐步缩小尺寸
        self.feature_extractor = nn.Sequential(
            # 阶段 1: 1024 -> 512
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度减半
            nn.Dropout(0.25),

            # 阶段 2: 512 -> 256
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度再次减半
            nn.Dropout(0.25),
        )
        # 经过此模块后，输出形状为: (batch, 128, 256)

        # 辅助函数，动态计算进入全连接层之前的展平尺寸
        dummy_input = torch.randn(1, input_channels, input_length)
        self._flattened_size = self._get_flattened_size(dummy_input)

        # --- 2. 并行分支 (每个分支更简单) ---
        self.branches = nn.ModuleList()
        for _ in range(self.num_classes):
            branch = nn.Sequential(
                # 分支内的卷积层，进一步提取特征和降采样
                # 256 -> 128
                nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                # 128 -> 64
                nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                
                # 展平并进入全连接层
                nn.Flatten(),
                nn.Linear(in_features=self._flattened_size, out_features=32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                # *** 关键：添加 Dropout 层以对抗过拟合 ***
                nn.Dropout(dropout_rate),
                
                nn.Linear(in_features=32, out_features=1)
            )
            self.branches.append(branch)

        # --- 3. 分类层 ---
        self.softmax = nn.Softmax(dim=1)

    def _get_flattened_size(self, dummy_input):
        """
        辅助函数，精确计算卷积/池化部分的最终输出维度。
        """
        x = self.feature_extractor(dummy_input)
        
        # 定义一个临时的分支卷积部分，用于尺寸计算
        branch_conv_part = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        output = branch_conv_part(x)
        return output.shape[1]

    def forward(self, x):
        features = self.feature_extractor(x)
        branch_outputs = [branch(features) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        output = self.softmax(concatenated)
        return output

def build_model(input_length=1024, num_classes=12, dropout_rate=0.5):
    """
    构建并返回一个适应短信号的 CapsuleNetworkAdapted 模型实例。
    """
    model = CapsuleNetworkAdapted(
        input_length=input_length, 
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model