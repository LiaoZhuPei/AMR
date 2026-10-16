"""
 LSTM 的核心思想是按时间顺序处理序列数据。在每个时间步 t，LSTM 单元接收当前的输入 x_t 和上一个时间步的隐藏状态 h_{t-1}。
 通过内部的门控机制（遗忘门、输入门、输出门），它会更新其细胞状态（长期记忆）并生成一个新的隐藏状态 h_t（短期记忆）。
 这个新的隐藏状态 h_t 既包含了当前时间步输入的信息，也融合了之前所有时间步的上下文信息。

"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    一个处理 IQ 幅度/相位数据的4层LSTM模型。
    """

    def __init__(self, input_features=2, lstm_hidden_size=128, num_classes=14):
        super(LSTM, self).__init__()

        # 使用一个 num_layers=2 的 LSTM 模块，更高效
        # 它内部包含了两个堆叠的LSTM层
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden_size,
            num_layers=4,
            batch_first=True,  # 输入格式为 (N, seq_len, features)
            #dropout=0.5,
            #bidirectional=True  #True为双向开启
        )

        # 全连接分类层
        # self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)   #双向LSTM
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播
        DataLoader提供 (N, 1024, 2) 格式的数据

        Args:
            x (torch.Tensor): 输入张量，形状为 (N, 1024, 2)
        """
        # LSTM的输入(N, seq_len, features) 格式

        # out 的形状是 (N, 1024, 128)
        # 只需要最后一个时间步的输出
        out, _ = self.lstm(x)
        last_step_output = out[:, -1, :]  # -> (N, 128)

        # 通过分类器得到 Logits
        logits = self.classifier(last_step_output)

        return logits


# 构建函数
def build_model(classes):
    return LSTM(num_classes=classes)
