import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    一个处理 IQ 幅度/相位数据的双层LSTM模型。
    等效于 Keras 的 CuDNNLSTM(128) -> CuDNNLSTM(128) -> Dense 结构。
    """

    def __init__(self, input_features, lstm_hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        # 使用一个 num_layers=2 的 LSTM 模块，更高效
        # 它内部包含了两个堆叠的LSTM层
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入格式为 (N, seq_len, features)
            dropout=0.5,
            bidirectional=True
        )

        # 全连接分类层
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)
        # self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        定义前向传播路径。
        PyTorch DataLoader 会直接提供 (N, 1024, 2) 格式的数据。

        Args:
            x (torch.Tensor): 输入张量，形状为 (N, 1024, 2)
        """
        # LSTM的输入已经是正确的 (N, seq_len, features) 格式

        # out 的形状是 (N, 1024, 128)
        # 我们只需要最后一个时间步的输出
        out, _ = self.lstm(x)
        last_step_output = out[:, -1, :]  # -> (N, 128)

        # 通过分类器得到 Logits
        logits = self.classifier(last_step_output)

        return logits


def build_model(input_size, hidden_size, num_layers, classes):
    """构建函数，接受所有关键参数"""
    return LSTM(
        input_features=input_size,
        lstm_hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=classes
    )
