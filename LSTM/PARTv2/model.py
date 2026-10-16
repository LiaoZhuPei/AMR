# model.py (REPLACEMENT VERSION)

import torch
import torch.nn as nn


# 我们使用一个稍微不同的结构来定义模型，功能完全一样
# 但这种写法可以帮助我们绕过任何潜在的未知问题。
class SimpleLSTMClassifier(nn.Module):
    """
    一个健壮的、用于序列分类的LSTM模型。
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 确保输入格式为 (batch, seq, feature)
        )

        # 定义一个 Dropout 层，用于正则化，防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 定义输出层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态为零
        # h0 shape: (num_layers, batch_size, hidden_size)
        # c0 shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # _: (h_n, c_n) a tuple of the final hidden and cell states
        out, _ = self.lstm(x, (h0, c0))

        # 我们只关心序列最后一个时间步的输出
        last_output = out[:, -1, :]

        # 应用 Dropout
        last_output = self.dropout(last_output)

        # 通过全连接层得到最终输出
        logits = self.fc(last_output)

        return logits


# build_model 函数也相应更新
def build_model(model_config, num_classes):
    """
    根据配置构建模型实例。
    """
    params = model_config['params']
    # 注意参数名称的对应关系
    return SimpleLSTMClassifier(
        input_size=params['input_features'],
        hidden_size=params['lstm_hidden_size'],
        num_layers=params['num_layers'],
        num_classes=num_classes
    )