import torch
import torch.nn as nn


class Attention(nn.Module):
    """时间维度自注意力：学习每个时间步的重要性权重"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden)
        attn_weights = self.attention(lstm_output)          # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)   # 归一化
        context = torch.sum(lstm_output * attn_weights, dim=1)  # 加权求和 → (batch, hidden)
        return context


class CNN_BiLSTM_Attention(nn.Module):
    """
    CNN(降采样 1024→128) + BiLSTM + Attention + 分类头
    输入: (batch, 1024, 2)
    """

    def __init__(self, num_classes=9, cnn_channels=[64, 128, 128],
                 hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        # === CNN 前端：Strided Conv 实现可学习的降采样 ===
        self.cnn = nn.Sequential(
            # Block 1: (batch, 2, 1024) → (batch, 64, 512)
            nn.Conv1d(2, cnn_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.GELU(),
            nn.Dropout(0.1),

            # Block 2: (batch, 64, 512) → (batch, 128, 256)
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.GELU(),
            nn.Dropout(0.1),

            # Block 3: (batch, 128, 256) → (batch, 128, 128)
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.GELU(),
        )
        # CNN 输出: (batch, 128, 128) → 序列长度128, 特征维度128

        # === BiLSTM 后端 ===
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],       # 128
            hidden_size=hidden_size,           # 128
            num_layers=num_layers,             # 2
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # === Attention ===
        self.attention = Attention(hidden_size * 2)  # *2 因为双向

        # === 分类头 ===
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, 1024, 2)
        x = x.permute(0, 2, 1)      # → (batch, 2, 1024) 适配 Conv1d

        # CNN 特征提取 + 降采样
        x = self.cnn(x)              # → (batch, 128, 128)
        x = x.permute(0, 2, 1)      # → (batch, 128, 128) 适配 LSTM

        # BiLSTM
        lstm_out, _ = self.lstm(x)   # → (batch, 128, 256)
        lstm_out = self.layer_norm(lstm_out)

        # Attention 加权聚合
        context = self.attention(lstm_out)  # → (batch, 256)

        # 分类
        output = self.classifier(context)   # → (batch, num_classes)
        return output


def build_model(classes=9):
    return CNN_BiLSTM_Attention(num_classes=classes)


# === 快速测试 ===
if __name__ == '__main__':
    model = build_model(classes=9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")