#v1
import torch
import torch.nn as nn
import torch.nn.functional as F


class My_CNNLSTM_Net_V1(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_num_layers, num_classes, dropout_rate=0.5):
        super(My_CNNLSTM_Net_V1, self).__init__()

        # --- CNN Blocks ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 池化, 序列长度: 1024 -> 512 (近似)
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # 池化, 序列长度: 512 -> 256 (近似)
            nn.Dropout(dropout_rate)
        )
        # 简化: 后两层不再进行池化，只做卷积
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2)), # <-- REMOVED POOLING
            nn.Dropout(dropout_rate)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2)), # <-- REMOVED POOLING
            nn.Dropout(dropout_rate)
        )

        self.conv_padding = (3, 4, 0, 1)

        # --- 动态计算维度 ---
        # 必须重新计算，因为序列长度变了
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 2, 1024)
            x = F.pad(dummy_input, self.conv_padding)
            x = self.conv_block1(x)
            x = F.pad(x, self.conv_padding)
            x = self.conv_block2(x)
            x = F.pad(x, self.conv_padding)
            x = self.conv_block3(x)
            x = F.pad(x, self.conv_padding)
            cnn_out = self.conv_block4(x)

            self.lstm_input_features = cnn_out.shape[1] * cnn_out.shape[2]
            self.final_seq_len = cnn_out.shape[3]

        print(f"CNN-V1 输出序列长度: {self.final_seq_len}, 特征维度: {self.lstm_input_features}")

        # --- 2. LSTM 序列处理器 ---
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True  # 双向通常效果更好
        )

        # --- 3. 最终的分类器 ---
        # 双向LSTM的输出维度是 2 * hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (N, 1, 2, 1024)
        """
        # --- CNN 部分 ---
        # 逐层应用卷积块和padding，和你原来的CNN forward完全一样
        x = F.pad(x, self.conv_padding)
        x = self.conv_block1(x)
        x = F.pad(x, self.conv_padding)
        x = self.conv_block2(x)
        x = F.pad(x, self.conv_padding)
        x = self.conv_block3(x)
        x = F.pad(x, self.conv_padding)
        cnn_out = self.conv_block4(x)
        # cnn_out shape: (N, 64, 2, 64)

        # --- 整形以适应 LSTM ---
        batch_size = cnn_out.shape[0]
        # 我们要将 (N, C, H, W) 变成 (N, W, C*H)
        # 1. 交换维度: (N, 64, 2, 64) -> (N, 64, 64, 2) (permute)
        # 2. 合并最后两维: -> (N, 64, 128)
        # 一个更直接的方法:
        # permute: (N, C, H, W) -> (N, W, C, H)
        lstm_in = cnn_out.permute(0, 3, 1, 2)  # -> (N, 64, 64, 2)
        # view/reshape: (N, 64, 64, 2) -> (N, 64, 128)
        lstm_in = lstm_in.contiguous().view(batch_size, self.final_seq_len, self.lstm_input_features)

        # --- LSTM 部分 ---
        lstm_out, _ = self.lstm(lstm_in)
        last_step_output = lstm_out[:, -1, :]

        # --- 分类器部分 ---
        logits = self.classifier(last_step_output)

        return logits


# 提供一个构建函数
def build_model(classes):
    return My_CNNLSTM_Net_V1(
        lstm_hidden_size=128,
        lstm_num_layers=2,
        num_classes=classes,
        dropout_rate=0.5
    )