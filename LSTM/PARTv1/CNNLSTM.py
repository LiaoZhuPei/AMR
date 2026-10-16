import torch
import torch.nn as nn
import torch.nn.functional as F


class My_CNNLSTM_Net(nn.Module):


    def __init__(self, lstm_hidden_size, lstm_num_layers, num_classes, dropout_rate=0.5):
        super(My_CNNLSTM_Net, self).__init__()

        # --- 1. 复用你成功的 CNN 卷积块 ---
        # 我们将你的CNN层分组，便于阅读和调用

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        # 定义与你CNN中相同的padding
        self.conv_padding = (3, 4, 0, 1)

        # --- 动态计算CNN输出的维度 ---
        # 这一步非常重要，以确保我们知道送入LSTM的特征维度是多少
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
            # cnn_out shape: (1, 64, 2, 64) -> (Batch, Channels, Height, Width/Seq_len)

            # 我们需要将 Channels 和 Height 两个维度合并，作为LSTM的输入特征
            self.lstm_input_features = cnn_out.shape[1] * cnn_out.shape[2]  # 64 * 2 = 128
            self.final_seq_len = cnn_out.shape[3]  # 64

        print(f"CNN前端输出序列长度: {self.final_seq_len}, 特征维度: {self.lstm_input_features}")

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
    return My_CNNLSTM_Net(
        lstm_hidden_size=128,
        lstm_num_layers=2,
        num_classes=classes,
        dropout_rate=0.5
    )