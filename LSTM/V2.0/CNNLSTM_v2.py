#v1
import torch
import torch.nn as nn
import torch.nn.functional as F


class My_CNNLSTM_Net_V2(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_num_layers, num_classes, dropout_rate=0.5):
        super(My_CNNLSTM_Net_V2, self).__init__()

        # --- 只保留前两个CNN块 ---
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

        self.conv_padding = (3, 4, 0, 1)

        # --- 动态计算维度 ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 2, 1024)
            x = F.pad(dummy_input, self.conv_padding)
            x = self.conv_block1(x)
            x = F.pad(x, self.conv_padding)
            cnn_out = self.conv_block2(x)

            self.lstm_input_features = cnn_out.shape[1] * cnn_out.shape[2]  # 128 * 2 = 256
            self.final_seq_len = cnn_out.shape[3]  # 约 256

        print(f"CNN-V2 输出序列长度: {self.final_seq_len}, 特征维度: {self.lstm_input_features}")

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
        # --- Forward也相应简化 ---
        x = F.pad(x, self.conv_padding)
        x = self.conv_block1(x)
        x = F.pad(x, self.conv_padding)
        cnn_out = self.conv_block2(x)

        # --- 整形并送入LSTM (逻辑不变) ---
        batch_size = cnn_out.shape[0]
        lstm_in = cnn_out.permute(0, 3, 1, 2)
        lstm_in = lstm_in.contiguous().view(batch_size, self.final_seq_len, self.lstm_input_features)

        lstm_out, _ = self.lstm(lstm_in)
        last_step_output = lstm_out[:, -1, :]
        logits = self.classifier(last_step_output)

        return logits


# 提供一个构建函数
def build_model(classes):
    return My_CNNLSTM_Net_V2(
        lstm_hidden_size=128,
        lstm_num_layers=2,
        num_classes=classes,
        dropout_rate=0.5
    )