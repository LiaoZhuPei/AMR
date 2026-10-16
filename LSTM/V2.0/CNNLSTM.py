import torch
import torch.nn as nn
import torch.nn.functional as F


class My_CNNLSTM_Net_V3(nn.Module):
    """
    一个简化的CNN-LSTM模型，使用1D卷积作为特征提取器，以最大程度保留时间信息。

    - CNN部分: 使用两层1D卷积，不进行池化，保持序列长度不变。
               作用类似于一个可学习的特征变换器，将(I, Q)信号映射到更高维的特征空间。
    - LSTM部分: 处理由CNN提取出的特征序列，捕捉时间依赖关系。
    """

    def __init__(self, lstm_hidden_size, lstm_num_layers, num_classes, dropout_rate=0.5):
        super(My_CNNLSTM_Net_V3, self).__init__()

        # --- 1. 1D卷积特征提取器 ---
        # 1D卷积的输入形状期望为: (Batch, Channels, Length)
        # 我们的输入将是 (N, 2, 1024)
        self.cnn_feature_extractor = nn.Sequential(
            # 第一层: 将 (I,Q) 2个通道映射到 64 个特征通道
            # kernel_size=7 or 8 是一个常见的选择
            # padding='same' 确保输出序列长度与输入相同
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # 第二层: 再从 64 映射到 128 个特征通道
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 在这种设计下，序列长度和特征维度是固定的
        self.final_seq_len = 1024  # 由于 padding='same'，序列长度保持不变
        self.lstm_input_features = 128  # 这是最后一个Conv1d的out_channels

        print(f"CNN-V3 初始化完成:")
        print(f"  - CNN输出序列长度: {self.final_seq_len}")
        print(f"  - CNN输出/LSTM输入特征维度: {self.lstm_input_features}")

        # --- 2. LSTM 序列处理器 ---
        # LSTM的输入形状期望为: (Batch, Sequence_Length, Features)
        # 我们的输入将是 (N, 1024, 128)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,  # 确保输入形状的第一个维度是Batch
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM以捕捉前后文信息
        )

        # --- 3. 最终的分类器 ---
        # 双向LSTM的输出特征维度是 2 * hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (torch.Tensor): 输入张量，原始形状为 (N, 1, 2, 1024)

        Returns:
            torch.Tensor: 分类器的输出 logits，形状为 (N, num_classes)
        """
        # --- 1. 整形以适应 Conv1d ---
        # 初始形状: (N, 1, 2, 1024)
        # .squeeze(1) 移除维度为1的通道维度
        # 目标形状: (N, 2, 1024)
        #x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        # --- 2. CNN 特征提取 ---
        # 输入形状: (N, 2, 1024)
        # 输出形状: (N, 128, 1024)
        x_features = self.cnn_feature_extractor(x)

        # --- 3. 整形以适应 LSTM ---
        # LSTM (batch_first=True) 需要 (Batch, Seq_Len, Features)
        # 当前形状: (N, 128, 1024) -> (Batch, Features, Seq_Len)
        # .permute(0, 2, 1) 交换最后两个维度
        # 目标形状: (N, 1024, 128)
        lstm_in = x_features.permute(0, 2, 1)

        # --- 4. LSTM 部分 ---
        # lstm_out 包含了每个时间步的输出，形状为 (N, 1024, lstm_hidden_size * 2)
        # 我们通常只关心最后一个时间步的输出用于分类
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)

        # 从双向LSTM的输出中提取最后一个时间步的有效信息
        # lstm_out[:, -1, :] 包含了最后一个时间步的前向和后向隐藏状态的拼接
        last_step_output = lstm_out[:, -1, :]

        # --- 5. 分类器部分 ---
        # 输入形状: (N, lstm_hidden_size * 2)
        # 输出形状: (N, num_classes)
        logits = self.classifier(last_step_output)

        return logits


# 提供一个构建函数，保持接口一致性
def build_model(classes):
    """
    构建并返回 My_CNNLSTM_Net_V3 模型实例。
    """
    return My_CNNLSTM_Net_V3(
        lstm_hidden_size=128,
        lstm_num_layers=2,
        num_classes=classes,
        dropout_rate=0.5
    )


# ===============================================================
# 使用示例 (用于测试)
# ===============================================================
if __name__ == '__main__':
    # 模拟参数
    BATCH_SIZE = 16
    NUM_CLASSES = 10

    # 构建模型
    model = build_model(classes=NUM_CLASSES)

    # 打印模型结构
    print("\n--- 模型结构 ---")
    print(model)

    # 创建一个假的输入张量来测试前向传播
    # 形状与您的数据加载器输出一致: (Batch, Channels, Height, Width)
    dummy_input_tensor = torch.randn(BATCH_SIZE, 1, 2, 1024)

    # 执行前向传播
    print("\n--- 测试前向传播 ---")
    print(f"输入张量形状: {dummy_input_tensor.shape}")

    with torch.no_grad():
        output_logits = model(dummy_input_tensor)

    print(f"输出 Logits 形状: {output_logits.shape}")

    # 验证输出形状是否正确
    assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
    print("\n模型测试通过！")