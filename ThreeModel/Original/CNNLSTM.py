import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsummary import summary
class CNNLSTM(nn.Module):
    def __init__(self, input_shape=[1, 2, 1024], classes=11, sps=8, **kwargs):
        super(CNNLSTM, self).__init__()
        
        self.filter_size = sps 
        pool_size = 2
        
        # --- 1. 即使有外部归一化，建议保留 InstanceNorm ---
        # 原因：Matlab 的 sequenceInputLayer(Normalization="zscore") 是对
        # 每个样本独立做 0均值 1方差 处理。这比单纯的 [-1, 1] 缩放更能
        # 凸显 16QAM 和 64QAM 的星座图密度差异。
        # affine=False 表示不学习参数，纯粹做统计标准化，模拟 Matlab 行为。
        self.input_norm = nn.InstanceNorm1d(2, affine=False)

        # --- CNN 结构 (严格对齐 Matlab) ---
        self.conv1 = nn.Conv1d(2, 16, kernel_size=self.filter_size, padding='same')
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=self.filter_size, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 48, kernel_size=self.filter_size, padding='same')
        self.bn3 = nn.BatchNorm1d(48)
        
        self.conv4 = nn.Conv1d(48, 64, kernel_size=self.filter_size, padding='same')
        self.bn4 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

        # --- LSTM 设置 ---
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=64,  # 上一层 Conv4 输出通道数
            hidden_size=self.lstm_hidden_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )

        # --- 分类头 ---
        self.dropout = nn.Dropout(0.2)
        # BiLSTM 输出是 hidden_size * 2
        self.fc = nn.Linear(self.lstm_hidden_size * 2, classes)
        
        # --- 权重初始化 (对 QAM 识别非常重要) ---
        self._init_weights()

    def _init_weights(self):
        """
        Matlab 默认使用 Glorot/Xavier 初始化，LSTM 内部使用 Orthogonal。
        PyTorch 默认是 Uniform，这可能导致 QAM 这种密集星座图难以收敛。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # 循环层使用正交初始化，有助于长期依赖记忆
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # [Batch, 1, 2, 1024] -> [Batch, 2, 1024]
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        # 1. 输入标准化 (即使外部有归一化，这层也很关键，确保分布稳定)
        x = self.input_norm(x)

        # 2. CNN 特征提取
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # 此时 x 形状: [Batch, 64 (Channels), 64 (Length)]

        # 3. 维度置换: 准备进入 LSTM
        # LSTM 需要 [Batch, Sequence_Length, Features]
        x = x.permute(0, 2, 1) 
        # 此时 x 形状: [Batch, 64, 64]

        # 4. LSTM 前向传播
        # output: 每个时间步的输出
        # h_n: 最后一个时间步的隐状态 (Hidden State)
        self.lstm.flatten_parameters() # 优化内存，加速
        _, (h_n, _) = self.lstm(x)
        
        # 5. --- 关键修正: 模拟 Matlab 'OutputMode', 'last' ---
        # h_n 的形状是 [num_layers * num_directions, batch, hidden_size]
        # 对于双向 LSTM:
        # h_n[-2] 是正向层 (Forward) 处理完整个序列后的状态
        # h_n[-1] 是反向层 (Backward) 处理完整个序列后的状态
        # 我们将这两者拼接，得到完整的全局上下文特征
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # 6. 全连接分类
        x = self.dropout(x)
        x = self.fc(x)

        return x

def build_model(weights_path=None, **kwargs):
    model = CNNLSTM(**kwargs) # 使用新的类名
    if weights_path:
        if not os.path.exists(weights_path):
            raise ValueError(f'The `weights_path` argument provided does not exist: {weights_path}')
        model.load_state_dict(torch.load(weights_path))
        print(f"Model weights loaded from {weights_path}")
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 假设输入是 [Batch, 2, 1024]
    # sps 应该根据你的数据设置，Matlab代码中它等于卷积核大小
    model = build_model(classes=11, sps=8).to(device)
    
    print("模型结构:")
    try:
        # 输入尺寸 (Channels, SequenceLength)
        summary(model, input_size=(2, 1024))
    except Exception as e:
        print(f"Summary Error: {e}")

    # 前向传播测试
    dummy_input = torch.randn(1, 1, 2, 1024).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 应该是 [1, 11]