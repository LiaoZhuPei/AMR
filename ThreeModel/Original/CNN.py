# 文件名: CNNtrain/rmlmodels/CNN.py

import os
import torch
import torch.nn as nn
from torchsummary import summary  # 可选，用于打印模型摘要，类似Keras的.summary()


class CNN(nn.Module):
    def __init__(self, input_shape=[1, 2, 1024], classes=26, **kwargs):
        """
        PyTorch implementation of the CNN model.
        Note: The input_shape is expected in [C, H, W] format.
        For this dataset, it's [1, 2, 1024].
        """
        super(CNN, self).__init__()

        self.netWidth = 1
        pool_size = 2
        sps = 4

        # 定义卷积层、池化层和激活函数
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16*self.netWidth, kernel_size=sps, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features = 16*self.netWidth)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        #self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(in_channels=16*self.netWidth, out_channels=32*self.netWidth, kernel_size=sps, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features = 32*self.netWidth)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        #self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=32*self.netWidth, out_channels=48*self.netWidth, kernel_size=sps, padding='same')
        self.bn3 = nn.BatchNorm1d(num_features = 48*self.netWidth)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=pool_size, stride = pool_size)
        self.dropout3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(in_channels=48*self.netWidth, out_channels=64*self.netWidth, kernel_size=sps, padding='same')
        self.bn4 = nn.BatchNorm1d(num_features = 64*self.netWidth)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=pool_size, stride = pool_size)
        self.dropout4 = nn.Dropout(0.5)

        self.conv5 = nn.Conv1d(in_channels=64*self.netWidth, out_channels=32*self.netWidth, kernel_size=sps, padding='same')
        self.bn5 = nn.BatchNorm1d(num_features=32*self.netWidth)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(32*self.netWidth, classes)

    def forward(self, x):
        
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))

        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))

        x = self.dropout5(self.relu5(self.bn5(self.conv5(x))))

        x = self.gap(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def build_model(weights_path=None, **kwargs):
    """
    辅助函数，用于构建模型并可选择性地加载权重。
    """
    model = CNN(**kwargs)

    if weights_path:
        if not os.path.exists(weights_path):
            raise ValueError(f'The `weights_path` argument provided does not exist: {weights_path}')
        # 加载PyTorch模型的state_dict
        model.load_state_dict(torch.load(weights_path))
        print(f"Model weights loaded from {weights_path}")

    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟 RML2016 数据集格式: [Batch, 1, 2, 1024]
    # 或者已经是 [Batch, 2, 1024]
    input_shape_test = (1, 2, 1024) 
    
    # sps (Samples Per Symbol) 对应 MATLAB 中的 sps 参数
    model = build_model(classes=11, sps=8).to(device)
    
    # 打印结构
    try:
        # torchsummary 这里的 input_size 不需要包含 batch 维度
        # 如果是 (2, 1024) 的时间序列
        summary(model, input_size=(2, 1024))
    except Exception as e:
        print(e)

    # 前向传播测试
    dummy_input = torch.randn(1, 1, 2, 1024).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 应该是 [1, 11]