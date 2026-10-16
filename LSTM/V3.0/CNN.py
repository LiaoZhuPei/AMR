import os
import torch
import torch.nn as nn
from torchsummary import summary  # 可选，用于打印模型摘要，类似Keras的.summary()
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape=[1, 2, 1024], classes=9, **kwargs):
        """
        PyTorch implementation of the CNN model.
        Note: The input_shape is expected in [C, H, W] format.
        For this dataset, it's [1, 2, 1024].
        """
        super(CNN, self).__init__()

        self.conv_padding = (3, 4, 0, 1)

        # 定义卷积层、池化层和激活函数
        # PyTorch的Conv2d padding='same'在较新版本中可用，可以保持尺寸不变
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(2, 8), padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(2, 8), padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 8), padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 8), padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout4 = nn.Dropout(0.5)

        # Flatten层
        self.flatten = nn.Flatten()

        # 计算Flatten之后的全连接层输入维度
        # 输入: (N, 1, 2, 1024)
        # pool1后: (N, 256, 2, 512)
        # pool2后: (N, 128, 2, 256)
        # pool3后: (N, 64, 2, 128)
        # pool4后: (N, 64, 2, 64)
        # Flatten后: 64 * 2 * 64 = 8192
        self.fc_input_features = 64 * 2 * (input_shape[2] // 16)  # 1024 / (2*2*2*2) = 64

        # 定义全连接层
        self.dense1 = nn.Linear(self.fc_input_features, 128)
        self.relu5 = nn.ReLU()

        # 输出层，注意没有Softmax
        self.dense2 = nn.Linear(128, classes)

    def forward(self, x):
        x = F.pad(x, self.conv_padding)
        x = self.dropout1(self.pool1(self.relu1(self.conv1(x))))

        x = F.pad(x, self.conv_padding)
        x = self.dropout2(self.pool2(self.relu2(self.conv2(x))))

        x = F.pad(x, self.conv_padding)
        x = self.dropout3(self.pool3(self.relu3(self.conv3(x))))

        x = F.pad(x, self.conv_padding)
        x = self.dropout4(self.pool4(self.relu4(self.conv4(x))))

        x = self.flatten(x)

        x = self.relu5(self.dense1(x))
        x = self.dense2(x)  # 输出原始logits
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
    # 测试模型构建和打印摘要
    # 需要安装 torchsummary: pip install torchsummary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设输入shape为 [1, 2, 1024]，类别为26
    input_shape_pt = (1, 2, 1024)
    model = build_model(classes=26).to(device)

    # 打印模型结构
    print(model)

    # 打印类似Keras的摘要
    try:
        summary(model, input_size=input_shape_pt)
    except ImportError:
        print("\n'torchsummary' is not installed. Skipping model summary.")
        print("To install: pip install torchsummary")