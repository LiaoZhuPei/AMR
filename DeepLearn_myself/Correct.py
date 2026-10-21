import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# -------数据加载与预处理-------
"""
PyTorch 提供了强大的数据加载和处理工具，主要包括：
torch.utils.data.Dataset：数据集的抽象类，需要自定义并实现 __len__（数据集大小）和 __getitem__（按索引获取样本）。
torch.utils.data.TensorDataset：基于张量的数据集，适合处理数据-标签对，直接支持批处理和迭代。
torch.utils.data.DataLoader：封装 Dataset 的迭代器，提供批处理、数据打乱、多线程加载等功能，便于数据输入模型训练。
torchvision.datasets.ImageFolder：从文件夹加载图像数据，每个子文件夹代表一个类别，适用于图像分类任务。
torchvision： 一个图形库，提供了图片数据处理相关的 API 和数据集接口，包括数据集加载函数和常用的图像变换。
"""
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# -------模型构建（CNN）-------
"""
##基本概念##
输入
卷积（Convolution）:使用卷积核在输入矩阵上滑动，提取特征，生成Feature Maps
池化（Pooling）:通常在Convolution layer之后，通过MaxPooling和AveragePooling减少特征图的尺寸，同时保留重要特征
归一化（Normalization）:局部响应归一化（Local Response Normalization, LRN）或批归一化（Batch Normalization）,加速训练过程，提高模型稳定性
特征提取（Feature Extraction）：通过多个卷积和池化层的组合，逐步提取图像的高级特征
展平层（Flatten layer）: 将多维的特征图转换为一维向量，以便输入到全连接层
全连接层（Fully Connected Layer）:用于将提取的特征映射到输出类别
分类（Classification）：网络的输出层，根据全连接层的输出进行分类
概率分布（Probabilistic Distribution）：输出层给出每个类别的概率，表示输入图像属于各个类别的可能性

##基本结构##
1、输入层（Input Layer）
2、卷积层（Convolutional Layer）
3、激活函数（Activation Function）
4、池化层（Pooling Layer）
5、归一化层（Normalization Layer，可选）
6、全连接层（Fully Connected Layer）
7、输出层（Output Layer）

##说明##
详见Layer_introduce.md

"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.f1 = nn.LazyLinear(128)  # 使用 LazyLinear，只填输出类别/节点数(128)，不写输入大小,它会自动识别出输入是 64*7*7=3136，并将其固定为标准的 nn.Linear
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = F.relu(self.f1(x))  # relu:防止多层退化 和 提供非线性拟合能力;如果不加激活函数，两层线性层在数学上完全等价于一层
        x = self.f2(x)
        return x

model = SimpleCNN()

# -------定义损失函数和优化器-------
"""
##说明##
详见loss_optim_introduce.md
"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# -------模型训练-------
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        ouputs = model(images)
        loss = criterion(ouputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# -------模型测试-------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preticed = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preticed == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")