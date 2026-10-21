import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 自定义数据集
class Mydataset(Dataset):
    """
    Dataset 是 PyTorch 中用于数据集抽象的类。
    自定义数据集需要继承 torch.utils.data.Dataset 并重写以下两个方法：
    __len__：返回数据集的大小。
    __getitem__：按索引获取一个数据样本及其标签。
    """
    def __init__(self, data, labels):
        # 初始化数据
        self.data = data
        self.labels = labels

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 按索引返回数据和标签
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# 生成示例数据
data = torch.randn(100, 5)
labels = torch.randint(0, 2, (100,))

dataset = Mydataset(data, labels)

"""

"""
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

for batch_idx, (batch_data, batch_labels) in  enumerate(dataloader):
    print(f"批次 {batch_idx + 1}")
    print("数据:", batch_data)
    print("标签:", batch_labels)
    if batch_idx == 2:  # 仅显示前 3 个批次
        break

"""
# 内置数据集使用
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# 加载训练数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

# 使用 DataLoader 加载数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 查看一个批次的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(f"批次图像大小: {images.shape}")  # 输出形状为 [batch_size, 1, 28, 28]
print(f"批次标签: {labels}")
"""

"""
# 将 CSV 文件 作为数据源，并通过自定义 Dataset 和 DataLoader 读取数据。
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 自定义 CSV 数据集
class CSVDataset(Dataset):
    def __init__(self, file_path):
        # 读取 CSV 文件
        self.data = pd.read_csv(file_path)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 使用 .iloc 明确基于位置索引
        row = self.data.iloc[idx]
        # 将特征和标签分开
        features = torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)  # 特征
        label = torch.tensor(row.iloc[-1], dtype=torch.float32)  # 标签
        return features, label

# 实例化数据集和 DataLoader
dataset = CSVDataset("runoob_pytorch_data.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 遍历 DataLoader
for features, label in dataloader:
    print("特征:", features)
    print("标签:", label)
    break
"""