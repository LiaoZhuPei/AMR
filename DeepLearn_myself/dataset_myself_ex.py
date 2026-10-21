import torch
from torch.utils.data import Dataset, DataLoader, random_split


"""
场景 1：数据集是一个文件，在代码中划分为训练集、验证集、测试集
适用场景：只有一个完整的表格/文件，想要在代码中按比例（如 7:2:1）拆分为 Train、Val、Test。
核心工具：torch.utils.data.random_split
"""
# 1. 基础 Dataset 定义
class FullDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 模拟加载了单文件的数据（共 100 条）
all_data = torch.randn(100, 5)
all_labels = torch.randint(0, 2, (100,))
full_dataset = FullDataset(all_data, all_labels)

# 2. 划分数据集：按比例计算具体样本数 (70 : 20 : 10)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size  # 保证总和等于 100

# 推荐固定随机数种子，确保每次运行划分一致
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 3. 分别构建 DataLoader
# 注意：训练集通常打乱 (shuffle=True)，验证/测试集通常不打乱 (shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

# ----------------------------------------------------------------
"""
场景 2：数据已经是不同的文件（如 train.csv、val.csv、test.csv）
适用场景：标准数据集中，训练集、验证集和测试集本身就是独立的文件。
实现技巧：在 Dataset 中增加一个 mode 参数或直接传入具体的文件路径
"""
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 模拟创建 3 个独立文件
pd.DataFrame({'feat1': [1.0, 2.0], 'feat2': [0.5, 0.8], 'label': [0, 1]}).to_csv("train.csv", index=False)
pd.DataFrame({'feat1': [3.0], 'feat2': [0.2], 'label': [0]}).to_csv("val.csv", index=False)
pd.DataFrame({'feat1': [4.0], 'feat2': [0.9], 'label': [1]}).to_csv("test.csv", index=False)

# 1. 设计支持读取指定文件的通用 Dataset
class FileSplitDataset(Dataset):
    def __init__(self, file_path):
        # 读取指定的文件
        df = pd.read_csv(file_path)
        
        # 假设前两列是特征，最后一列是标签
        self.features = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 2. 分别为每个文件实例化 Dataset
train_dataset = FileSplitDataset("train.csv")
val_dataset   = FileSplitDataset("val.csv")
test_dataset  = FileSplitDataset("test.csv")

# 3. 构造各自的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False)

for x, y in train_loader:
    print("训练批次特征:", x, "标签:", y)
"""

"""
场景 3：标签和数据在同一个文件中
适用场景：表格数据、单一的 .csv 或 .tsv 文件，特征（Feature）和标签（Target/Label）在不同列。
核心要点：通过切片将 DataFrame / 矩阵分离为 X 和 y
"""
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 模拟创建包含特征和标签的单个文件
demo_df = pd.DataFrame({
    'age': [25, 30, 45, 22],
    'income': [50000, 80000, 110000, 35000],
    'credit_score': [650, 720, 800, 600],
    'approved': [0, 1, 1, 0]  # 这是标签
})
demo_df.to_csv("user_data.csv", index=False)

class UnifiedTableDataset(Dataset):
    def __init__(self, csv_file, label_col='approved'):
        df = pd.read_csv(csv_file)
        
        # 提取标签列
        labels = df[label_col].values
        # 排除标签列，其余作为特征
        features = df.drop(columns=[label_col]).values
        
        # 转换为 Tensor
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = UnifiedTableDataset("user_data.csv", label_col='approved')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for x_batch, y_batch in dataloader:
    print("X 形状:", x_batch.shape, "y 形状:", y_batch.shape)
"""

"""
场景 4：标签和数据保存在不同的文件中
适用场景：
特征存在 features.npy，标签存在 labels.npy；
图像存在一个文件夹（如 images/001.jpg），而标签存在一个对应的索引表格（labels.csv，列为 image_id, label)
"""
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 模拟保存独立的“特征文件”和“标签文件”
np.save("features.npy", np.random.randn(50, 128))  # 50条，每条128维
pd.DataFrame({'label': np.random.randint(0, 2, 50)}).to_csv("labels.csv", index=False)

class SeparatedDataset(Dataset):
    
    #数据与标签分开保存的数据集加载器
    
    def __init__(self, data_file_path, label_file_path):
        # 1. 读取数据文件 (以 .npy 为例)
        data_array = np.load(data_file_path)
        self.data = torch.tensor(data_array, dtype=torch.float32)

        # 2. 读取标签文件 (以 .csv 为例)
        label_df = pd.read_csv(label_file_path)
        self.labels = torch.tensor(label_df['label'].values, dtype=torch.long)

        # 3. 校验两者的样本数量是否严格一致（非常重要！）
        assert len(self.data) == len(self.labels), \
            f"数据样本数 ({len(self.data)}) 与标签数 ({len(self.labels)}) 不一致！"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 实例化并使用
dataset = SeparatedDataset(
"""