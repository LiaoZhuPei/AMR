这三个扩展从**标准组件规范**、**模块化封装**到**残差设计**逐步递进，适合直接作为深度学习教学与工程模板。

---

### 扩展 1：规范化模板（包含 Conv + BN + Activation + Pooling + 自适应池化）

**设计要点与避坑指南：**
1. **标准执行顺序**：`Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`。
2. **`bias=False`**：当卷积层后面接 `BatchNorm` 时，卷积层的偏置会被归一化消除，设为 `False` 可以节省计算量与显存。
3. **`AdaptiveAvgPool2d`（自适应池化）**：这是工业界容错率最高的设计。它将任意尺寸的特征图强制转换为固定大小（如 $1 \times 1$ 或 $4 \times 4$），**彻底避免因为输入图片尺寸改变导致的全连接层维度不匹配报错**。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StandardCNN, self).__init__()
        
        # 第一层：Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层：Conv -> BN -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 容错核心：自适应平均池化，强制将输出特征图尺寸固定为 (1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),      # 加入 Dropout 防止过拟合
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)
        
        # Global Pooling & Head
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
```

---

### 扩展 2：复用卷积块模板（Double Conv Block）

**设计要点：**
1. 将通用的“连续两次卷积”封装为独立的 `DoubleConvBlock`（类似 VGG 或 U-Net 的经典骨干构建块）。
2. 在 Block 内部使用 `nn.Sequential` 打包，支持通道参数化，使得搭建深层网络就像搭积木一样简单。

```python
import torch
import torch.nn as nn

# 1. 定义可复用的两层卷积块
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.block = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二层卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# 2. 组装网络：利用 DoubleConvBlock 复用
class BlockCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(BlockCNN, self).__init__()
        
        # 复用卷积块
        self.stage1 = DoubleConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.stage2 = DoubleConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)
        
        x = self.stage2(x)
        x = self.pool2(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

---

### 扩展 3：标准简洁的残差连接模板（Residual Block）

**设计要点：**
1. **残差恒等映射**：核心是 $F(x) + x$。
2. **维度对齐（Shortcut/Downsample）**：当输入输出通道不同，或者通过 `stride=2` 降低分辨率时，$x$ 与 $F(x)$ 无法直接相加，此时 shortcut 路径必须通过 $1 \times 1$ 卷积自动将通道与尺寸对齐。
3. **加和后再激活**：相加操作 `x = out + identity` 发生在第二层 BN 之后、最后一个激活函数之前。

```python
import torch
import torch.nn as nn

# 1. 标准 BasicBlock 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主分支第一层（如果 stride > 1，这里负责下采样）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 主分支第二层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 捷径分支（Shortcut）
        self.shortcut = nn.Sequential()
        # 如果尺寸发生改变或通道数不一致，shortcut 需要用 1x1 卷积进行升维/降采样
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 残差直连分支
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity               # 核心残差相加
        out = self.relu(out)          # 相加后再通过激活函数
        return out


# 2. 组装极简 ResNet 骨干
class MiniResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(MiniResNet, self).__init__()
        
        # 初始特征提取
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 残差阶段：stride=1 保持尺寸；stride=2 下采样
        self.res_block1 = ResidualBlock(in_channels=32, out_channels=32, stride=1)
        self.res_block2 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
```