"""
classic_models.py  (v2 — 1D 序列专用版)
----------------------------------------
为 IQ 调制识别任务提供专为时序信号设计的经典模型。

输入 shape : (batch, 1024, 2)   ← Dataset.__getitem__ 输出，无需修改
输出 shape : (batch, num_classes)

可选模型（均无需预训练权重，从零训练即可快速收敛）：
  ┌─────────────────────┬────────┬──────────────────────────────────────┐
  │ model_name          │ 参数量 │ 特点                                 │
  ├─────────────────────┼────────┼──────────────────────────────────────┤
  │ resnet1d            │  ~0.9M │ 1D ResNet，调制识别主流基线           │
  │ inception1d         │  ~1.5M │ 多尺度并行卷积，捕捉不同粒度特征      │
  │ densenet1d          │  ~0.7M │ 密集连接，特征复用强，梯度流动好      │
  │ tcn                 │  ~1.2M │ 时序卷积网络，因果膨胀卷积，感受野大  │
  └─────────────────────┴────────┴──────────────────────────────────────┘

用法：
    from classic_models import build_classic_model
    model = build_classic_model('resnet1d', num_classes=14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. ResNet1D
#    经典残差网络的 1D 版本，调制识别领域最常用基线
#    参考: "Deep Learning Based Automatic Modulation Classification" (2018)
# ============================================================

class ResBlock1D(nn.Module):
    """标准残差块（pre-activation 变体，训练更稳定）"""
    def __init__(self, channels: int, kernel_size: int = 3, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        pad    = kernel_size // 2

        self.bn1   = nn.BatchNorm1d(channels // 2 if downsample else channels)
        self.conv1 = nn.Conv1d(
            channels // 2 if downsample else channels,
            channels, kernel_size, stride=stride, padding=pad, bias=False
        )
        self.bn2   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)

        self.shortcut = nn.Sequential(
            nn.Conv1d(channels // 2 if downsample else channels,
                      channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(channels)
        ) if downsample else nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class ResNet1D(nn.Module):
    """
    输入: (B, 1024, 2)  →  内部转置为 (B, 2, 1024) 做 Conv1d
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),  # → (B,32,512)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)                               # → (B,32,256)
        )
        # 4 stage，每个 stage 2个残差块，通道翻倍并下采样
        self.stage1 = nn.Sequential(ResBlock1D(32),  ResBlock1D(32))           # → (B,32,256)
        self.stage2 = nn.Sequential(ResBlock1D(64,  downsample=True),
                                    ResBlock1D(64))                            # → (B,64,128)
        self.stage3 = nn.Sequential(ResBlock1D(128, downsample=True),
                                    ResBlock1D(128))                           # → (B,128,64)
        self.stage4 = nn.Sequential(ResBlock1D(256, downsample=True),
                                    ResBlock1D(256))                           # → (B,256,32)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B,1024,2) → (B,2,1024)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ============================================================
# 2. Inception1D
#    多尺度并行卷积，同时捕捉不同时间粒度的特征
#    适合调制信号：不同调制方式在不同时间尺度上有不同特征
# ============================================================

class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch = out_ch // 4   # 4个分支各占 1/4 输出通道

        # 分支1：1×1 卷积（点特征）
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, branch, 1, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(inplace=True)
        )
        # 分支2：1×1 → 3×1 （短时特征）
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, branch, 1, bias=False), nn.BatchNorm1d(branch), nn.ReLU(inplace=True),
            nn.Conv1d(branch, branch, 3, padding=1, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(inplace=True)
        )
        # 分支3：1×1 → 7×1 （中时特征）
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, branch, 1, bias=False), nn.BatchNorm1d(branch), nn.ReLU(inplace=True),
            nn.Conv1d(branch, branch, 7, padding=3, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(inplace=True)
        )
        # 分支4：MaxPool → 1×1 （全局池化特征）
        self.b4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch, 1, bias=False),
            nn.BatchNorm1d(branch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class Inception1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)                    # → (B,32,256)
        )
        self.blocks = nn.Sequential(
            InceptionBlock1D(32,  64),                              # → (B,64,256)
            nn.MaxPool1d(2),                                        # → (B,64,128)
            InceptionBlock1D(64,  128),                             # → (B,128,128)
            nn.MaxPool1d(2),                                        # → (B,128,64)
            InceptionBlock1D(128, 256),                             # → (B,256,64)
            nn.MaxPool1d(2),                                        # → (B,256,32)
            InceptionBlock1D(256, 256),                             # → (B,256,32)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B,1024,2) → (B,2,1024)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ============================================================
# 3. DenseNet1D
#    密集连接：每层都接收之前所有层的输出，特征复用率极高
#    在样本量充足时收敛速度快
# ============================================================

class DenseLayer1D(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_ch), nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, growth_rate * 4, 1, bias=False),      # bottleneck
            nn.BatchNorm1d(growth_rate * 4), nn.ReLU(inplace=True),
            nn.Conv1d(growth_rate * 4, growth_rate, 3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock1D(nn.Module):
    def __init__(self, in_ch: int, num_layers: int, growth_rate: int = 16):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer1D(ch, growth_rate))
            ch += growth_rate
        self.block   = nn.Sequential(*layers)
        self.out_ch  = ch

    def forward(self, x):
        return self.block(x)


class DenseNet1D(nn.Module):
    def __init__(self, num_classes: int, growth_rate: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)                   # → (B,32,256)
        )
        # 3 个 Dense Block，每个 Block 后接 Transition（压缩通道+下采样）
        g = growth_rate
        b1 = DenseBlock1D(32,  num_layers=6, growth_rate=g)       # out: 32+6*16=128
        t1_out = b1.out_ch // 2
        b2 = DenseBlock1D(t1_out, num_layers=8, growth_rate=g)    # out: 64+8*16=192
        t2_out = b2.out_ch // 2
        b3 = DenseBlock1D(t2_out, num_layers=8, growth_rate=g)    # out: 96+8*16=224

        self.block1 = b1
        self.trans1 = self._transition(b1.out_ch, t1_out)
        self.block2 = b2
        self.trans2 = self._transition(b2.out_ch, t2_out)
        self.block3 = b3
        self.bn_final = nn.BatchNorm1d(b3.out_ch)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.head  = nn.Linear(b3.out_ch, num_classes)

    @staticmethod
    def _transition(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.BatchNorm1d(in_ch), nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool1d(2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = F.relu(self.bn_final(x))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ============================================================
# 4. TCN (Temporal Convolutional Network)
#    因果膨胀卷积：感受野随层数指数增长，无 RNN 的递归瓶颈
#    参考: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" (2018)
# ============================================================

class TCNBlock(nn.Module):
    """带残差的因果膨胀卷积块"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation   # 因果卷积需要在左侧填充

        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           dilation=dilation, padding=pad)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           dilation=dilation, padding=pad)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.causal_pad = pad
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu       = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        out = out[:, :, :x.size(2)]   # 裁掉因果填充，保持长度不变
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_classes: int, num_channels=None, kernel_size: int = 7):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128, 256, 256]   # 6层，感受野 = (7-1)*(1+2+4+8+16+32)*2+1 = 762

        self.input_proj = nn.Conv1d(2, num_channels[0], 1)   # 2→64 投影

        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch    = num_channels[i-1] if i > 0 else num_channels[0]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
        self.network = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_channels[-1], num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)           # (B,1024,2) → (B,2,1024)
        x = self.input_proj(x)            # (B,2,1024) → (B,64,1024)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ============================================================
# 统一入口
# ============================================================
_MODEL_REGISTRY = {
    'resnet1d'   : ResNet1D,
    'inception1d': Inception1D,
    'densenet1d' : DenseNet1D,
    'tcn'        : TCN,
}


def build_classic_model(model_name: str, num_classes: int,
                        pretrained: bool = False) -> nn.Module:
    """
    构建适配 IQ 信号的 1D 经典分类模型。

    Args:
        model_name  : 'resnet1d' | 'inception1d' | 'densenet1d' | 'tcn'
        num_classes : 类别数
        pretrained  : 保留参数（1D模型无预训练权重，此参数忽略）

    Returns:
        nn.Module，输入 (batch, 1024, 2)，输出 (batch, num_classes)
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    if pretrained:
        print(f"[build_classic_model] Note: '{model_name}' has no pretrained weights, "
              f"training from scratch.")

    model = _MODEL_REGISTRY[model_name](num_classes=num_classes)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_classic_model] Model   : {model_name}")
    print(f"[build_classic_model] Trainable params: {total:,}")
    return model


if __name__ == '__main__':
    dummy = torch.randn(4, 1024, 2)
    for name in _MODEL_REGISTRY:
        m = build_classic_model(name, num_classes=14)
        out = m(dummy)
        print(f"  {name:15s}: input {list(dummy.shape)} → output {list(out.shape)}")
