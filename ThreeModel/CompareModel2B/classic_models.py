"""
classic_models.py
-----------------
为 IQ 调制识别任务封装经典 CV 骨干网络。

输入 shape : (batch, 1024, 2)  ← 与原始 Dataset.__getitem__ 一致
输出 shape : (batch, num_classes)

支持的模型：
  - resnet50
  - efficientnet_b3
  - mobilenet_v3_large
  - densenet121

用法：
    from classic_models import build_classic_model
    model = build_classic_model('resnet50', num_classes=14, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# 公共适配层：将 (batch, 1024, 2) 转换为 (batch, 3, H, W)
# 这里把 1024 个时间步展开成 32×32 的"图像"，3 通道由线性映射得到
# ============================================================
class IQAdapter(nn.Module):
    """
    (batch, 1024, 2) → (batch, 3, 32, 32)

    步骤：
      1. Linear(2 → 3)           : 把 I/Q 两路映射成 3 通道
      2. reshape (1024, 3) → (3, 32, 32) : 展成类图像格式
    """
    def __init__(self):
        super().__init__()
        self.channel_proj = nn.Linear(2, 3)   # I/Q → RGB-like

    def forward(self, x):
        # x: (B, 1024, 2)
        x = self.channel_proj(x)              # (B, 1024, 3)
        x = x.permute(0, 2, 1)               # (B, 3, 1024)
        x = x.reshape(x.size(0), 3, 32, 32)  # (B, 3, 32, 32)
        return x


# ============================================================
# 通用包装：适配层 + 骨干网络
# ============================================================
class ClassicModelWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.adapter = IQAdapter()
        self.backbone = backbone

    def forward(self, x):
        x = self.adapter(x)       # (B, 3, 32, 32)
        return self.backbone(x)   # (B, num_classes)


# ============================================================
# 各模型构建函数
# ============================================================

def _build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    # 替换最后的全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_efficientnet_b3(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_large(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _build_densenet121(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# ============================================================
# 统一入口
# ============================================================
_MODEL_REGISTRY = {
    'resnet50':           _build_resnet50,
    'efficientnet_b3':    _build_efficientnet_b3,
    'mobilenet_v3_large': _build_mobilenet_v3_large,
    'densenet121':        _build_densenet121,
}


def build_classic_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    构建适配 IQ 信号的经典分类模型。

    Args:
        model_name  : 'resnet50' | 'efficientnet_b3' | 'mobilenet_v3_large' | 'densenet121'
        num_classes : 类别数
        pretrained  : 是否加载 ImageNet 预训练权重（迁移学习）

    Returns:
        nn.Module，输入 (batch, 1024, 2)，输出 (batch, num_classes)
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    builder = _MODEL_REGISTRY[model_name]
    backbone = builder(num_classes, pretrained)
    wrapped = ClassicModelWrapper(backbone)

    print(f"[build_classic_model] Model   : {model_name}")
    print(f"[build_classic_model] Pretrained: {pretrained}")
    total = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    print(f"[build_classic_model] Trainable params: {total:,}")

    return wrapped


if __name__ == '__main__':
    # 快速冒烟测试
    for name in _MODEL_REGISTRY:
        m = build_classic_model(name, num_classes=14, pretrained=False)
        dummy = torch.randn(4, 1024, 2)
        out = m(dummy)
        print(f"  {name}: input {dummy.shape} → output {out.shape}")
