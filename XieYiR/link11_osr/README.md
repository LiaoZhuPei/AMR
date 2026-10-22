# Link11 开放集协议识别系统 (Open-Set Recognition)

## 项目概述

本项目使用深度学习解决**开放集识别 (Open-Set Recognition, OSR)** 问题：
- **核心任务**：判断输入信号是否为 Link11（合法目标信号）
- **开放集挑战**：测试时会出现训练集中从未见过的未知信号，需正确拒识

## 方法选择：**Deep SVDD + 对比特征学习**

### 为什么选这个方法？
| 方法 | 优点 | 缺点 |
|------|------|------|
| 普通分类器 + Softmax | 简单 | 无法拒识未知类，输出必然指向某已知类 |
| One-Class SVM | 经典 | 特征需手工提取，泛化弱 |
| **Deep SVDD (本项目)** | 端到端、可扩展、天然支持开放集 | 需调阈值 |
| Prototype Network | 性能更好 | 需要更多正负对样本设计 |

### 核心思路
```
信号 → 特征提取器(CNN) → 紧凑特征球 → 距离判断
                              ↑
                         超球面中心 c
                         (由MT63训练确定)

距离 < 阈值 → "Link11 合法信号" ✓
距离 ≥ 阈值 → "未知信号，拒识"  ✗
```

## 数据集配置

**数据来源**: [Panoradio SDR Radio Signal Classification Dataset](https://panoradio-sdr.de/radio-signal-classification-dataset/)

| 角色 | 信号类型 | 用途 |
|------|----------|------|
| **正类 (Known Target)** | MT63 | 模拟 Link11，训练目标 |
| **已知负类 (Known Unknowns)** | RTTY, SSB语音, AM广播 | 训练时的负样本 |
| **未知负类 (Unknown Unknowns)** | PSK31, Morse Code, Olivia | 仅测试，模型从未见过 |

## 项目结构

```
link11_osr/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml          # 统一配置文件
├── src/
│   ├── dataset.py           # 数据加载与预处理（IQ → 时频图）
│   ├── model.py             # CNN特征提取器 + SVDD头
│   ├── trainer.py           # 训练逻辑（SVDD损失 + 对比辅助损失）
│   ├── evaluator.py         # 评估：AUROC, FPR@95TPR, 混淆矩阵
│   └── utils.py             # 公共工具函数
├── train.py                 # 训练入口
├── evaluate.py              # 评估入口
├── infer.py                 # 单信号推理入口
├── data/
│   ├── raw/                 # 原始数据放这里
│   └── processed/           # 预处理后的数据
├── models/                  # 保存的模型权重
├── results/                 # 评估结果、图表
└── notebooks/
    └── exploration.ipynb    # 数据探索笔记本
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
下载数据集后，将各信号的 `.wav` 或 IQ 文件放入 `data/raw/`，
目录结构如下：
```
data/raw/
├── MT63/
├── RTTY/
├── SSB/
├── AM/
├── PSK31/
├── Morse/
└── Olivia/
```

### 3. 训练
```bash
python train.py --config configs/config.yaml
```

### 4. 评估（开放集测试）
```bash
python evaluate.py --config configs/config.yaml --model models/best_model.pth
```

### 5. 单信号推理
```bash
python infer.py --signal path/to/signal.wav --model models/best_model.pth
```

## 扩展方向

- [ ] 替换骨干网络为 ResNet / Transformer
- [ ] 加入 Prototype Network 提升小样本性能
- [ ] 使用能量分数 (Energy Score) 替代距离阈值
- [ ] 自动阈值校准（基于验证集 FPR 约束）
- [ ] 实时流信号推理
