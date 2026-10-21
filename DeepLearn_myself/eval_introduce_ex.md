在机器学习与深度学习中，评价指标是模型的“体检报告”。单一指标（如单纯看准确率）往往存在严重盲区，必须结合任务类型组合使用。

---

### 一、 核心评价指标详解（分类与通用指标）

#### 1. 损失值（Loss / Train Loss & Val Loss）
* **核心意义**：优化目标函数的量化值，反映当前模型参数对训练目标的拟合程度。
* **如何反映模型状态**（**诊断模型的核心风向标**）：
  * **Train Loss 持续下降，Val Loss 持续下降** $\rightarrow$ 模型处于**健康学习**状态。
  * **Train Loss 持续下降，Val Loss 开始上升** $\rightarrow$ 模型出现**过拟合（Overfitting）**，记住了噪声，需加正则化、早停或补充数据。
  * **Train Loss 下降缓慢或停滞在较高值** $\rightarrow$ 模型**欠拟合（Underfitting）**，模型容量不足或学习率设置不当。
  * **Train Loss 剧烈震荡甚至出现 `NaN`/`Inf`** $\rightarrow$ 学习率过大、梯度爆炸或数据中存在脏数据。
* **使用阶段**：**全周期监控**（训练集用于梯度反向传播；验证集用于调参与早停判断）。
* **使用方法**：
  ```python
  # 训练/验证循环中直接从 criterion 获取
  loss = criterion(outputs, labels)
  batch_loss = loss.item()
  ```

---

#### 2. 准确率（Accuracy / Top-1 & Top-5 Acc）
* **核心意义**：预测正确的样本数占总样本数的比例：
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **如何反映模型状态**：
  * **适用场景**：**各类别样本数量高度均衡**时，最直观的性能指标。
  * **致命盲区（样本不平衡陷阱）**：在罕见病检测中，99% 为阴性，1% 为阳性。模型就算“摆烂”全猜阴性，准确率也是 99%，但其实毫无实用价值。
  * *注：Top-5 准确率常用于超多分类任务（如 ImageNet 1000类），只要概率最高的前 5 个预测中包含真实标签即算正确。*
* **使用阶段**：训练集、验证集、测试集均可监控。
* **使用方法（PyTorch 原生写法）**：
  ```python
  _, preds = torch.max(outputs, dim=1)
  correct = (preds == labels).sum().item()
  acc = correct / labels.size(0)
  ```

---

#### 3. 精确率（Precision / 查准率）与 召回率（Recall / 查全率）
* **核心意义**：
  * **Precision**：**预测为阳性的样本中，有多少是真的阳性**：$\frac{TP}{TP + FP}$（反映“宁缺毋滥”的程度）。
  * **Recall**：**所有真正的阳性样本中，被模型找出来了多少**：$\frac{TP}{TP + FN}$（反映“宁可错杀三千，不放过一个”的程度）。
* **如何反映模型状态**：
  * **高 Precision，低 Recall**：模型决策**极其保守**。只有把握极大时才报阳性，漏诊严重。
    * *典型业务*：垃圾邮件拦截、个性化商品推荐（宁愿少推，不能推错恶心用户）。
  * **低 Precision，高 Recall**：模型决策**非常激进**。宁可多报假警报，也不能放过潜在风险。
    * *典型业务*：癌症筛查、地震预警、金融欺诈风控（后续可以人工复审，但漏报代价巨大）。
* **使用阶段**：**验证集与测试集**（通常在每个 Epoch 结束时全量评估一次，不适合按小 Batch 计算）。
* **使用方法**：
  ```python
  from sklearn.metrics import precision_score, recall_score
  # 必须使用全部样本的真实值与预测类别
  precision = precision_score(y_true, y_pred, average='macro') # 多分类常用 macro 或 weighted
  recall = recall_score(y_true, y_pred, average='macro')
  ```

---

#### 4. F1-Score（F1 分数）
* **核心意义**：精确率与召回率的**调和平均数**：
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **如何反映模型状态**：
  * Precision 与 Recall 是天然互相博弈的（阈值调高，Precision 上升而 Recall 下降）。
  * F1 对极端偏低的单项指标施加了重罚（调和平均的特性）。当**正负样本分布极不均匀**时，F1-Score 是评估模型综合能力**最推荐的单一量化指标**。
* **使用阶段**：**验证集与测试集**（通常作为早停和选取最佳权重 `best_model` 的依据）。
* **使用方法**：
  ```python
  from sklearn.metrics import f1_score
  f1 = f1_score(y_true, y_pred, average='weighted')
  ```

---

#### 5. AUC-ROC（ROC 曲线下面积）
* **核心意义**：
  * ROC 曲线是以假阳性率（FPR）为横轴、真阳性率（TPR）为纵轴绘制的曲线。
  * **AUC** 是该曲线下的几何面积，数值范围在 $[0.5, 1.0]$。其物理含义是：**随机挑选一个正样本和一个负样本，模型给正样本打的分数（概率）大于负样本的概率**。
* **如何反映模型状态**：
  * **与分类阈值解耦**：Accuracy/Precision 均依赖你指定的分类阈值（如 $0.5$），而 AUC **衡量的是模型对正负样本排序能力的本质优劣**。
  * **对样本不均衡不敏感**：即使负样本暴增 100 倍，AUC 的数值表现依然稳健。
  * **数值诊断**：
    * $\text{AUC} = 0.5$：等同于随机瞎猜。
    * $\text{AUC} \in [0.7, 0.8]$：具备可用性。
    * $\text{AUC} > 0.9$：优秀的判别能力。
* **使用阶段**：**验证集与测试集**（工业界点击率预测 CTR、信用评分的标准上线指标）。
* **使用方法**：
  ```python
  from sklearn.metrics import roc_auc_score
  # 注意：必须传入预测的【概率值/Logits】，绝不能传入分类类别索引！
  auc = roc_auc_score(y_true, y_probs)
  ```

---

#### 6. 混淆矩阵（Confusion Matrix）
* **核心意义**：以矩阵形式展示每个真实类别被预测为各个类别的具体样本数量。
* **如何反映模型状态**（**定性分析的神器**）：
  * 单纯的数字指标只告诉你“模型表现不好”，而混淆矩阵会精准告诉你**“模型把谁错认成了谁”**。
  * *例如在交通标志识别中，混淆矩阵能一眼看出是“限速 60”经常被错认成“限速 80”，从而指导针对性地对这两种样本进行数据增强。*
* **使用阶段**：**测试集最终验收与复盘**。
* **使用方法**：
  ```python
  from sklearn.metrics import confusion_matrix
  import seaborn as sns
  import matplotlib.pyplot as plt

  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.savefig("confusion_matrix.png")
  ```

---

### 二、 核心评价指标详解（连续值回归任务指标）

如果网络任务是连续值数值预测（如房价、风速、坐标回归），指标完全不同：

| 指标 | 物理意义与计算 | 诊断与反映 | 使用阶段 |
| :--- | :--- | :--- | :--- |
| **MAE** (平均绝对误差) | $\frac{1}{N}\sum \|y - \hat{y}\|$ | 直观反映预测值与真实值的平均绝对物理差距，单位与目标一致，对离群点容忍度高。 | 验证 / 测试集 |
| **RMSE** (均方根误差) | $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$ | 相比 MAE，RMSE 对极端大误差有平方放大作用，对**离群大错极度敏感**。 | 验证 / 测试集 |
| **$R^2$ Score** (决定系数) | $1 - \frac{\sum (y-\hat{y})^2}{\sum (y-\bar{y})^2}$ | 取值范围 $\le 1$。越接近 1 说明模型解释方差能力越强；**若 $R^2 < 0$，说明模型表现甚至不如“直接全猜平均值”**。 | 验证 / 测试集 |

---

### 三、 模块化多指标评估工程模板（标准验证/测试写法）

在实际项目中，应在验证阶段（`model.eval()`）将整个 Epoch 的预测结果收集起来，统一计算完整的指标报告：

```python
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate_model(model, dataloader, device):
    """
    针对分类任务的标准多指标评估函数 (用在验证集或测试集)
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 获得预测类别和预测概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # 收集至 CPU 列表中
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 1. 准确率
    acc = accuracy_score(all_targets, all_preds)

    # 2. 精确率、召回率、F1 分数 (macro: 算术平均，对小类别权重平等)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    # 3. 多分类 AUC (需设置 multi_class='ovr')
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan') # 某类别未全部采样到时的异常容错

    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    }
    
    return metrics
```

### 总结选用口诀
1. **看训练过程健康度**：盯紧 **Train Loss vs Val Loss** 曲线。
2. **常规均衡多分类**：看 **Accuracy**。
3. **样本极度不平衡（如反欺诈、故障检测）**：看 **Recall**（防止漏报）、**Precision**（防止误报）与 **F1-Score**。
4. **评估分类排序质量、与阈值无关的底子**：看 **AUC-ROC**。
5. **项目收尾归因找 Bug**：绘制 **Confusion Matrix**。


### 一、 推理（Inference / Testing）模式的核心逻辑与避坑指南

将模型从“训练态”切换到“生产/推理态”，不仅仅是跑一遍 `forward`，它包含严密的数据流与环境配置：

#### 1. 核心构造逻辑（标准五步法）
1. **加载结构与权重**：先实例化与训练时结构完全一致的网络，再通过 `load_state_dict(torch.load(..., map_location=device))` 加载权重。
2. **冻结模型状态（`model.eval()`）**：
   * 必须显式调用！它通知网络：
     * **Dropout**：不再随机丢弃神经元，所有连接激活并按概率缩放。
     * **BatchNorm**：不再动态计算当前 Batch 的均值和方差，改用训练阶段累积固化的全局滑动统计量（Running Mean & Var）。
3. **关闭梯度与优化计算图（`torch.inference_mode()`）**：
   * 在 PyTorch 1.9+ 中，**更推荐使用 `torch.inference_mode()` 代替 `torch.no_grad()`**。它不仅关闭梯度反向传播，还会彻底销毁用于自动求导的张量版本计数器（Version Counter），推理性能更高、显存占用更低。
4. **单张数据升维对齐（Batch 维度扩充）**：
   * 卷积网络强制要求输入为 4 维张量 `(Batch, C, H, W)`。读取单张图片 `(C, H, W)` 后，必须通过 `unsqueeze(0)` 增加虚拟批次维度变成 `(1, C, H, W)`。
5. **后处理与概率解析**：
   * 模型输出是无边界的原始分值（Logits），需通过 `torch.softmax(outputs, dim=-1)` 转为和为 1 的置信度概率分布，再由 `torch.max` 提取最大预测类别及对应置信度。

#### 2. 推理时的核心避坑注意事项
* **预处理与归一化必须同源**：推理时图片的 Resize 方式、插值算法、减均值除方差（Mean & Std）必须与训练时的数据变换（Transforms）**像素级一致**，否则精度会大幅崩塌。
* **GPU 测速的“异步陷阱”**：
  * **致命误区**：在 GPU 上直接用 `time.time()` 包裹推理代码测出来的不是“计算耗时”，仅仅是“把任务丢给 CUDA 驱动的耗时”。
  * **标准做法**：GPU 测速前后必须调用 **`torch.cuda.synchronize()`**，强制等待 GPU 核函数真正执行完毕，计时才真实有效。
* **预热（GPU Warmup）**：GPU 初次加载模型时会有 CUDA 上下文初始化、内存分配开销。测试单次耗时前，通常建议先拿一个 Dummy Tensor 跑 1~2 次“空转预热”。

---

### 二、 独立工程文件扩展：`predict.py`

该文件可作为独立的推理脚本直接运行，支持单文件、整目录遍历、置信度输出、高精度测速，且默认在 CPU 运行，支持指定 GPU。

```python
"""
predict.py: 标准模型推理模板脚本
功能：
1. 默认运行在 CPU，可通过参数指定 GPU。
2. 支持单张图像推理，或自动递归遍历整个图像目录。
3. 输出预测类别、置信度（Confidence）。
4. 包含 GPU 同步的高精度单次推理耗时（Latency）。
"""

import os
import time
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ==========================================
# 1. 示例模型骨干 (需与训练时保持完全一致)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ==========================================
# 2. 推理执行器封装类
# ==========================================
class Predictor:
    def __init__(self, weights_path: str, device: str = 'cpu', in_channels: int = 1, num_classes: int = 10):
        # (d) 默认 CPU，支持按需指定 GPU
        self.device = torch.device(device)
        print(f"[*] 正在初始化推理引擎，当前分配运行设备: {self.device}")

        # 实例化网络并加载权重
        self.model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"[✓] 成功加载模型权重: {weights_path}")
        else:
            print(f"[!] 警告: 未找到权重文件 {weights_path}，使用未训练的初始化权重演示！")

        self.model.to(self.device)
        self.model.eval()  # 核心：必须切换为评估模式

        # 定义严格一致的图像预处理流水线
        # 注意：此处以单通道 28x28 灰度图为例，如果是 RGB 则改成 3 通道和对应的 Resize 尺寸
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # (可选) 模型预热 Warmup
        self._warmup()

    def _warmup(self):
        """消除 CUDA 初始化引起的初次推理延迟偏差"""
        if self.device.type == 'cuda':
            dummy = torch.randn(1, 1, 28, 28, device=self.device)
            with torch.inference_mode():
                for _ in range(2):
                    _ = self.model(dummy)
            torch.cuda.synchronize()

    def predict_single(self, image_path: Path):
        """
        单张图片推理
        :return: (pred_class_id, confidence, latency_ms)
        """
        # 1. 读取并预处理
        image = Image.open(image_path).convert('L')
        img_tensor = self.transform(image)
        # 核心：增加 Batch 维度 (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 2. 推理与高精度计时
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # (c) 保证开始计时前 GPU 无挂起任务
        t0 = time.perf_counter()

        with torch.inference_mode():  # 极致优化的非求导环境
            outputs = self.model(img_tensor)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # (c) 等待 GPU 计算真正完成
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0  # 毫秒耗时

        # 3. (b) 计算置信度 (Softmax 概率化)
        probs = F.softmax(outputs, dim=-1)
        conf, pred_class = torch.max(probs, dim=-1)

        return pred_class.item(), conf.item(), latency_ms

    def predict_path(self, source_path: str):
        """
        (a) 自动识别是单文件还是目录，进行分发推理
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"输入路径不存在: {source_path}")

        # 支持的图片拓展名集合
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # 获取文件列表
        if src.is_file():
            img_list = [src]
        elif src.is_dir():
            img_list = [p for p in src.rglob('*') if p.suffix.lower() in valid_exts]
            if len(img_list) == 0:
                print(f"[!] 目录 {source_path} 下未发现有效格式图片！")
                return
        else:
            return

        print(f"\n{'='*75}")
        print(f"{'文件名':<35} | {'预测类别':<8} | {'置信度':<10} | {'耗时(ms)':<10}")
        print(f"{'-'*75}")

        total_time = 0.0
        for img_p in img_list:
            pred_id, conf, latency = self.predict_single(img_p)
            total_time += latency
            print(f"{img_p.name:<35} | {pred_id:<8} | {conf*100:>6.2f}%   | {latency:>7.2f} ms")

        print(f"{'='*75}")
        print(f"处理完成！共推理 {len(img_list)} 张图片，平均耗时: {total_time / len(img_list):.2f} ms/张\n")


# ==========================================
# 3. 命令行交互入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch 标准图像推理脚本模板")
    # (a) 文件或文件夹路径
    parser.add_argument('--source', type=str, required=True, 
                        help='目标输入：可传入单张图像路径，或包含图像的文件夹路径')
    # 权重路径
    parser.add_argument('--weights', type=str, default='best_model.pth', 
                        help='模型权重文件路径 (.pth)')
    # (d) 默认选择 CPU，可显式指定为 cuda 或 cuda:0
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], 
                        help='推理使用的计算硬件，默认: cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 实例化并执行推理
    predictor = Predictor(weights_path=args.weights, device=args.device)
    predictor.predict_path(args.source)
```

---

### 三、 使用方法与调用示例

#### 1. 命令行调用方式
* **单图片测试（默认 CPU）**：
  ```bash
  python predict.py --source ./data/test_img_01.png
  ```
* **整文件夹遍历测试（使用 GPU 加速）**：
  ```bash
  python predict.py --source ./data/test_folder/ --weights best_model.pth --device cuda
  ```

#### 2. 输出效果终端样式
```text
[*] 正在初始化推理引擎，当前分配运行设备: cuda
[✓] 成功加载模型权重: best_model.pth

===========================================================================
文件名                                 | 预测类别 | 置信度     | 耗时(ms)  
---------------------------------------------------------------------------
digit_7.png                         | 7        |  99.82%   |    1.15 ms
digit_2_blur.png                    | 2        |  87.41%   |    0.89 ms
digit_0.png                         | 0        | 100.00%   |    0.85 ms
===========================================================================
处理完成！共推理 3 张图片，平均耗时: 0.96 ms/张
```