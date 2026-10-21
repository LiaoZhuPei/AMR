在 PyTorch 中，卷积神经网络的核心组件在 **1D（时序、音频、文本序列）** 和 **2D（图像、空间网格）** 下的设计逻辑高度一致，区别主要在于**张量的维度与滑动窗口的方向**。

以下是各个核心层的**输入输出规范**、**核心参数含义**与**工程数值选择指导**。

---

### 2. 卷积层（Convolutional Layer）

负责局部特征提取与模式匹配。

#### ① 张量输入输出格式
* **2D（图像）**：输入 `(Batch, in_channels, H, W)` $\rightarrow$ 输出 `(Batch, out_channels, H_out, W_out)`
  * 如输入图像为单通道灰度图 $28 \times 28$：`(N, 1, 28, 28)`。
* **1D（时序/音频）**：输入 `(Batch, in_channels, L)` $\rightarrow$ 输出 `(Batch, out_channels, L_out)`
  * 如心电信号采样点长度为 1000，双通道导联：`(N, 2, 1000)`。

#### ② 关键参数与数值选择指导

| 参数名 | 含义 | 2D 常用选择与经验准则 | 1D 常用选择与经验准则 |
| :--- | :--- | :--- | :--- |
| **`in_channels`** | 输入通道数 | 初始层取决于数据（灰度图=1，RGB=3）；中间层等于上一层的 `out_channels`。 | 单传感器/单变量时序=1，多变量特征/词向量维度=特征数。 |
| **`out_channels`** | 输出通道数（卷积核数量） | 常用 $2^n$ 递增：**$16 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$**。层数越深特征越抽象，通道数翻倍。 | 常用 $32 \rightarrow 64 \rightarrow 128$。 |
| **`kernel_size`** | 卷积核尺寸 | 绝大多数情况下首选 **`3`**（即 $3 \times 3$），极少情况下首层用 `5` 或 `7`。 | 常用奇数 **`3`、`5`、`7`**（时序跨度更大时可用 `7` 或 `9` 捕捉更长局部上下文）。 |
| **`stride`** | 步长（滑动距离） | **默认选 `1`**（保持空间尺寸）。若用于替代池化做下采样，则设为 **`2`**。 | 默认选 `1`；需要降低采样率/缩短序列时选 `2`。 |
| **`padding`** | 边缘填充 | **“保尺寸公式”**：当 `stride=1` 时，设 **`padding = kernel_size // 2`**。<br>即核为 3 时 `padding=1`；核为 5 时 `padding=2`。 | 同 2D 公式：`padding = kernel_size // 2`。时序因果卷积常用特殊单侧 padding。 |
| **`bias`** | 偏置项 | **若紧接着 `BatchNorm`，务必设为 `False`**（因为 BN 会消除偏置）。无归一化时设 `True`。 | 同 2D 原则。 |

---

### 3. 激活函数（Activation Function）

引入非线性表达能力，使深层网络能拟合复杂函数。

#### ① 张量输入输出格式
* **1D & 2D 通用**：输入与输出形状**完全一致**（逐元素独立计算，不改变张量 Shape）。

#### ② 常用激活函数与选择指导

* **`nn.ReLU(inplace=True)`（首选/基准）**
  * **指导**：CNN 中最标准的默认选择；设置 `inplace=True` 会直接原地覆盖内存，**显著减少显存占用**。
* **`nn.LeakyReLU(negative_slope=0.01)`**
  * **指导**：当发现模型出现“神经元坏死”（Dead ReLU，梯度长时间为 0）时选用。
* **`nn.GELU()`**
  * **指导**：现代骨干网络（如 ConvNeXt、Transformer/ViT）的标配，平滑度高，但在小型 CNN 上相比 ReLU 提升有限且略微增加计算量。
* **输出层专用激活：**
  * **分类网络**：如果损失函数采用 `nn.CrossEntropyLoss()`，**最后一层全连接后面严禁加 Softmax**（PyTorch 内部已集成 LogSoftmax）。
  * **多标签/二分类**：最后一层接 `nn.Sigmoid()`，配合 `nn.BCELoss`；或直接输出 Logits 配合 `nn.BCEWithLogitsLoss`（推荐）。

---

### 4. 池化层（Pooling Layer）

负责降维、减少计算量、扩大感受野并提供一定的平移不变性。

#### ① 张量输入输出格式
* **2D**：输入 `(B, C, H, W)` $\rightarrow$ 输出 `(B, C, H_out, W_out)`。**通道数 C 保持不变，仅压缩空间分辨率**。
* **1D**：输入 `(B, C, L)` $\rightarrow$ 输出 `(B, C, L_out)`。

#### ② 关键函数与数值选择指导

1. **`nn.MaxPool2d` / `nn.MaxPool1d`（最大池化）**
   * **参数**：`kernel_size=2, stride=2`（常用默认）。
   * **效果**：长宽或序列长度**直接减半**（$H/2, W/2$ 或 $L/2$）。用于特征提取阶段的下采样。
2. **`nn.AdaptiveAvgPool2d` / `nn.AdaptiveAvgPool1d`（自适应平均池化，工程核心推荐）**
   * **参数**：
     * 2D：`output_size=(1, 1)`（全局池化）或 `(4, 4)`。
     * 1D：`output_size=1`。
   * **工程指导**：**在卷积层与全连接层交界处，强烈推荐使用自适应池化**。无论输入图片尺寸是 $28\times28$ 还是 $224\times224$，只要指定 `(1, 1)`，输出一定是固定大小，杜绝全连接层维度算错而崩溃。

---

### 5. 归一化层（Normalization Layer）

加速深层网络收敛，防止梯度消失/爆炸，起到轻微正则化效果。

#### ① 张量输入输出格式与参数
* **`nn.BatchNorm2d(num_features)`**
  * **输入**：`(B, C, H, W)`。
  * **参数**：`num_features` **必须等于该位置的特征图通道数 `C`**（即对应卷积的 `out_channels`）。
* **`nn.BatchNorm1d(num_features)`**
  * **输入**：时序 `(B, C, L)` 或 全连接输出 `(B, C)`。
  * **参数**：`num_features` 等于通道数或特征维度 `C`。

#### ② 数值选择与工程避坑指南
* **批大小（Batch Size）敏感性**：
  * 当训练的 **Batch Size $\ge 16$** 时，`BatchNorm` 效果极佳。
  * **避坑**：如果 Batch Size 太小（如显存吃紧，单卡 Batch $\le 4$），BN 统计的均值方差极其不稳定，容易导致模型不收敛，此时应改用 **`nn.GroupNorm`** 或 **`nn.LayerNorm`**。
* **与 Conv 的配合**：
  * 牢记公式：`nn.Conv2d(..., bias=False)` 紧跟 `nn.BatchNorm2d(...)`。

---

### 6. 全连接层（Fully Connected / Linear Layer）

整合多层卷积抽取的局部特征，完成最终的高层分类或回归。

#### ① 张量输入输出格式
* **1D & 2D 通用**：输入 `(Batch, in_features)` $\rightarrow$ 输出 `(Batch, out_features)`。
  * 在送入全连接层前，卷积输出必须经过 `nn.Flatten()` 将多维展开为 1 维。

#### ② 关键参数与数值选择指导

| 参数名 | 含义 | 2D 视觉典型选择 | 1D 时序/序列典型选择 |
| :--- | :--- | :--- | :--- |
| **`in_features`** | 输入神经元个数 | 1. **标准做法**：接在 `AdaptiveAvgPool2d((1, 1))` 之后时，`in_features` **恰好等于最后一个卷积的通道数**。<br>2. **传统做法**：展开扁平化，`in_features = C * H_final * W_final`。 | 接在 `AdaptiveAvgPool1d(1)` 之后时，`in_features` 直接等于最后一层 1D 卷积通道数。 |
| **`out_features`** | 输出神经元个数 | 1. 中间隐藏层常用过渡尺寸：**$128$、$256$、$512$** 等。<br>2. 最终输出层必须**严格等于任务类别数 `num_classes`**（回归任务为 $1$）。 | 最终输出层通常为 `num_classes`（时序分类）或预测步长 $K$（时序预测）。 |
| **`bias`** | 是否带偏置 | 默认且强烈推荐设为 **`True`**（分类决策需要偏移量）。 | 默认设为 **`True`**。 |

---

### 总结对照速查表（1D vs 2D 常用组合模板）

| 组件 | 2D 图像标准规范 (Input: `B, 1, 28, 28`) | 1D 时序标准规范 (Input: `B, 1, 1000`) |
| :--- | :--- | :--- |
| **卷积层** | `nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)` | `nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False)` |
| **归一化** | `nn.BatchNorm2d(32)` | `nn.BatchNorm1d(32)` |
| **激活** | `nn.ReLU(inplace=True)` | `nn.ReLU(inplace=True)` |
| **池化** | `nn.MaxPool2d(kernel_size=2, stride=2)` | `nn.MaxPool1d(kernel_size=2, stride=2)` |
| **过渡池化**| `nn.AdaptiveAvgPool2d((1, 1))` | `nn.AdaptiveAvgPool1d(1)` |
| **扁平化** | `nn.Flatten()` (输出形状: `B, 32`) | `nn.Flatten()` (输出形状: `B, 32`) |
| **分类头** | `nn.Linear(32, 10)` | `nn.Linear(32, 2)` |