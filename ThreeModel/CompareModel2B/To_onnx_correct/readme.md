这次的警告其实是**无害的、预期中的**（后面详细解释），所以依然出现并不意味着修复失败。但既然 NaN 仍然复现，我们需要往更深一层去找——我现在有一个新的、更具体的怀疑对象，而且这次能解释"为什么只有这份真实数据在 ONNX 下出问题，PyTorch 却完全正常"。

---

## 一、先澄清：这两条 TracerWarning 不是问题根源，可以忽略

```
qi = [int(q * (L - 1)) for q in (...)]
step = max(1, L // 256)
```

这两条警告的含义是："序列长度 L 被当作常量（1024）固化进了计算图"。只要你**所有推理输入都严格是 1024 长度**（你的 C++ 代码和推理脚本都做了这个校验），这个警告就只是提示信息，不影响数值正确性。它和 NaN 问题**不是同一件事**，不需要为了消除这个警告再改代码。

---

## 二、新的、更精准的怀疑对象：`atan2` 在 `(0, 0)` 点的行为差异

### 关键线索重新梳理

你说的是：**用 128QAM 的真实数据**，PyTorch 端完全正常，ONNX 端出 NaN。这是一个非常重要的区别——**不是用随机数据复现的，而是用了具体的、真实的信号样本**。这意味着问题很可能和这份数据里的**某个特殊数值**有关，而不是普遍性的计算逻辑错误（因为如果是通用逻辑错误，应该对所有数据都会复现，包括随机数据）。

### 核心怀疑：`torch.atan2(Q, I)` 在 `I=0, Q=0` 时的处理

你的代码里有两处用到 `atan2`：

```python
phase = torch.atan2(Q, I)
...
dp = torch.atan2(torch.sin(dp), torch.cos(dp))
```

**PyTorch 原生的 `atan2(y, x)` 对 `(0, 0)` 这个特殊点有明确定义**（按照 IEEE 754 标准，`atan2(0, 0) = 0`），这是一个被内部特殊处理（hard-coded）的边界情况，不是通过 `y/x` 做除法算出来的。

但是，当高版本 PyTorch 把 `atan2` 导出为 ONNX 时，它是**用别的基础算子组合分解实现**的，这个分解公式的核心通常是：

```
atan2(y, x) ≈ atan(y / x)   # 再根据 x, y 的符号做象限修正
```

**如果 `x` 和 `y`同时为 0**，那么 `y / x = 0 / 0`，这在浮点运算中结果是 **NaN**（而不是 0！）。也就是说：

- **PyTorch 原生 `atan2(0, 0)` → 返回 `0`**（正确、符合预期）
- **ONNX 分解后的 `atan2(0, 0)` → 可能返回 `NaN`**（因为内部做了 `0/0` 的除法）

### 为什么只有这份 128QAM 数据会触发，而不是所有数据

这完全取决于**这份具体信号里，是否存在某个采样点恰好 `I=0` 且 `Q=0`**（或者在 `power_normalize` 之后，某个点因为极低幅度/量化误差导致恰好是 `(0,0)`）。这种情况在真实采集的信号中是完全可能出现的——比如：

- 信号包络过零点附近的采样
- ADC 量化导致的精确 0 值
- 滤波器群时延/边沿效应导致个别采样点幅度为 0

**只要 1024 个点里有一个点满足 `(I,Q)=(0,0)`，这个 NaN 就会从 `phase` 这个中间变量产生，然后通过后续的 `dp`、`f_phase_std`、`f_phase_mean` 一路传播，最终污染整个 42 维特征向量，再经过 `torch.cat` 污染整个输出，表现为你看到的"全部输出都是 NaN"。**

这个解释完美对应你观察到的现象：**同一个模型结构、同一套代码逻辑，PyTorch 正常而 ONNX 异常，且具体某份数据触发、不是普遍性现象**。

---

## 三、修复方案：手写一个在 `(0,0)` 处明确返回 0 的安全版 `atan2`

不要依赖 ONNX 导出器自动分解 `atan2`（它的边界处理行为你无法控制，且在不同 PyTorch/opset 版本间可能不一致）。改为自己实现一个**显式处理 `(0,0)` 特殊情况**的版本，确保 PyTorch 和 ONNX 导出后行为完全一致。

在 `stat_features.py` 文件顶部添加：

```python
def safe_atan2(y, x, eps=1e-12):
    """
    数值稳定、ONNX导出安全的 atan2 实现。
    显式处理 (0,0) 情况，避免 0/0 产生 NaN。
    """
    # 用一个极小的偏移量防止除以精确的0
    x_safe = torch.where(
        (x.abs() < eps) & (y.abs() < eps),
        torch.full_like(x, eps),   # (0,0)点：给x一个极小正数，使atan(y/x)=atan(0/eps)=0
        x
    )
    atan_val = torch.atan(y / (x_safe + torch.sign(x_safe) * eps + eps))

    result = torch.where(
        x > 0, atan_val,
        torch.where(
            (x < 0) & (y >= 0), atan_val + torch.pi,
            torch.where(
                (x < 0) & (y < 0), atan_val - torch.pi,
                torch.where(
                    (x == 0) & (y > 0), torch.full_like(x, torch.pi / 2),
                    torch.where(
                        (x == 0) & (y < 0), torch.full_like(x, -torch.pi / 2),
                        torch.zeros_like(x)   # x==0 and y==0 → 显式返回0
                    )
                )
            )
        )
    )
    return result
```

然后把文件里两处 `torch.atan2` 的调用替换掉：

```python
# 原来:
# phase = torch.atan2(Q, I)
phase = safe_atan2(Q, I)

# 原来:
# dp = torch.atan2(torch.sin(dp), torch.cos(dp))
dp = safe_atan2(torch.sin(dp), torch.cos(dp))
```

**关于这个实现的说明**：
- 用 `torch.where` 枚举所有分支（`x>0`、`x<0且y≥0`、`x<0且y<0`、`x=0且y>0`、`x=0且y<0`、`x=0且y=0`），每个分支都是显式的数学表达式，不依赖 ONNX 导出器自己去猜测怎么分解。
- 关键的 `(x==0, y==0)` 分支被**显式写成 `torch.zeros_like(x)`**，不涉及任何除法，从根本上杜绝了 `0/0` 产生 NaN 的可能。
- 第二个 `atan2(sin(dp), cos(dp))` 的调用中，`sin(dp)` 和 `cos(dp)` 理论上不会同时为 0（因为 `sin²+cos²=1`），所以这处风险本身较小，但统一用 `safe_atan2` 替换，保证两处行为一致、代码更健壮。

---

## 四、如何验证这个假设是否正确（强烈建议先做这步，再决定要不要应用修复）

在应用修复之前，建议你先**实锤定位**，而不是直接改完就重新导出（这样万一没解决，你不知道是假设错了还是修复没生效）。

### 验证脚本：直接检查这份真实数据里是否存在 `(I,Q)=(0,0)` 的点

```python
import h5py
import numpy as np

# 换成你实际触发NaN的那个128QAM文件
file_path = 'xxx.mat'

with h5py.File(file_path, 'r') as f:
    signal = f['data_save'][:]  # (1, 2, 1024)

I = signal[0, 0, :]
Q = signal[0, 1, :]

# 做和推理脚本一致的功率归一化
power = np.mean(I**2 + Q**2)
I_n = I / np.sqrt(power)
Q_n = Q / np.sqrt(power)

# 检查是否存在精确的 (0,0) 点，或者极度接近0的点
zero_mask = (np.abs(I_n) < 1e-7) & (np.abs(Q_n) < 1e-7)
print("是否存在(I,Q)≈(0,0)的点:", zero_mask.any())
print("具体位置:", np.where(zero_mask))
print("该点附近的I,Q值:", I_n[zero_mask], Q_n[zero_mask])
```

如果这里打印出 `True` 并且有具体的点，那基本就实锤了——**这份数据里确实存在导致 `atan2` 退化为 `0/0` 的点**。

### 进一步验证：用这个具体点手动构造最小复现用例

```python
import torch

# 构造一个最小测试：把这个有问题的点单独喂给atan2
y = torch.tensor([Q_n[zero_mask][0]])
x = torch.tensor([I_n[zero_mask][0]])

print("PyTorch原生 atan2:", torch.atan2(y, x))

# 导出一个只有atan2的微型模型，看ONNX怎么算这个点
class Atan2Test(torch.nn.Module):
    def forward(self, y, x):
        return torch.atan2(y, x)

m = Atan2Test()
torch.onnx.export(m, (y, x), "atan2_test.onnx",
                   input_names=['y', 'x'], output_names=['out'],
                   opset_version=13)

import onnxruntime as ort
sess = ort.InferenceSession("atan2_test.onnx")
onnx_result = sess.run(None, {'y': y.numpy(), 'x': x.numpy()})
print("ONNX atan2:", onnx_result)
```

如果 PyTorch 输出是一个正常数值（比如 `0.0` 或其他具体角度），而 ONNX 输出是 `nan`，**这就是 100% 实锤的根因**，可以放心应用上面第三部分的修复方案。

---

## 五、如果验证后发现并不是这个问题

如果上面的检查显示这份数据里**并不存在** `(0,0)` 附近的点，那么需要用之前提到的"中间特征对拍"方法（把 `HigherOrderStats._extract` 单独导出，对比 PyTorch 和 ONNX 的 42 维原始特征，逐列定位 NaN 出现在哪一维），这样能更精确地缩小范围。考虑到你已经修复了直方图和切片两处，如果 `atan2` 也排除了，剩下的嫌疑点按风险排序是：

1. `f_sigma_a = a4 / (a2.pow(2) + 1e-10)` 和 `f_sigma_6 = a6 / (a2.pow(3) + 1e-10)`：如果信号幅度 `a2` 在 ONNX 下的浮点计算路径和 PyTorch 略有差异（比如算子融合顺序不同导致的浮点误差累积），极端情况下可能放大误差，但这个可能性较低。
2. `torch.cuda.amp.autocast(enabled=False)` 这个上下文管理器在 ONNX 导出时的行为——**这个其实也值得怀疑**，因为 `autocast` 相关的上下文管理器在 trace 模式下有时不会被正确处理，建议导出时确认这里是否引入了意外的精度转换。可以临时去掉这个 context 测试一下（不影响数值结果，只是精度策略）：

```python
def forward(self, x):
    with torch.no_grad():  # 临时去掉 autocast，只保留 no_grad
        feats = self._extract(x.float())
    return self.proj(feats)
```

---

## 建议的执行顺序

1. **先跑第四部分的验证脚本**，确认这份 128QAM 数据里是否真的存在 `(I,Q)≈(0,0)` 的点，以及用最小复现用例验证 `atan2` 在 ONNX 下对这个点是否真的输出 NaN。
2. 如果验证通过 → 应用第三部分的 `safe_atan2` 修复，替换文件里两处调用，重新导出，用"PyTorch vs ONNX 同一输入对比"的方法确认 NaN 消失。
3. 如果验证不通过 → 按第五部分，做完整的 42 维中间特征对拍，把结果发给我，我们继续往下缩小范围。

把第四部分验证脚本的输出结果发给我，这样能最快确认问题根源。
