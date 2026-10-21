### 一、 经典常用的损失函数与优化器速查手册

在模型训练的循环中，标准模板通常遵循：
```python
optimizer.zero_grad()               # 1. 清空旧梯度
outputs = model(inputs)            # 2. 前向传播
loss = criterion(outputs, targets) # 3. 计算标量损失
loss.backward()                    # 4. 反向传播累积梯度
optimizer.step()                   # 5. 更新参数权重
```

---

#### 1. 常用经典损失函数（Criterion）

| 损失函数 | 典型适用场景 | 常用参数与取值指导 | 功能描述与避坑注意 |
| :--- | :--- | :--- | :--- |
| **`nn.CrossEntropyLoss`** | **多分类**（如 MNIST, CIFAR, ImageNet） | • `weight=None`：可传入各类别权重 Tensor 处理类别不平衡。<br>• `reduction='mean'`：可选 `'mean'`, `'sum'`, `'none'`。<br>• `label_smoothing=0.0`：可设为 `0.1` 提升泛化能力。 | **功能**：集成了 `LogSoftmax` + `NLLLoss`。<br>**输入要求**：`pred` 是未经 Softmax 的 **Logits**（形状 `(N, C)`，浮点型）；`target` 是**类别索引**（形状 `(N,)`，必须为 **`torch.long`**）。 |
| **`nn.BCEWithLogitsLoss`** | **二分类** 或 **多标签分类**（一张图可同时包含猫和狗） | • `pos_weight=None`：正样本权重，用于极度不平衡场景。<br>• `reduction='mean'`。 | **功能**：集成了 `Sigmoid` + `BCELoss`，数值稳定性远高于手动写 `Sigmoid` + `BCELoss`。<br>**输入要求**：`pred` 和 `target` 的 shape 必须完全一致（通常均为 `(N, 1)` 或 `(N, C)`），且 `target` 必须为 **浮点型（`float`）**。 |
| **`nn.MSELoss`** | **连续值回归**（均方误差 / L2 Loss） | • `reduction='mean'`：返回平均误差。 | **功能**：计算预测值与真实值的平方差。对大误差惩罚极大（离群点敏感）。<br>**输入要求**：`pred` 与 `target` 形状严格一致，均为浮点型。 |
| **`nn.L1Loss`** | **连续值回归**（平均绝对误差 / MAE） | • `reduction='mean'`。 | **功能**：对异常点/离群值鲁棒，梯度是恒定的，但收敛到最优点附近可能震荡。 |
| **`nn.SmoothL1Loss`** (Huber Loss) | **目标检测边界框回归**、高鲁棒性回归任务 | • `beta=1.0`：阈值。在误差小于 `beta` 时表现为 L2，大于时表现为 L1。 | **功能**：结合了 MSE（平滑收敛）与 L1（抑制极端离群点梯度爆炸）的优点。 |

---

#### 2. 常用经典优化器（Optimizer）

| 优化器 | 关键超参数及经验推荐范围 | 适用场景与工程经验 |
| :--- | :--- | :--- |
| **`torch.optim.SGD`**<br>(含 Momentum & Nesterov) | • `lr` (学习率): `0.001` ~ `0.1`（常用 `0.01` 起手）<br>• `momentum`: 动量因子，通常设 **`0.9`** 或 `0.99`<br>• `weight_decay` (L2正则化): `1e-4` ~ `1e-5`<br>• `nesterov=False`: 可选 `True`（配合动量使用加速收敛） | **场景**：计算机视觉（CV）中经典模型（如 ResNet）微调与最终冲榜首选。<br>**特点**：收敛较慢，但泛化性（Generalization）往往上限最高；对学习率调节（配合 Cosine 退火策略）要求高。 |
| **`torch.optim.AdamW`**<br>(现代深度学习标配基线) | • `lr`: `1e-4` ~ `3e-3`（标准起始值 **`1e-3`** 或 `3e-4`）<br>• `betas=(0.9, 0.999)`：一阶、二阶动量系数，极少改动<br>• `eps=1e-8`：防除零极小项<br>• `weight_decay`: 推荐 **`0.01` ~ `0.1`** | **场景**：Transformer、NLP、小型 CNN、自监督学习、强化学习的首选。<br>**特点**：**AdamW 解除了原版 Adam 中 Weight Decay 与自适应梯度的耦合**。调参容错率极高，开箱即用。 |

---

### 二、 自定义损失函数（Custom Loss）

#### 1. 接入网络的核心要求与铁律
1. **输入与输出格式**：
   * 输入参数建议定义为 `forward(self, pred, target)`。
   * 输出**必须是包含计算图的 PyTorch Tensor**（通常在 `reduction='mean'` 下输出为 0 维标量 Tensor，如 `tensor(0.5342, requires_grad=True)`）。
2. **计算图连通性（禁止破坏梯度链）**：
   * 损失计算内部必须使用 **PyTorch 原生算子（Tensor API / `torch.*`）**。
   * **严禁转换数据类型跳出计算图**：绝对不能出现 `.numpy()`、`float(x)`、`int(x)` 等操作，否则反向传播无法追踪梯度。
3. **数值稳定性**：
   * 遇到除法需加保护项（如 `pred / (norm + 1e-8)`）。
   * 遇到对数需加截断（如 `torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)`），避免出现 `log(0) -> NaN` 破坏网络权重。

#### 2. 自定义损失函数指导模板（以解决样本不均衡的 Focal Loss 为例）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    针对分类任务样本不均衡的 Focal Loss 模板:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param pred: 模型未经过激活的 Logits，形状 (N, C)
        :param target: 真实标签索引，形状 (N,)，类型为 torch.long
        :return: 损失标量 Tensor
        """
        # 1. 维度与设备类型校验（保证容错）
        assert pred.device == target.device, "预测值与目标值必须位于同一计算设备 (CPU/GPU)！"
        
        # 2. 纯 PyTorch 算子保持计算图连通
        # 使用 log_softmax + gather 保证对数计算的数值稳定性
        log_prob = F.log_softmax(pred, dim=-1)
        prob = torch.exp(log_prob)
        
        # 提取真实类别对应的预测概率值
        target = target.view(-1, 1)
        log_pt = log_prob.gather(1, target).view(-1)
        pt = prob.gather(1, target).view(-1)

        # 3. 计算 Focal 权重
        focal_weight = self.alpha * torch.pow((1.0 - pt), self.gamma)
        loss = -focal_weight * log_pt

        # 4. 根据 reduction 决定最终输出（必须为 PyTorch 标量 Tensor）
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

# === 使用演示 ===
criterion = FocalLoss(alpha=0.25, gamma=2.0)
logits = torch.randn(4, 10, requires_grad=True) # 4个样本，10分类
labels = torch.tensor([1, 5, 2, 8], dtype=torch.long)
loss = criterion(logits, labels)
loss.backward() # 能够成功反向传播
```

---

### 三、 自定义优化器（Custom Optimizer）

#### 1. 接入网络的核心要求与铁律
1. **必须继承基类 `torch.optim.Optimizer`**：
   * 必须在 `__init__` 中将外部超参数整合成字典 `defaults`，并显式调用 `super().__init__(params, defaults)`。
2. **就地更新参数与禁用自动求导**：
   * 参数更新过程**不能追踪计算图**，必须用 `@torch.no_grad()` 装饰器修饰 `step()` 函数。
   * 更新必须是**原地操作（In-place）**（如 `p.add_()`、`p.mul_()`、`p.data.copy_()`），严禁写成 `p = p - lr * grad`（这样会切断张量对象的内部指向，无法更新模型参数）。
3. **状态持久化（State Management）**：
   * 如果优化器需要记录历史状态（如动量累积、二阶方差移动平均），使用基类字典 **`self.state[p]`** 来存储，确保模型状态能被 `state_dict()` 正确序列化与加载。
4. **支持 `closure` 回调函数**：
   * `step(self, closure=None)` 必须支持 `closure` 参数，某些二阶优化算法（如 L-BFGS）需要反复重新评估损失。

#### 2. 自定义优化器指导模板（以实现带梯度裁剪与动量的 CustomSGD 为例）

```python
import torch
from torch.optim.Optimizer import Optimizer

class CustomClampedSGD(Optimizer):
    """
    通用自定义优化器模板:
    实现带有梯度截断 (Gradient Clamping) 与动量 (Momentum) 的随机梯度下降
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, max_grad_norm=1.0):
        # 1. 参数合法性断言
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        # 2. 将默认参数打包为 dict 传给父类初始化
        defaults = dict(lr=lr, momentum=momentum, max_grad_norm=max_grad_norm)
        super(CustomClampedSGD, self).__init__(params, defaults)

    @torch.no_grad() # 核心: 更新参数过程严禁追踪梯度
    def step(self, closure=None):
        """
        标准的优化器 step 框架
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 3. 遍历所有的参数组 (param_group)
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            max_grad_norm = group['max_grad_norm']

            # 4. 遍历当前组中的每一个参数张量 p
            for p in group['params']:
                # 若参数无需梯度或尚未完成 backward，直接跳过
                if p.grad is None:
                    continue

                grad = p.grad

                # 自定义计算逻辑 1: 梯度截断 (防止梯度爆炸)
                if max_grad_norm is not None:
                    grad = torch.clamp(grad, -max_grad_norm, max_grad_norm)

                # 自定义计算逻辑 2: 获取历史状态 (利用 self.state 维护)
                state = self.state[p]
                
                # 初始化状态缓存
                if len(state) == 0:
                    # 动量缓冲区初始全为 0，设备和形状与 p 一致
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state['momentum_buffer']
                
                # 核心动量更新公式: v_{t} = momentum * v_{t-1} + grad
                buf.mul_(momentum).add_(grad)

                # 核心参数更新公式: p = p - lr * v_{t} (必须是 in-place 操作)
                p.add_(buf, alpha=-lr)

        return loss

# === 使用演示 ===
model = torch.nn.Linear(10, 2)
optimizer = CustomClampedSGD(model.parameters(), lr=0.01, momentum=0.9, max_grad_norm=5.0)

# 正常调用
optimizer.zero_grad()
x = torch.randn(8, 10)
out = model(x).sum()
out.backward()
optimizer.step()
```
在多任务学习（如目标检测中的“分类+定位”、自监督学习、生成对抗网络）中，组合多个损失函数极其常见。

我们以典型的**多任务场景（3个损失函数）**为例：
* $\mathcal{L}_1$（主分类任务，如 `CrossEntropyLoss`）
* $\mathcal{L}_2$（坐标/数值回归任务，如 `SmoothL1Loss`）
* $\mathcal{L}_3$（特征辅助/正则化约束，如余弦相似度或重构 `MSELoss`）

---

### 一、 多个损失函数组合的核心条件与注意事项

#### 1. 数量级失衡（Loss Scale Imbalance）——最致命的问题
* **现象**：假设分类损失 $\mathcal{L}_1 \approx 2.0$，辅助约束损失 $\mathcal{L}_3 \approx 0.001$。若直接相加，$\mathcal{L}_1$ 的梯度将彻底吞没 $\mathcal{L}_3$；反之，若 $\mathcal{L}_2 \approx 100.0$，模型将只专注于回归，完全放弃分类。
* **规则**：**在训练初始阶段，务必打印各个子 Loss 的初始数值大小**，通过设置权重因子 $\lambda_1, \lambda_2, \lambda_3$，让各个子 Loss 经过加权后的**有效数值保持在同一数量级（通常建议在 $0.1 \sim 10$ 之间）**。

#### 2. 计算图与反向传播机制（禁止多次随意 backward）
* **错误写法**：
  ```python
  # 极其低效且容易报错
  loss1.backward()
  loss2.backward() # 报错：RuntimeError: Trying to backward through the graph a second time...
  ```
* **正确规则**：**将各个子 Loss 加权求和得到总损失（Scalar Tensor）后，仅执行一次 `total_loss.backward()`**。PyTorch 的 Autograd 机制会自动将梯度按链式法则在共享特征层进行线性累加。

#### 3. 内存泄漏避坑（Logging 必须使用 `.item()` 或 `.detach()`）
* **陷阱**：为了观察训练情况，我们经常需要把各个 Loss 记录到 TensorBoard、WandB 或字典中。
* **规则**：如果直接保存 Tensor（例如 `loss_dict['l1'] = loss1`），该 Tensor 会**带着完整的动态计算图常驻显存**，导致训练几个 Epoch 后直接 **CUDA Out of Memory (OOM)**。
* **写法**：记录日志时**必须**转换为 Python 浮点数：`loss1.detach().item()`。

#### 4. 梯度冲突（Gradient Conflict / 负迁移）
* **现象**：当共享 Backbone（特征提取层）同时优化多个任务时，任务 A 的梯度方向与任务 B 的梯度方向夹角可能大于 90°（即 $\mathbf{g}_A \cdot \mathbf{g}_B < 0$），两个任务在共享层“互相打架”。
* **应对**：
  * **结构解耦**：缩短共享层深度，增加各个任务独享的分支深度（Task-Specific Heads）。
  * **梯度截断（Detach）**：如果 $\mathcal{L}_3$ 仅仅是辅助监督，不希望它干扰主干特征，可使用 `x.detach()` 截断梯度回传。

---

### 二、 规范的“复合损失函数”自定义工程模板

下面将 3 个损失函数封装为一个高容错、带日志记录、支持动态权重的 `nn.Module`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCompoundLoss(nn.Module):
    """
    组合损失函数标准模板：包含 3 个子任务损失
    L_total = w_cls * L_cls + w_reg * L_reg + w_aux * L_aux
    """
    def __init__(self, w_cls=1.0, w_reg=2.0, w_aux=0.5):
        super(MultiTaskCompoundLoss, self).__init__()
        # 1. 损失权重定义
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.w_aux = w_aux

        # 2. 实例化各个子损失模块
        self.loss_cls_fn = nn.CrossEntropyLoss()
        self.loss_reg_fn = nn.SmoothL1Loss()
        self.loss_aux_fn = nn.MSELoss()

    def forward(self, preds: dict, targets: dict):
        """
        :param preds: 模型输出的字典，包含各类预测结果
        :param targets: 标签字典，包含各个任务的真值
        :return: (total_loss, loss_details)
        """
        # ==================== 1. 分别计算各子损失 ====================
        # 子损失 1：分类 (预测为 Logits, 真实值为类别序号)
        l_cls = self.loss_cls_fn(preds['cls_logits'], targets['cls_labels'])

        # 子损失 2：回归 (预测值与真值形状一致)
        l_reg = self.loss_reg_fn(preds['reg_coords'], targets['reg_coords'])

        # 子损失 3：特征辅助约束 (如特征重建或潜在空间对齐)
        l_aux = self.loss_aux_fn(preds['feat_repr'], targets['target_feat'])

        # ==================== 2. 加权合成总损失 ====================
        total_loss = (self.w_cls * l_cls) + (self.w_reg * l_reg) + (self.w_aux * l_aux)

        # ==================== 3. 打包监控字典 (必须 detach 转数值) ====================
        # 供外部 Tensorboard/日志安全打印，杜绝 OOM 显存泄露
        loss_details = {
            'loss_total': total_loss.detach().item(),
            'loss_cls': l_cls.detach().item(),
            'loss_reg': l_reg.detach().item(),
            'loss_aux': l_aux.detach().item()
        }

        # 核心：必须返回保持计算图的 total_loss 供 backward 使用
        return total_loss, loss_details
```

---

### 三、 训练循环中的调用规范

```python
# 初始化
criterion = MultiTaskCompoundLoss(w_cls=1.0, w_reg=2.0, w_aux=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for data, labels in dataloader:
    optimizer.zero_grad()
    
    # 模型前向传播返回多分支字典
    outputs = model(data) # {'cls_logits': ..., 'reg_coords': ..., 'feat_repr': ...}
    
    # 一次性计算总损失与记录指标
    loss, loss_metrics = criterion(outputs, labels)
    
    # 仅需对总损失 backward，PyTorch 内部会自动分配梯度
    loss.backward()
    optimizer.step()
    
    # 安全记录，绝不泄露显存
    # print(f"Batch Loss: {loss_metrics['loss_total']:.4f}, Cls: {loss_metrics['loss_cls']:.4f}")
```

---

### 四、 自定义时的高级进阶技巧（动态/可学习权重）

手动调节多个 $\lambda$（超参数搜索）通常很耗时，如果不想手动调参，可以使用 CVPR 经典的**基于不确定性的可学习权重（Uncertainty Weighting）**：

* **原理**：将各任务的方差（不确定性）作为网络的可学习参数 $\sigma_i$，让网络自动平衡任务权重，公式形式为：
  $$\mathcal{L}_{\text{total}} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$$
* **代码实现提示**：
  ```python
  # 将权重包装为 nn.Parameter，直接注册进优化器参与训练
  self.log_var1 = nn.Parameter(torch.zeros(1))
  self.log_var2 = nn.Parameter(torch.zeros(1))
  self.log_var3 = nn.Parameter(torch.zeros(1))

  # 在 forward 中动态平衡（加入 log 项防止方差趋向于正无穷）：
  loss = (torch.exp(-self.log_var1) * l1 + self.log_var1 +
          torch.exp(-self.log_var2) * l2 + self.log_var2 +
          torch.exp(-self.log_var3) * l3 + self.log_var3)
  ```
  这样就免去了繁琐的手动调试 $\lambda_1, \lambda_2, \lambda_3$ 的工作。