import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
# 数据加载
# dataset读数据{标签和数据分开的}{标签和数据一起的}{npy,pkl,csv,wav,dat}
# DataLoader加载数据，划分数据集，随机打乱，num_worker

# 数据预处理

# 开启GPU

class DEEPLEARN(nn.Module):
    def __init__(self, inpuit_channel):
        #super
        super().__init__()
        self.input_channnel = inpuit_channel
        self.conv1 = nn.Conv2d(self.input_channel, 16, padding=1, stride=1) # 各个参数的意义； 怎么构建可复用卷积块
        self.Line = nn.linear(16,4)
    def forward(self, x):
        out = self.conv1(x) # 卷积块
        out = F.Batchnorm(out)
        out = F.relu(out)
        out = out.flatten(16, 1) # 卷积的输出怎么展开成线形层的输入
        out = self.line(out)

        return out

model = DEEPLEARN(input_channel=8)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(lr=0.01)

epoch = 100

#早停
#记录训练过程
#绘制loss曲线
#保存最佳权重
#数据集的加载和迭代的关系
for i in epoch:
    for data, labels in train_dataloader:
        #开启训练模式
        model.train()
        # 输入模型
        # 计算损失
        # 训练记录
        # 优化
        # 保存最佳权重判断

model.eval()
# 加载最佳权重
# 输入模型
# 输出预测结果





