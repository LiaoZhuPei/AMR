import os
import torch
import torch.nn as nn
from torchsummary import summary

class CapsuleNetworkPyTorch(nn.Module):
    def __init__(self, input_channels=2, input_length=1024, num_classes=12):
        """
        用于数字调制分类的胶囊网络。
        - 已为 1024 长度的输入信号调整了架构。
        - 输出类别的数量是可配置的。

        参数:
            input_channels (int): 输入通道数 (I/Q为2)。
            input_length (int): 输入信号的长度。
            num_classes (int): 输出类别的数量 (应与数据集匹配)。
        """
        super(CapsuleNetworkPyTorch, self).__init__()
        self.num_classes = num_classes

        # --- 特征提取前端 (CNN FE) ---
        # 使用为 1024 长度输入调整过的步长，以避免特征图过度缩小
        k, s = 22, 4
        pad = k - 1
        pad_l, pad_r = pad // 2, pad - (pad // 2)
        self.feature_extractor = nn.Sequential(
            nn.ConstantPad1d((pad_l, pad_r), 0),
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=k, stride=s, padding=0),
            nn.BatchNorm1d(64),
            nn.Tanh()
        )
        
        # 辅助函数，用于动态计算进入全连接层之前的展平尺寸
        dummy_input = torch.randn(1, input_channels, input_length)
        self._flattened_size = self._get_flattened_size(dummy_input)

        # --- 并行分支 (初级胶囊) ---
        # 分支的数量现在由 num_classes 决定
        self.branches = nn.ModuleList()
        for _ in range(self.num_classes):
            # 分支内的 CNN 1
            k1, s1 = 23, 3
            pad1 = k1 - 1
            pad1_l, pad1_r = pad1 // 2, pad1 - (pad1 // 2)
            cnn1 = nn.Sequential(
                nn.ConstantPad1d((pad1_l, pad1_r), 0),
                nn.Conv1d(in_channels=64, out_channels=48, kernel_size=k1, stride=s1, padding=0),
                nn.BatchNorm1d(48),
                nn.Tanh()
            )

            # 分支内的 CNN 2
            k2, s2 = 22, 4
            pad2 = k2 - 1
            pad2_l, pad2_r = pad2 // 2, pad2 - (pad2 // 2)
            cnn2 = nn.Sequential(
                nn.ConstantPad1d((pad2_l, pad2_r), 0),
                nn.Conv1d(in_channels=48, out_channels=64, kernel_size=k2, stride=s2, padding=0),
                nn.BatchNorm1d(64),
                nn.Tanh()
            )

            # 平均池化和全连接层
            k_p = 4
            p_p = (k_p - 1) // 2
            branch_tail = nn.Sequential(
                nn.AvgPool1d(kernel_size=k_p, stride=1, padding=p_p),
                nn.Flatten(),
                nn.Linear(in_features=self._flattened_size, out_features=32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=1)
            )

            # 组合分支的所有部分
            full_branch = nn.Sequential(cnn1, cnn2, branch_tail)
            self.branches.append(full_branch)

        # --- 分类层 ---
        self.softmax = nn.Softmax(dim=1)

    def _get_flattened_size(self, dummy_input):
        """
        辅助函数，用于计算卷积/池化部分的输出维度。
        """
        x = self.feature_extractor(dummy_input)
        
        # 定义一个临时的分支结构，仅用于尺寸计算
        k1, s1 = 23, 3
        pad1 = k1 - 1
        pad1_l, pad1_r = pad1 // 2, pad1 - (pad1 // 2)
        cnn1_calc = nn.Sequential(
            nn.ConstantPad1d((pad1_l, pad1_r), 0),
            nn.Conv1d(in_channels=64, out_channels=48, kernel_size=k1, stride=s1, padding=0)
        )

        k2, s2 = 22, 4
        pad2 = k2 - 1
        pad2_l, pad2_r = pad2 // 2, pad2 - (pad2 // 2)
        cnn2_calc = nn.Sequential(
            nn.ConstantPad1d((pad2_l, pad2_r), 0),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=k2, stride=s2, padding=0)
        )

        k_p = 4
        p_p = (k_p - 1) // 2
        pool_calc = nn.AvgPool1d(kernel_size=k_p, stride=1, padding=p_p)

        x = cnn1_calc(x)
        x = cnn2_calc(x)
        x = pool_calc(x)
        output = nn.Flatten()(x)
        
        return output.shape[1]

    def forward(self, x):
        """
        定义模型的前向传播路径。
        """
        # 1. 通过公共的特征提取器
        features = self.feature_extractor(x)
        
        # 2. 将特征并行送入所有分支
        branch_outputs = [branch(features) for branch in self.branches]
            
        # 3. 连接所有分支的输出
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # 4. 应用 Softmax 得到分类概率
        output = self.softmax(concatenated)
        
        return output
    
def build_model(weights_path=None, input_length=1024, num_classes=12,**kwargs):
    """
    辅助函数，用于构建模型并可选择性地加载权重。
    """
    model = CapsuleNetworkPyTorch(
        input_length=input_length, 
        num_classes=num_classes
    )

    if weights_path:
        if not os.path.exists(weights_path):
            raise ValueError(f'The `weights_path` argument provided does not exist: {weights_path}')
        # 加载PyTorch模型的state_dict
        model.load_state_dict(torch.load(weights_path))
        print(f"Model weights loaded from {weights_path}")

    return model
