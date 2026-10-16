import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# ================= 配置区域 =================
# [修改] 你的模型路径
MODEL_PATH = "./training_results_tensor_Ultimate/best_model_Ultimate.pth"

# [修改] 你的实测样本路径 (请替换为真实的 A 和 B 的 .pt 文件路径)
# 你可以在 dataset_tri_RAD_Tensor_Ultimate/real_test/QPSK/ 下找到生成的 .pt
SAMPLE_A_PATH = "./dataset_tri_RAD_Tensor_Ultimate/real_test/QPSK/real_0000.pt" # 假设这是能识别对的
SAMPLE_B_PATH = "./dataset_tri_RAD_Tensor_Ultimate2/real_test/QPSK/real_0001.pt" # 假设这是识别错的

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]

# ================= 模型定义 (必须与训练时一致) =================
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# ================= Grad-CAM 工具类 =================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. 前向传播
        self.model.zero_grad()
        output = self.model(x) # (1, 12)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        # 2. 反向传播
        score = output[0, class_idx]
        score.backward()
        
        # 3. 生成 CAM
        gradients = self.gradients # (1, 64, 8, 8)
        activations = self.activations # (1, 64, 8, 8)
        
        # Global Average Pooling of Gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (1, 64, 1, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (1, 1, 8, 8)
        
        # ReLU
        cam = F.relu(cam)
        
        # Upsample to input size (64x64)
        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-9)
        
        return cam.squeeze().cpu().detach().numpy(), class_idx.item(), output

# ================= 主流程 =================
def analyze_sample(path, name, model, grad_cam):
    print(f"\nAnalyzing {name} ({path})...")
    if not os.path.exists(path):
        print("  File not found!")
        return

    # 加载数据
    tensor = torch.load(path).unsqueeze(0).to(DEVICE) # (1, 3, 64, 64)
    
    # 运行 Grad-CAM
    # 这里的 class_idx=None 表示解释"模型预测出的类别"
    heatmap, pred_idx, logits = grad_cam(tensor)
    
    # 获取 Top-3 预测
    probs = torch.softmax(logits, dim=1)
    top3_prob, top3_idx = torch.topk(probs, 3)
    print(f"  Prediction: {CLASS_NAMES[pred_idx]}")
    print(f"  Top-3 Conf: {[(CLASS_NAMES[i], f'{p:.2%}') for i, p in zip(top3_idx[0], top3_prob[0])]}")

    # 准备绘图数据
    img_np = tensor.squeeze().cpu().numpy() # (3, 64, 64)
    
    # 绘图
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Analysis of {name}\nPred: {CLASS_NAMES[pred_idx]}", fontsize=14)
    
    # 1. R 通道 (ATM)
    plt.subplot(1, 5, 1)
    plt.imshow(img_np[0], cmap='inferno', origin='lower')
    plt.title("Ch1: R (ATM)\n(Amp Dynamics)")
    plt.axis('off')
    
    # 2. A 通道 (Mag-Freq)
    plt.subplot(1, 5, 2)
    plt.imshow(img_np[1], cmap='viridis', origin='lower')
    plt.title("Ch2: A (Mag-Freq)\n(Zero-Crossing)")
    plt.axis('off')
    
    # 3. D 通道 (Diff)
    plt.subplot(1, 5, 3)
    plt.imshow(img_np[2], cmap='magma', origin='lower')
    plt.title("Ch3: D (Diff)\n(Phase Jumps)")
    plt.axis('off')
    
    # 4. Grad-CAM 热力图
    plt.subplot(1, 5, 4)
    plt.imshow(heatmap, cmap='jet', origin='lower')
    plt.title("Model Attention\n(Where it looks)")
    plt.axis('off')
    
    # 5. 叠加图 (Overlay)
    # 将热力图叠加在 RGB 合成图上 (R=Ch1, G=Ch2, B=Ch3)
    rgb_img = np.stack([img_np[0], img_np[1], img_np[2]], axis=2)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) # Norm to 0-1
    
    plt.subplot(1, 5, 5)
    plt.imshow(rgb_img, origin='lower')
    plt.imshow(heatmap, cmap='jet', alpha=0.5, origin='lower') # 半透明叠加
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. 加载模型
    model = TinyCNN(len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. 初始化 Grad-CAM
    # 目标层：features 的最后一层 (索引 14: Conv2d(32->64))
    # 注意: TinyCNN 的 features 结构如下:
    # 0: Conv(3->16), 1: BN, 2: ReLU, 3: Pool, 4: Drop
    # 5: Conv(16->32), 6: BN, 7: ReLU, 8: Pool, 9: Drop
    # 10: Conv(32->64), 11: BN, 12: ReLU, 13: Pool
    # 我们通常看最后一个卷积层，即 index 10 (Conv2d) 或 index 12 (ReLU)
    # 这里的 features 是 Sequential，最后一层是 MaxPool2d(idx 13)
    # 我们应该 Hook 最后一个卷积层：features[10]
    target_layer = model.features[10]
    
    grad_cam = GradCAM(model, target_layer)
    
    # 3. 分析
    print("=== Model Explainability Analysis ===")
    
    # 分析样本 A (Correct)
    analyze_sample(SAMPLE_A_PATH, "Sample A (Correct)", model, grad_cam)
    
    # 分析样本 B (Wrong)
    analyze_sample(SAMPLE_B_PATH, "Sample B (Wrong)", model, grad_cam)

if __name__ == "__main__":
    main()