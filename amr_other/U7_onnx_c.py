import torch
import torch.nn as nn
import os

# 定义模型结构 (必须与训练时一致)
class TinyCNN(nn.Module):
    def __init__(self, num_classes=12):
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
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# 路径
RESULTS_DIR = "./training_results_tensor"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
ONNX_PATH = os.path.join(RESULTS_DIR, "best_model_compatible.onnx") # 新名字
IMAGE_SIZE = 64

def export_compatible():
    device = torch.device("cpu")
    model = TinyCNN(12).to(device)
    # 加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print(f"正在导出兼容版模型: {ONNX_PATH}")
    # ================= 关键修改 =================
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,           # <--- 强制使用 Opset 11 或 12，兼容老版本 ORT
        do_constant_folding=True,
        input_names=['input'],      # 固定输入名
        output_names=['output'],    # 固定输出名
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("导出完成！请将此模型复制到 VS 工程目录。")

if __name__ == "__main__":
    export_compatible()