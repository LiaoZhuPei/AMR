import torch
import torch.nn as nn

# ===== 改动 1：导入 v2 双分支模型，与推理脚本一致 =====
from model_tcn_v2 import build_model

# ===== 改动 2：类别列表与推理脚本一致 =====
classes = ['16QAM', '32QAM', '64QAM', '128QAM', '256QAM']

# ===== 改动 3：权重路径与推理脚本一致 =====
weights_path = 'weights/tcn_v2_SionnaAll_qam.pt'

device = "cpu"

# ===== 改动 4：用 build_model 构建模型 =====
model = build_model(classes=len(classes))

# 加载模型权重
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model = model.to(device)

# 模型推理模式
model.eval()
model.cpu()

# 打印参数量（可选，方便确认模型正确）
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ===== 改动 5：虚拟输入 shape 与推理脚本一致 (batch, SIGNAL_LENGTH, 2) =====
SIGNAL_LENGTH = 1024
dummy_input = torch.randn(1, SIGNAL_LENGTH, 2).to(device)

# 验证前向传播正常
with torch.no_grad():
    test_output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print("Forward pass OK ✓")

# ===== 改动 6：导出 ONNX，更新文件名 =====
onnx_filename = "AMR_tcn_v2_SionnaAll_qam.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    verbose=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)
"""
dynamic_axes = {  # 支持动态 batch size
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}
"""
print(f"\nONNX model exported to: {onnx_filename}")