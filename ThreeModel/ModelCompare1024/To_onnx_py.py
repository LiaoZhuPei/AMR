import torch
import torch.nn as nn

# ===== 改动 1：导入新模型 =====
from model_tcn import build_model  # 替换原来的 import LSTM as mcl

# ===== 改动 2：类别列表与推理代码一致 =====
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4DQPSK',
                '16QAM', '2ASK', '16APSK', '32APSK',
                '64QAM', '32QAM']

# ===== 改动 3：权重路径与推理代码一致 =====
weights_path = 'weights/tcn_sionna1024_1.pt'

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

# ===== 改动 5：虚拟输入 shape 与模型输入一致 (batch, 1024, 2) =====
dummy_input = torch.randn(1, 1024, 2).to(device)

# 验证前向传播正常
with torch.no_grad():
    test_output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print("Forward pass OK ✓")

# ===== 改动 6：导出 ONNX，更新文件名 =====
# onnx_filename = "AMR_CNN_BiLSTM_Attention_cla9.onnx"  # eights/cnn_bilstm_attn_matb1_1.pt
onnx_filename = "AMR_tcn_Attention_cla11_nobatch.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    verbose=True,
    opset_version=13,
    do_constant_folding=True,           # 改为 True，优化常量折叠
    input_names=['input'],              # 添加输入名称
    output_names=['output'],            # 添加输出名称

)
"""
dynamic_axes = {  # 支持动态 batch size
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}
"""
print(f"\nONNX model exported to: {onnx_filename}")