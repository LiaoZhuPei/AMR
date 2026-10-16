import numpy as np
import torch
from test01_model import build_model
import onnxruntime as ort
# 读同一个 CSV 文件
file_path = "E:/C_CNN_ONNX/onnxruntime_AMR_x/data_GMSK5/segment_000099.csv"
with open(file_path, 'r') as f:
    lines = f.readlines()

i_data = np.array([float(x) for x in lines[0].strip().split(',')], dtype=np.float32)
q_data = np.array([float(x) for x in lines[1].strip().split(',')], dtype=np.float32)

print(f"[PY] I[0:5] = {i_data[:5]}")
print(f"[PY] Q[0:5] = {q_data[:5]}")

# 功率归一化
power = np.mean(i_data**2 + q_data**2)
print(f"[PY] power = {power:.10f}")
scale = 1.0 / np.sqrt(power)
i_norm = i_data * scale
q_norm = q_data * scale
print(f"[PY] i_norm[0:5] = {i_norm[:5]}")

# 交错排列
interleaved = np.empty(1024 * 2, dtype=np.float32)
interleaved[0::2] = i_norm
interleaved[1::2] = q_norm
print(f"[PY] interleaved[0:10] = {interleaved[:10]}")

# 推理
tensor = torch.from_numpy(interleaved.reshape(1, 1024, 2))
model = build_model(classes=9)
model.load_state_dict(torch.load('weights/cnn_bilstm_attn_matb1_1.pt', map_location='cpu'))
model.eval()
with torch.no_grad():
    output = model(tensor)
    print(f"[PY] raw output = {output}")

# 构造和 C++ 完全一样的输入
sess = ort.InferenceSession("AMR_CNN_BiLSTM_Attention_cla9.onnx")

# 用上面同一个文件的 interleaved 数据
input_tensor = interleaved.reshape(1, 1024, 2)
result = sess.run(None, {'input': input_tensor})
print(f"[ONNX-PY] output = {result[0]}")
print(f"[ONNX-PY] predicted = {np.argmax(result[0])}")