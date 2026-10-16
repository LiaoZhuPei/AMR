import os
import numpy as np
import torch
import onnxruntime as ort

# ================= 配置区域 =================
# 你的 .iq 文件路径
TARGET_FILE = "E:/Project/AMR-Benchmark/QAM64-03-24-25-10h51m51s805.iq"
# 你的量化模型路径
MODEL_PATH = "./training_results_tensor/best_model_int8.onnx"

# 必须和训练时完全一致的参数
SAMPLES_PER_IMAGE = 10240
IMAGE_SIZE = 64
CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK",
    "16QAM", "32QAM", "64QAM",
    "2ASK", "16APSK", "32APSK", "FSK2"
]


# ================= 1. 移植预处理逻辑 (从生成器复制而来) =================
# 注意：推理通常在 CPU 上进行，所以这里去掉 .to(device) 或默认为 cpu

def fast_hist2d(x, y, bins=64, range_bounds=((-1, 1), (-1, 1))):
    x_min, x_max = range_bounds[0]
    y_min, y_max = range_bounds[1]
    x = torch.clamp(x, x_min, x_max)
    y = torch.clamp(y, y_min, y_max)
    x_idx = ((x - x_min) / (x_max - x_min) * (bins - 1)).long()
    y_idx = ((y - y_min) / (y_max - y_min) * (bins - 1)).long()
    flat_idx = y_idx * bins + x_idx
    hist = torch.bincount(flat_idx, minlength=bins * bins)
    return hist.view(bins, bins).float()


def preprocess_iq_to_tensor(iq_complex_chunk, img_size=64):
    """
    输入: torch.complex64 (10240,)
    输出: numpy float32 (1, 3, 64, 64) -> 准备好给ONNX吃了
    """
    if iq_complex_chunk.numel() < 100:
        return None

    # 归一化
    iq_data = iq_complex_chunk
    scale = torch.sqrt(torch.mean(iq_data.real ** 2 + iq_data.imag ** 2)) + 1e-9
    norm_iq = iq_data / (3.0 * scale)

    # 提取特征
    real = norm_iq.real
    imag = norm_iq.imag
    phase = torch.angle(norm_iq) / np.pi
    mag = torch.abs(norm_iq)

    diff_signal = norm_iq[1:] * torch.conj(norm_iq[:-1])
    d_scale = torch.sqrt(torch.mean(diff_signal.real ** 2 + diff_signal.imag ** 2)) + 1e-9
    diff_signal = diff_signal / (3.0 * d_scale)
    d_real = diff_signal.real
    d_imag = diff_signal.imag

    # 生成直方图
    bins = img_size
    hist_r = fast_hist2d(real, imag, bins=bins, range_bounds=((-1.2, 1.2), (-1.2, 1.2)))
    hist_a = fast_hist2d(phase, mag, bins=bins, range_bounds=((-1.0, 1.0), (0.0, 1.5)))
    hist_d = fast_hist2d(d_real, d_imag, bins=bins, range_bounds=((-1.2, 1.2), (-1.2, 1.2)))

    # 数值处理
    def process(h):
        h = torch.log1p(h)
        mx = h.max()
        if mx > 1e-6: h = h / mx
        return h

    # 堆叠成 (3, 64, 64)
    tensor = torch.stack([process(hist_r), process(hist_a), process(hist_d)], dim=0)
    tensor = torch.flip(tensor, [1])  # 翻转Y轴

    # 转为 Numpy 并且增加 Batch 维度 (1, 3, 64, 64)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


# ================= 2. 读取二进制 IQ 文件 =================

def read_iq_file(file_path):
    print(f"Reading file: {file_path}")
    # 假设你的 IQ 文件是 int16 格式 (Real, Imag, Real, Imag...)
    # 如果是 float32 格式，请将 dtype 改为 np.float32
    try:
        raw = np.fromfile(file_path, dtype=np.int16)
        if len(raw) % 2 != 0: raw = raw[:-1]  # 确保偶数
        # 归一化到 -1~1 之间 (int16最大值是32767)
        iq_complex = (raw[::2] + 1j * raw[1::2]).astype(np.complex64) / 32767.0
        return iq_complex
    except Exception as e:
        print(f"读取失败: {e}")
        return None


# ================= 3. 主流程 =================

def main():
    # A. 加载模型
    if not os.path.exists(MODEL_PATH):
        print("找不到模型文件，请检查路径")
        return

    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # B. 读取原始文件
    iq_data = read_iq_file(TARGET_FILE)
    if iq_data is None: return

    # C. 切片与推理
    # 一个大的 IQ 文件可能包含几十万个点，我们需要每 10240 个点切一段进行预测
    num_slices = len(iq_data) // SAMPLES_PER_IMAGE
    print(f"文件包含 {len(iq_data)} 个采样点，将切分为 {num_slices} 个片段进行预测...")

    predictions = []

    # 只需要 torch CPU
    device = torch.device("cpu")

    for i in range(num_slices):
        # 1. 切片
        chunk_np = iq_data[i * SAMPLES_PER_IMAGE: (i + 1) * SAMPLES_PER_IMAGE]
        chunk_torch = torch.from_numpy(chunk_np).to(device)

        # 2. 预处理 (RAW -> Tensor)
        input_tensor = preprocess_iq_to_tensor(chunk_torch, IMAGE_SIZE)

        if input_tensor is None: continue

        # 3. 推理 (Tensor -> Result)
        outputs = sess.run([output_name], {input_name: input_tensor})
        pred_idx = np.argmax(outputs[0], axis=1)[0]

        predictions.append(pred_idx)

        # 打印前几个片段的结果看看
        if i < 5:
            print(f"Slice {i}: Predicted {CLASS_NAMES[pred_idx]}")

    # D. 投票统计最终结果
    if len(predictions) > 0:
        counts = np.bincount(predictions, minlength=len(CLASS_NAMES))
        final_class_idx = np.argmax(counts)
        final_class_name = CLASS_NAMES[final_class_idx]
        confidence = counts[final_class_idx] / len(predictions)

        print("\n" + "=" * 30)
        print(f"最终判定结果: {final_class_name}")
        print(f"置信度: {confidence:.2%}")
        print("=" * 30)

        # 打印详细投票情况
        print("详细投票:")
        for idx, count in enumerate(counts):
            if count > 0:
                print(f"  {CLASS_NAMES[idx]}: {count} 票")
    else:
        print("数据不足，无法预测")


if __name__ == "__main__":
    main()