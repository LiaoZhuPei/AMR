import os
import time
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================

DATASET_ROOT = "./dataset_tri_RAD_Tensor"
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")
RESULTS_DIR = "./training_results_tensor"

# 定义要对比的两个模型
MODEL_FP32_PATH = os.path.join(RESULTS_DIR, "best_model2.onnx")  # 原始模型
MODEL_INT8_PATH = os.path.join(RESULTS_DIR, "best_model_int8.onnx")  # 量化模型

CM_SAVE_PATH = os.path.join(RESULTS_DIR, "compare_confusion_matrix.png")

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK",
    "16QAM", "32QAM", "64QAM",
    "2ASK", "16APSK", "32APSK", "FSK2"
]


# ================= 辅助函数 =================

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def load_session(path):
    if not os.path.exists(path):
        print(f"Warning: Model not found at {path}")
        return None, None, None
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


# ================= 主推理逻辑 =================

def main():
    print("===== 模型对比测试 (FP32 vs INT8) =====")

    # 1. 对比文件大小
    size_fp32 = get_file_size_mb(MODEL_FP32_PATH)
    size_int8 = get_file_size_mb(MODEL_INT8_PATH)

    print(f"\n[1] 模型体积对比:")
    print(f"    FP32 原模型: {size_fp32:.2f} MB")
    print(f"    INT8 量化版: {size_int8:.2f} MB")
    print(f"    -> 压缩率:   {size_fp32 / size_int8:.2f}倍")

    # 2. 加载模型
    print(f"\n[2] 加载推理引擎...")
    sess_fp32, in_fp32, out_fp32 = load_session(MODEL_FP32_PATH)
    sess_int8, in_int8, out_int8 = load_session(MODEL_INT8_PATH)

    if sess_fp32 is None or sess_int8 is None:
        print("错误: 缺少模型文件，无法进行对比。")
        return

    # 3. 准备数据
    test_files = []
    print(f"\n[3] 扫描测试数据...")
    for label_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(REAL_TEST_DIR) / cls_name
        if cls_dir.exists():
            files = list(cls_dir.glob("*.pt"))
            test_files.extend([(p, label_idx) for p in files])

    if not test_files:
        print("未找到测试数据。")
        return

    print(f"    找到 {len(test_files)} 个样本。开始双模型推理...")

    # 4. 循环推理与统计
    # 统计变量
    latencies_fp32 = []
    latencies_int8 = []
    mse_errors = []  # 记录 logits 的均方误差

    y_true = []
    y_pred_fp32 = []
    y_pred_int8 = []

    # 预热 (Warmup) - 防止第一次推理过慢影响统计
    dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)
    for _ in range(10):
        sess_fp32.run([out_fp32], {in_fp32: dummy})
        sess_int8.run([out_int8], {in_int8: dummy})

    # 正式测试
    for file_path, label in tqdm(test_files, unit="sample"):
        try:
            # 准备输入
            tensor_data = torch.load(file_path)
            numpy_input = to_numpy(tensor_data).astype(np.float32)
            numpy_input = np.expand_dims(numpy_input, axis=0)  # (1, 3, 64, 64)

            # --- 运行 FP32 模型 ---
            t0 = time.perf_counter()  # 使用高精度计时器
            res_fp32 = sess_fp32.run([out_fp32], {in_fp32: numpy_input})[0]
            t_fp32 = (time.perf_counter() - t0) * 1000  # 转换为毫秒

            # --- 运行 INT8 模型 ---
            t0 = time.perf_counter()
            res_int8 = sess_int8.run([out_int8], {in_int8: numpy_input})[0]
            t_int8 = (time.perf_counter() - t0) * 1000  # 转换为毫秒

            # 记录数据
            latencies_fp32.append(t_fp32)
            latencies_int8.append(t_int8)

            # 计算两个模型输出的具体数值差异 (Mean Squared Error)
            diff = np.mean((res_fp32 - res_int8) ** 2)
            mse_errors.append(diff)

            # 记录预测结果
            pred_fp32 = np.argmax(res_fp32, axis=1)[0]
            pred_int8 = np.argmax(res_int8, axis=1)[0]

            y_true.append(label)
            y_pred_fp32.append(pred_fp32)
            y_pred_int8.append(pred_int8)

        except Exception as e:
            print(f"Error: {e}")

    # 5. 结果汇总输出
    acc_fp32 = accuracy_score(y_true, y_pred_fp32)
    acc_int8 = accuracy_score(y_true, y_pred_int8)

    avg_lat_fp32 = np.mean(latencies_fp32)
    avg_lat_int8 = np.mean(latencies_int8)
    avg_mse = np.mean(mse_errors)

    print(f"\n{'=' * 20} 最终对比报告 {'=' * 20}")
    print(f"{'指标':<15} | {'FP32 (原版)':<15} | {'INT8 (量化)':<15} | {'变化情况'}")
    print(f"{'-' * 60}")
    print(
        f"{'准确率 (Acc)':<15} | {acc_fp32:.2%}          | {acc_int8:.2%}          | {'基本持平' if abs(acc_fp32 - acc_int8) < 0.01 else '有波动'}")
    print(
        f"{'平均耗时/样本':<15} | {avg_lat_fp32:.3f} ms       | {avg_lat_int8:.3f} ms       | 速度提升 {(avg_lat_fp32 / avg_lat_int8):.2f}x")
    print(
        f"{'模型大小':<15} | {size_fp32:.2f} MB         | {size_int8:.2f} MB         | 体积缩小 {(size_fp32 / size_int8):.2f}x")
    print(f"{'-' * 60}")
    print(f"数值精度损失 (MSE): {avg_mse:.6f} (越小越好)")

    # 额外检查：有没有样本是 FP32 对了但 INT8 错了的？
    y_true = np.array(y_true)
    y_pred_fp32 = np.array(y_pred_fp32)
    y_pred_int8 = np.array(y_pred_int8)

    disagreements = np.sum(y_pred_fp32 != y_pred_int8)
    print(f"\n其中有 {disagreements} 个样本两个模型给出了不同的预测结果。")


if __name__ == "__main__":
    main()