import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as signal
from collections import Counter

# ================= 1. 配置与模型加载 =================
MODEL_NAME = 'cnn'
MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_sionna_two_A.pt'
DATASET_PATH = './sionna_dataset/Sionna_head_mid1.npz'

# 信号参数
L_ALL = 4800
L_ZC = 768
L_WIN = 1024
NUM_CHUNKS = (L_ALL - L_ZC) // L_WIN  # 4032 // 1024 = 3 段

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 这里假设你的模型定义文件在当前路径
if MODEL_NAME == 'cnn':
    from cnn_frontend import build_model


# ... 其他模型导入逻辑 ...

# ================= 2. 核心处理类 =================

class SignalProcessor:
    def __init__(self, sps=4, head_sym=192, u=25):
        self.sps = sps
        self.head_sym = head_sym
        self.l_zc = head_sym * sps
        self.u = u
        self.local_ref = self._get_local_reference()

    def _get_local_reference(self):
        n = np.arange(self.head_sym)
        zc = np.exp(-1j * np.pi * self.u * n * (n + 1) / self.head_sym)
        zc_up = np.zeros(self.l_zc, dtype=complex)
        zc_up[::self.sps] = zc
        rrc = self._design_rrc(101, self.sps, 0.35)
        ref = np.convolve(zc_up, rrc, mode='same')
        return ref / np.linalg.norm(ref)

    def _design_rrc(self, ntaps, sps, beta):
        T = float(sps)
        n = np.arange(-(ntaps // 2), ntaps // 2 + 1, dtype=np.float64)
        h = np.zeros_like(n)
        for i, t in enumerate(n):
            if abs(t) < 1e-12:
                h[i] = 1.0 - beta + 4.0 * beta / np.pi
            elif abs(abs(t) - T / (4 * beta)) < 1e-12:
                h[i] = beta / np.sqrt(2) * (
                            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            else:
                num = (np.sin(np.pi * t / T * (1 - beta)) + 4 * beta * t / T * np.cos(np.pi * t / T * (1 + beta)))
                den = np.pi * t / T * (1 - (4 * beta * t / T) ** 2)
                h[i] = num / den
        return (h / np.linalg.norm(h))

    def detect_and_remove_zc(self, rx_signal_iq):
        """检测并剔除 ZC 段"""
        complex_sig = rx_signal_iq[0, :] + 1j * rx_signal_iq[1, :]
        num_seg = 16
        seg_len = self.l_zc // num_seg
        out_len = L_ALL - self.l_zc + 1
        corr_sum = np.zeros(out_len)

        for i in range(num_seg):
            chunk = self.local_ref[i * seg_len: (i + 1) * seg_len]
            c = signal.correlate(complex_sig, chunk, mode='valid')
            corr_sum += np.abs(c[i * seg_len: i * seg_len + out_len])

        p = np.argmax(corr_sum)
        # 拼接剩余净荷
        clean_payload = np.delete(rx_signal_iq, np.arange(p, p + self.l_zc), axis=1)
        return clean_payload, p


def power_normalize(data):
    pwr = np.mean(data[0] ** 2 + data[1] ** 2)
    return (data / np.sqrt(pwr + 1e-10)).astype(np.float32)


# ================= 3. 主程序 =================

def main():
    # 1. 加载模型
    classes = ['OQPSK', 'Pi4DQPSK']
    model = build_model(classes=len(classes))
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"找不到权重文件: {MODEL_WEIGHTS_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device).eval()

    # 2. 加载数据集的测试部分 (X_test)
    if not os.path.exists(DATASET_PATH):
        print(f"找不到数据集: {DATASET_PATH}")
        return
    data = np.load(DATASET_PATH)

    # 核心：只使用测试集部分，确保结果公平
    X_test = data['X_test']  # 形状 (N, 2, 4800)
    Y_test = data['Y_test']  # 形状 (N,)
    snr_test = data['snr_test']  # 形状 (N,)
    cls_names = data['classes']  # 类别名称映射

    processor = SignalProcessor()

    # 设置测试样本数量（不超过测试集总量）
    total_available = len(X_test)
    num_to_test = min(200, total_available)

    print(f"\n模型: {MODEL_NAME} | 测试样本量: {num_to_test} (来自 Test Split)")
    print("-" * 125)
    # 增加了 SNR 列
    print(f"{'Idx':<5} | {'真实类别':<10} | {'SNR':<6} | {'ZC位置':<6} | {'切片预测结果 (1/2/3)':<35} | {'最终投票'}")
    print("-" * 125)

    correct = 0

    for i in range(num_to_test):
        raw_sig = X_test[i]
        true_label = cls_names[Y_test[i]]
        current_snr = snr_test[i]  # 提取当前样本的信噪比

        # Step 1: 同步检测与剔除
        payload, p_pos = processor.detect_and_remove_zc(raw_sig)

        # Step 2: 净荷切片 (1024 * 3)
        chunk_preds = []
        for j in range(NUM_CHUNKS):
            start = j * L_WIN
            chunk = payload[:, start: start + L_WIN]

            chunk = power_normalize(chunk)
            inp = torch.from_numpy(chunk).permute(1, 0).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(inp)
                pred_idx = torch.argmax(out, dim=1).item()
                chunk_preds.append(classes[pred_idx])

        # Step 3: 投票逻辑
        votes = Counter(chunk_preds)
        cA, cB = votes.get(classes[0], 0), votes.get(classes[1], 0)

        if cA > cB:
            final_res = classes[0]
        elif cB > cA:
            final_res = classes[1]
        else:
            final_res = "无法预测"

        # 统计准确率
        if final_res == true_label:
            correct += 1
            mark = "✅"
        else:
            mark = "❌"

        # 格式化打印
        chunk_detail = " | ".join(chunk_preds)
        print(
            f"[{i + 1:<3}] | {true_label:<10} | {current_snr:>4.1f}dB | {p_pos:<6} | {chunk_detail:<35} | {final_res:<10} {mark}")

    # 4. 总结
    print("-" * 125)
    print(f"测试完成!")
    print(f"最终平均准确率: {correct / num_to_test * 100:.2f}% ({correct}/{num_to_test})")


if __name__ == "__main__":
    main()