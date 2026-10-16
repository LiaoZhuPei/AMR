import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as signal
from collections import Counter

# ================= 1. 配置与模型加载 =================
MODEL_NAME = 'cnn'  # 切换: 'transformer', 'tcn', 'mamba'
MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_sionna_two_A.pt'
DATASET_PATH = './sionna_dataset/Sionna_head_mid1.npz'

# 信号参数
L_ALL = 4800
L_ZC = 768
L_WIN = 1024
NUM_CHUNKS = (L_ALL - L_ZC) // L_WIN  # 4032 // 1024 = 3 段

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if MODEL_NAME == 'cnn':
    from cnn_frontend import build_model


# ... 其他模型导入保持一致

# ================= 2. 核心算法工具类 =================

class SignalProcessor:
    def __init__(self, sps=4, head_sym=192, u=25):
        self.sps = sps
        self.head_sym = head_sym
        self.l_zc = head_sym * sps
        self.u = u
        self.local_ref = self._get_local_reference()

    def _get_local_reference(self):
        # 内部逻辑保持不变，用于生成本地ZC模板
        n = np.arange(self.head_sym)
        zc = np.exp(-1j * np.pi * self.u * n * (n + 1) / self.head_sym)
        zc_up = np.zeros(self.l_zc, dtype=complex)
        zc_up[::self.sps] = zc
        # 简化版RRC脉冲成形（与生成保持一致）
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
        """
        1. 检测位置 2. 剔除并拼接
        rx_signal_iq: (2, 4800)
        """
        complex_sig = rx_signal_iq[0, :] + 1j * rx_signal_iq[1, :]

        # 使用分段抗频偏相关
        num_seg = 16
        seg_len = self.l_zc // num_seg
        out_len = L_ALL - self.l_zc + 1
        corr_sum = np.zeros(out_len)

        for i in range(num_seg):
            chunk = self.local_ref[i * seg_len: (i + 1) * seg_len]
            c = signal.correlate(complex_sig, chunk, mode='valid')
            corr_sum += np.abs(c[i * seg_len: i * seg_len + out_len])

        # 找到最强峰位置 p
        p = np.argmax(corr_sum)

        # 剔除 ZC 段并拼接剩余部分
        # 剩余部分 = [0:p] + [p+768:]
        clean_payload = np.delete(rx_signal_iq, np.arange(p, p + self.l_zc), axis=1)

        return clean_payload, p


def power_normalize(data):
    pwr = np.mean(data[0] ** 2 + data[1] ** 2)
    return (data / np.sqrt(pwr + 1e-10)).astype(np.float32)


# ================= 3. 主预测流程 =================

def main():
    # A. 加载模型
    classes = ['OQPSK', 'Pi4DQPSK']
    model = build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device).eval()

    # B. 加载数据
    data = np.load(DATASET_PATH)
    X_test, Y_test, snr_test = data['X_test'], data['Y_test'], data['snr_test']
    cls_names = data['classes']

    processor = SignalProcessor()

    print(f"开始测试... 目标切片数: {NUM_CHUNKS} 段/信号 (1024x3)")
    print("-" * 100)
    print(f"{'Idx':<5} | {'真实':<10} | {'ZC位置':<6} | {'各段预测结果':<35} | {'最终投票'}")
    print("-" * 100)

    correct = 0
    total = 1000  # 测试前200个

    for i in range(total):
        raw_sig = X_test[i]  # (2, 4800)
        true_label = cls_names[Y_test[i]]

        # 1. 检测 ZC 并剔除
        payload, p_pos = processor.detect_and_remove_zc(raw_sig)  # payload shape: (2, 4032)

        # 2. 切分成 1024 长度的块
        chunk_preds = []
        for j in range(NUM_CHUNKS):
            start = j * L_WIN
            chunk = payload[:, start: start + L_WIN]

            # 归一化并推理
            chunk = power_normalize(chunk)
            inp = torch.from_numpy(chunk).permute(1, 0).unsqueeze(0).to(device)  # (1, 1024, 2)

            with torch.no_grad():
                out = model(inp)
                pred_idx = torch.argmax(out, dim=1).item()
                chunk_preds.append(classes[pred_idx])

        # 3. 最终投票
        votes = Counter(chunk_preds)
        cA, cB = votes.get(classes[0], 0), votes.get(classes[1], 0)

        if cA > cB:
            final_res = classes[0]
        elif cB > cA:
            final_res = classes[1]
        else:
            final_res = "无法预测"

        if final_res == true_label: correct += 1

        # 4. 打印详情
        chunk_str = " | ".join(chunk_preds)
        mark = "✅" if final_res == true_label else "❌"
        print(f"[{i:<3}] | {true_label:<10} | {p_pos:<6} | {chunk_str:<35} | {final_res:<10} {mark}")

    print("-" * 100)
    print(f"测试完成! 准确率: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()