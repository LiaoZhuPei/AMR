"""
统一预测脚本：从 RSPduo IQ WAV 文件进行 AMR 预测
适用模型：CNN_BiLSTM_Attention (来自 test01_model)
输入：立体声 WAV（L=I, R=Q），RSPduo IQ-OUT 模式录制
模型期望输入：(batch=1, 1024, 2)
"""

import os
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

try:
    import soundfile as sf
except ImportError:
    raise ImportError("请安装 soundfile: pip install soundfile")

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    raise ImportError("请安装 scipy: pip install scipy")

# ===== 导入你的模型 =====
from test01_model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# 1. 基本配置
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

# 模型权重路径
# cnn_bilstm_attn_matb1_1.pt
# cnn_bilstm_attn_sionna_b2_1.pt
MODEL_WEIGHTS_PATH = 'weights/cnn_bilstm_attn_multirate_3.pt'

# ── WAV 采集参数（按实际 RSPduo 录制时的设置填写）─────────────────────────────
WAV_FILE = 'E:/SUN/Right/16QAM_3M_01_SDRuno_20260520_102801Z_99960kHz.wav'  # 待预测的 WAV 文件路径
FS_RX = 3_000_000  # RSPduo 采样率 (Hz)，例如 2 MHz
SYM_RATE = 750_000  # 符号速率 (Hz)，例如 500 kSym/s

# ── 训练与数据参数 ───────────────────────────────────────────────────────────
SPS_TARGET = 4  # 你在生成训练集(mat文件)时使用的过采样率 (Samples Per Symbol)
FRAME_LEN_TGT = 1024  # 你的模型期望的输入长度 (1024)

NUM_FRAMES = 200  # 预测帧数，默认测试前 200 帧
TEMPERATURE = 1.0  # softmax 温度（一般保持 1.0）

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 信号处理与工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_iq(path: str):
    """读取 RSPduo IQ-OUT WAV 文件，返回复数 IQ 信号 (N,) 和实际采样率"""
    data, fs = sf.read(path, always_2d=True, dtype='float32')
    if data.shape[1] < 2:
        raise ValueError("WAV 文件为单声道，RSPduo IQ-OUT 应为立体声（L=I, R=Q）")
    iq = data[:, 0] + 1j * data[:, 1]  # 复数基带信号
    print(f"  WAV loaded: {os.path.basename(path)}")
    print(f"  采样率    : {fs} Hz")
    print(f"  总样点数  : {len(iq)}   ({len(iq) / fs:.3f} s)")
    return iq, int(fs)


def resample_iq(iq: np.ndarray, fs_in: int, fs_out_target_sps: int, sym_rate: int) -> np.ndarray:
    """将接收域 IQ 重采样至训练时的目标 SPS"""
    fs_out = sym_rate * fs_out_target_sps
    if fs_in == fs_out:
        return iq
    g = gcd(int(fs_out), int(fs_in))
    up = int(fs_out) // g
    down = int(fs_in) // g

    i_rs = resample_poly(iq.real.astype(np.float64), up, down)
    q_rs = resample_poly(iq.imag.astype(np.float64), up, down)
    print(f"  重采样    : {fs_in} Hz → {fs_out} Hz  (up={up}, down={down})")
    return (i_rs + 1j * q_rs).astype(np.complex64)


def segment_frames(iq: np.ndarray, frame_len: int, num_frames: int):
    """将连续长信号切分为等长帧 (1024样点)"""
    guard = frame_len  # 丢弃开头的一小段避免初始暂态
    usable = iq[guard: len(iq) - guard]
    max_frames = len(usable) // frame_len
    n = min(num_frames, max_frames)
    if n < num_frames:
        print(f"  ⚠ 信号长度只够切出 {max_frames} 帧（请求 {num_frames} 帧）")
    frames = [usable[k * frame_len: (k + 1) * frame_len] for k in range(n)]
    return frames


def power_normalize(iq_frame: np.ndarray) -> np.ndarray:
    """
    功率归一化（与你的目标代码中完全一致的计算逻辑）
    输入: (N,) complex
    输出: (2, N) float32 [I; Q] -> 用于给你的模型作为基底
    """
    I = iq_frame.real
    Q = iq_frame.imag
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        sig = np.stack([I, Q], axis=0)
        return sig.astype(np.float32)

    scale = 1.0 / np.sqrt(power)
    sig = np.stack([I * scale, Q * scale], axis=0)  # (2, 1024)
    return sig.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 单帧预测函数 (适配目标代码中 (1, 1024, 2) 的需求)
# ─────────────────────────────────────────────────────────────────────────────

def predict_frame(model, signal_2x1024: np.ndarray, device, class_list, temperature=1.0):
    """
    维度变换流程:
        (2, 1024)  →  permute  →  (1024, 2)
                   →  unsqueeze → (1, 1024, 2)  ← 适配 CNN_BiLSTM_Attention
    """
    model.eval()

    # 转换为模型所需张量
    x = torch.from_numpy(signal_2x1024).float()  # (2, 1024)
    x = x.permute(1, 0)  # (1024, 2)
    x = x.unsqueeze(0).to(device)  # (1, 1024, 2)

    with torch.no_grad():
        output = model(x)  # (1, num_classes)

    probs = F.softmax(output / temperature, dim=1)
    conf, idx = torch.max(probs, dim=1)

    return class_list[idx.item()], conf.item(), probs.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 命令行参数配置（兼容终端直接跑或者IDE里跑）
    parser = argparse.ArgumentParser(description='WAV-based AMR prediction')
    parser.add_argument('--wav', type=str, default=WAV_FILE, help='输入 WAV 文件路径')
    parser.add_argument('--n', type=int, default=NUM_FRAMES, help='预测帧数')
    parser.add_argument('--fs', type=int, default=FS_RX, help='采样率 Hz')
    parser.add_argument('--sr', type=int, default=SYM_RATE, help='符号速率 Hz')
    args = parser.parse_args()

    wav_path = args.wav
    n_frames = args.n
    fs_rx = args.fs
    sym_rate = args.sr

    print(f"\n{'=' * 60}")
    print(f"  AMR Predictor | CNN_BiLSTM_Attention | WAV Input")
    print(f"{'=' * 60}")
    print(f"  Device        : {device}")
    print(f"  Target Shape  : (Batch, {FRAME_LEN_TGT}, 2)")

    # ── 1. 加载模型 ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\n[Error] 模型权重文件不存在: {MODEL_WEIGHTS_PATH}")
        return

    print("\nLoading model architecture...")
    model = build_model(classes=len(CLASSES))
    print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params:,}\n")

    # ── 2. 读取及处理 WAV ────────────────────────────────────────────────────
    if not os.path.isfile(wav_path):
        print(f"[Error] WAV 文件不存在: {wav_path}")
        return

    iq_raw, fs_actual = load_wav_iq(wav_path)

    # 重采样
    iq_rs = resample_iq(iq_raw, fs_actual, SPS_TARGET, sym_rate)

    # 切分帧
    frames = segment_frames(iq_rs, FRAME_LEN_TGT, n_frames)
    print(f"  实际切出帧数  : {len(frames)}\n")

    if not frames:
        print("[Error] 帧数为 0，信号太短或参数设置有误")
        return

    # ── 3. 逐帧预测 ──────────────────────────────────────────────────────────
    prediction_counts = Counter()
    prob_accumulator = np.zeros(len(CLASSES), dtype=np.float64)

    print(f"  {'帧':>5}  {'预测类别':<12}  {'置信度':>8}  {'Top-3'}")
    print(f"  {'-' * 55}")

    for i, frame in enumerate(frames):
        # frame 此时为 (1024,) 的复数数组
        # power_normalize 将其归一化并拆分为 (2, 1024) 的实数组
        normalized_signal_2x1024 = power_normalize(frame)

        # 传入预测函数，内部会将其变换为模型需要的 (1, 1024, 2)
        pred_cls, conf, probs = predict_frame(
            model, normalized_signal_2x1024, device, CLASSES, TEMPERATURE)

        prediction_counts[pred_cls] += 1
        prob_accumulator += probs

        # 打印信息 (Top-3 类别)
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_str = '  '.join(f"{CLASSES[i]}:{probs[i] * 100:.0f}%" for i in top3_idx)
        print(f"  {i + 1:>5}  {pred_cls:<12}  {conf * 100:>7.1f}%  {top3_str}")

    # ── 4. 统计摘要 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("           PREDICTION SUMMARY")
    print("=" * 60)

    total_processed = sum(prediction_counts.values())
    avg_probs = prob_accumulator / total_processed

    for cls in CLASSES:
        cnt = prediction_counts.get(cls, 0)
        pct = (cnt / total_processed * 100) if total_processed > 0 else 0
        avg_p = avg_probs[CLASSES.index(cls)] * 100
        bar = '█' * int(pct / 2)
        print(f"  {cls:<12}: {cnt:>4} 帧  ({pct:5.1f}%)  avg_prob={avg_p:4.1f}%  {bar}")

    if prediction_counts:
        top = prediction_counts.most_common(1)[0]
        pct = top[1] / total_processed * 100
        print(f"\n  ▶ 硬投票结果 (最常预测): {top[0]} ({top[1]} 帧, {pct:.1f}%)")

    # 软投票结果 (根据整个片段平均概率计算)
    voted_idx = int(np.argmax(avg_probs))
    print(f"  ▶ 软投票结果 (平均后验): {CLASSES[voted_idx]} ({avg_probs[voted_idx] * 100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()