"""
统一预测脚本：从 RSPduo IQ WAV 文件进行 AMR 预测
融合了目标代码的模型/类别配置与参考代码的 WAV 处理流程
输入：立体声 WAV（L=I, R=Q），RSPduo IQ-OUT 模式录制
模型期望：(batch=1, 1024, 2)
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
    raise ImportError("请安装 soundfile:  pip install soundfile")

try:
    from scipy.signal import resample_poly
    from math import gcd
except ImportError:
    raise ImportError("请安装 scipy:  pip install scipy")

# ╔══════════════════════════════════════════════════════════════╗
# ║  修改这里切换模型                                              ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME = 'tcn'  # 选项: 'transformer', 'tcn', 'mamba'

if MODEL_NAME == 'transformer':
    from model_transformer import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn import build_model
elif MODEL_NAME == 'mamba':
    from model_mamba import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# 全局配置 (基于目标代码)
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

# _sionna_b2_2
MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_multirate_2_dynamic_2.pt'

# ── WAV 采集参数（按实际 RSPduo 设置填写）──────────────────────────────────────

WAV_FILE        = 'E:/SUN/Right/16QAM_3M_01_SDRuno_20260520_102801Z_99960kHz.wav'  # 待预测 WAV 文件
FS_RX           = 3_000_000   # RSPduo 采样率 (Hz)，例如 2 MHz
SYM_RATE        = 750_000     # 符号速率 (Hz)，例如 500 kSym/s
SPS_RX          = int(FS_RX / SYM_RATE)  # 接收端每符号样点数

# ── 训练参数（与训练数据生成一致）──────────────────────────────────────
SPS_TARGET      = 4           # 训练集仿真时使用的过采样率 (假设为4)
FRAME_LEN_TGT   = 1024        # 目标域每帧样点数（目标代码中为 1024）

NUM_FRAMES      = 200         # 预测帧数，默认 200
TEMPERATURE     = 1.0         # softmax 温度（调高=更"平滑"，一般保持 1.0）

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数 (基于参考代码)
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_iq(path: str):
    """
    读取 RSPduo IQ-OUT WAV 文件。
    返回复数 IQ 信号 (N,) 和实际采样率。
    """
    data, fs = sf.read(path, always_2d=True, dtype='float32')
    if data.shape[1] < 2:
        raise ValueError("WAV 文件为单声道，RSPduo IQ-OUT 应为立体声（L=I, R=Q）")
    iq = data[:, 0] + 1j * data[:, 1]   # 复数基带信号
    print(f"  WAV loaded: {os.path.basename(path)}")
    print(f"  采样率    : {fs} Hz   (期望 {FS_RX} Hz)")
    print(f"  总样点数  : {len(iq)}   ({len(iq)/fs:.3f} s)")
    if int(fs) != FS_RX:
        print(f"  ⚠ 采样率不匹配！请将命令行参数 --fs 改为 {int(fs)}")
    return iq, int(fs)


def resample_iq(iq: np.ndarray, fs_in: int, fs_out_target_sps: int,
                sym_rate: int) -> np.ndarray:
    """
    将接收域 IQ 重采样至目标 SPS（SPS_TARGET）。
    """
    fs_out = sym_rate * fs_out_target_sps           # 目标采样率
    if fs_in == fs_out:
        return iq
    g = gcd(int(fs_out), int(fs_in))
    up   = int(fs_out) // g
    down = int(fs_in)  // g
    i_rs = resample_poly(iq.real.astype(np.float64), up, down)
    q_rs = resample_poly(iq.imag.astype(np.float64), up, down)
    print(f"  重采样: {fs_in} Hz → {fs_out} Hz  (up={up}, down={down})")
    return (i_rs + 1j * q_rs).astype(np.complex64)


def segment_frames(iq: np.ndarray, frame_len: int, num_frames: int):
    """
    将长 IQ 序列切分为等长帧。
    去掉两端各 frame_len 样点（避免边缘效应）。
    """
    guard = frame_len                        # 两端各丢弃 1 帧
    usable = iq[guard: len(iq) - guard]
    max_frames = len(usable) // frame_len
    n = min(num_frames, max_frames)
    if n < num_frames:
        print(f"  ⚠ 信号只够切出 {max_frames} 帧（请求 {num_frames} 帧）")
    frames = [usable[k * frame_len: (k + 1) * frame_len] for k in range(n)]
    return frames


def power_normalize(iq_frame: np.ndarray) -> np.ndarray:
    """
    功率归一化（统一接口：接收复数1D，输出 2xN 浮点张量）。
    """
    I = iq_frame.real
    Q = iq_frame.imag
    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        sig = np.stack([I, Q], axis=0)
        return sig.astype(np.float32)
    scale = 1.0 / np.sqrt(power)
    sig = np.stack([I * scale, Q * scale], axis=0)   # (2, N)
    return sig.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 单帧预测
# ─────────────────────────────────────────────────────────────────────────────

def predict_frame(model, signal_2xN: np.ndarray,
                  device, class_list, temperature=1.0):
    """
    signal_2xN: np.ndarray shape (2, 1024)
    模型输入期望: (batch=1, 1024, 2)
    """
    model.eval()
    x = torch.from_numpy(signal_2xN).float()   # (2, 1024)
    x = x.permute(1, 0)                        # (1024, 2)
    x = x.unsqueeze(0).to(device)              # (1, 1024, 2)

    with torch.no_grad():
        output = model(x)

    probs = F.softmax(output / temperature, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return class_list[idx.item()], conf.item(), probs.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WAV-based AMR prediction')
    parser.add_argument('--wav',   type=str, default=WAV_FILE,
                        help='输入 WAV 文件路径')
    parser.add_argument('--n',     type=int, default=NUM_FRAMES,
                        help='预测帧数（默认200）')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                        help='模型名称：transformer / tcn / mamba')
    parser.add_argument('--fs',    type=int, default=FS_RX,
                        help='WAV 采样率 Hz（默认2000000）')
    parser.add_argument('--sr',    type=int, default=SYM_RATE,
                        help='符号速率 Hz（默认500000）')
    args = parser.parse_args()

    wav_path = args.wav
    n_frames = args.n
    fs_rx    = args.fs
    sym_rate = args.sr
    sps_rx   = fs_rx // sym_rate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  AMR Predictor  |  WAV Input  |  Model: {args.model}")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  WAV file      : {wav_path}")
    print(f"  RX sample rate: {fs_rx} Hz")
    print(f"  Symbol rate   : {sym_rate} Hz")
    print(f"  SPS (RX)      : {sps_rx}  →  重采样至 SPS={SPS_TARGET}")
    print(f"  Frame length  : {FRAME_LEN_TGT} samples/frame")
    print(f"  Frames to test: {n_frames}")

    # ── 加载模型 ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\n[Error] 权重文件不存在: {MODEL_WEIGHTS_PATH}")
        return

    model = build_model(classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [Model] {args.model}  |  参数量: {total_params:,}\n")

    # ── 读取 WAV ─────────────────────────────────────────────────────────────
    if not os.path.isfile(wav_path):
        print(f"[Error] 文件不存在: {wav_path}")
        return

    iq_raw, fs_actual = load_wav_iq(wav_path)

    # ── 重采样至 SPS_TARGET ─────────────────────────────────────────────────
    iq_rs = resample_iq(iq_raw, fs_actual, SPS_TARGET, sym_rate)
    print(f"  重采样后样点数: {len(iq_rs)}")

    # ── 帧切分 ──────────────────────────────────────────────────────────────
    frames = segment_frames(iq_rs, FRAME_LEN_TGT, n_frames)
    print(f"  实际切出帧数  : {len(frames)}\n")

    if not frames:
        print("[Error] 帧数为 0，信号太短或参数设置有误")
        return

    # ── 逐帧预测 ────────────────────────────────────────────────────────────
    prediction_counts = Counter()
    prob_accumulator  = np.zeros(len(CLASSES), dtype=np.float64)

    print(f"  {'帧':>5}  {'预测类别':<12}  {'置信度':>8}  {'Top-3'}")
    print(f"  {'-'*55}")

    for k, frame in enumerate(frames):
        # 功率归一化 -> 输出 (2, 1024)
        sig = power_normalize(frame)

        pred_cls, conf, probs = predict_frame(
            model, sig, device, CLASSES, TEMPERATURE)

        prediction_counts[pred_cls] += 1
        prob_accumulator += probs

        # Top-3 统计
        top3_idx  = np.argsort(probs)[::-1][:3]
        top3_str  = '  '.join(
            f"{CLASSES[i]}:{probs[i]*100:.0f}%" for i in top3_idx)

        print(f"  {k+1:>5}  {pred_cls:<12}  {conf*100:>7.1f}%  {top3_str}")

    # ── 汇总 ────────────────────────────────────────────────────────────────
    total = sum(prediction_counts.values())
    avg_probs = prob_accumulator / total

    print(f"\n{'='*60}")
    print(f"  [{args.model}]  预测汇总  |  共 {total} 帧")
    print(f"{'='*60}")
    for cls in CLASSES:
        cnt = prediction_counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        avg_p = avg_probs[CLASSES.index(cls)] * 100
        bar   = '█' * int(pct / 2)
        print(f"  {cls:<12}: {cnt:>4} 帧  ({pct:5.1f}%)  avg_prob={avg_p:4.1f}%  {bar}")

    if prediction_counts:
        top = prediction_counts.most_common(1)[0]
        pct = top[1] / total * 100
        print(f"\n  ▶ 最终硬投票预测: {top[0]}  ({top[1]} 帧, {pct:.1f}%)")

    # 软投票（基于平均概率）
    voted_idx = int(np.argmax(avg_probs))
    print(f"  ▶ 最终软投票预测: {CLASSES[voted_idx]}  "
          f"(平均后验 {avg_probs[voted_idx]*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()