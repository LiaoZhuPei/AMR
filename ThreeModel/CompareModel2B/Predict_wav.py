"""
统一预测脚本：从 RSPduo IQ WAV 文件进行 AMR 预测
输入：立体声 WAV（L=I, R=Q），RSPduo IQ-OUT 模式录制
模型期望：(batch=1, 1024, 2)，SPS=4（与训练一致）

修改自 .mat 版本，新增：
  - WAV 文件读取（soundfile）
  - 按接收 SPS 重采样至训练 SPS=4
  - 帧切分（每帧1024符号 × SPS_TARGET = 4096样点）
  - 前 N 帧预测（默认 200）
  - IQ归一化（功率归一化，与训练一致）
"""

import os
import glob
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
MODEL_NAME = 'tcn'   # 选项: 'transformer', 'tcn', 'mamba'

if MODEL_NAME == 'transformer':
    from model_transformer import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn import build_model
elif MODEL_NAME == 'mamba':
    from model_mamba import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_SionnaA_6.pt'

# ── WAV 采集参数（按实际 RSPduo 设置填写）──────────────────────────────────────
WAV_FILE        = 'E:/SUN/Right/256QAM_07_SDRuno_20260422_073501Z_99970kHz.wav'  # 待预测 WAV 文件
FS_RX           = 2_000_000   # RSPduo 采样率 (Hz)，例如 2 MHz
SYM_RATE        = 500_000     # 符号速率 (Hz)，例如 100 kSym/s
SPS_RX          = int(FS_RX / SYM_RATE)  # 接收端每符号样点数，此例 = 20

# ── 训练参数（与训练数据生成一致，勿改）──────────────────────────────────────
SPS_TARGET      = 4           # 训练集仿真时使用的过采样率
N_SYMS_PER_FRAME = 1024       # 每帧符号数（与训练一致）
FRAME_LEN_RX    = SPS_RX * N_SYMS_PER_FRAME      # 接收域每帧样点数（此例 20480）
FRAME_LEN_TGT   = SPS_TARGET * N_SYMS_PER_FRAME  # 目标域每帧样点数（此例 4096）

NUM_FRAMES      = 200         # 预测帧数，默认 200
TEMPERATURE     = 1.0         # softmax 温度（调高=更"平滑"，一般保持 1.0）

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_iq(path: str):
    """
    读取 RSPduo IQ-OUT WAV 文件。
    返回复数 IQ 信号 (N,) 和实际采样率。
    支持 16-bit PCM / 32-bit float / 24-bit PCM。
    """
    data, fs = sf.read(path, always_2d=True, dtype='float32')
    if data.shape[1] < 2:
        raise ValueError("WAV 文件为单声道，RSPduo IQ-OUT 应为立体声（L=I, R=Q）")
    iq = data[:, 0] + 1j * data[:, 1]   # 复数基带信号
    print(f"  WAV loaded: {os.path.basename(path)}")
    print(f"  采样率    : {fs} Hz   (期望 {FS_RX} Hz)")
    print(f"  总样点数  : {len(iq)}   ({len(iq)/fs:.3f} s)")
    if int(fs) != FS_RX:
        print(f"  ⚠ 采样率不匹配！请将 FS_RX 改为 {int(fs)}")
    return iq, int(fs)


def resample_iq(iq: np.ndarray, fs_in: int, fs_out_target_sps: int,
                sym_rate: int) -> np.ndarray:
    """
    将接收域 IQ（SPS=fs_in/sym_rate）重采样至目标 SPS（SPS_TARGET）。
    使用多相滤波重采样（polyphase），保证抗混叠。
    """
    fs_out = sym_rate * fs_out_target_sps           # 目标采样率
    if fs_in == fs_out:
        return iq
    g = gcd(int(fs_out), int(fs_in))
    up   = int(fs_out) // g
    down = int(fs_in)  // g
    # 分别对实部和虚部重采样（resample_poly 不支持复数）
    i_rs = resample_poly(iq.real.astype(np.float64), up, down)
    q_rs = resample_poly(iq.imag.astype(np.float64), up, down)
    print(f"  重采样: {fs_in} Hz → {fs_out} Hz  (up={up}, down={down})")
    return (i_rs + 1j * q_rs).astype(np.complex64)


def segment_frames(iq: np.ndarray, frame_len: int, num_frames: int):
    """
    将长 IQ 序列切分为等长帧。
    返回 list of (frame_len,) complex arrays。
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
    功率归一化（与训练预处理一致）。
    输入: (N,) complex
    输出: (2, N) float32，归一化后的 [I; Q]
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
    signal_2xN: np.ndarray shape (2, FRAME_LEN_TGT)  即 (2, 4096)
    模型输入期望: (batch=1, FRAME_LEN_TGT, 2)
    """
    model.eval()
    # (2, 4096) → (4096, 2) → (1, 4096, 2)
    x = torch.from_numpy(signal_2xN).float()   # (2, 4096)
    x = x.permute(1, 0)                        # (4096, 2)
    x = x.unsqueeze(0).to(device)              # (1, 4096, 2)

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
                        help='符号速率 Hz（默认100000）')
    args = parser.parse_args()

    wav_path = args.wav
    n_frames = args.n
    fs_rx    = args.fs
    sym_rate = args.sr
    sps_rx   = fs_rx // sym_rate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  AMR Predictor  |  WAV Input  |  Model: {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  WAV file      : {wav_path}")
    print(f"  RX sample rate: {fs_rx} Hz")
    print(f"  Symbol rate   : {sym_rate} Hz")
    print(f"  SPS (RX)      : {sps_rx}  →  重采样至 SPS={SPS_TARGET}")
    print(f"  Frame length  : {N_SYMS_PER_FRAME} syms × SPS={SPS_TARGET} = {FRAME_LEN_TGT} samples")
    print(f"  Frames to test: {n_frames}")

    # ── 加载模型 ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\n[Error] 权重文件不存在: {MODEL_WEIGHTS_PATH}")
        return

    model = build_model(classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [Model] {MODEL_NAME}  |  参数量: {total_params:,}\n")

    # ── 读取 WAV ─────────────────────────────────────────────────────────────
    if not os.path.isfile(wav_path):
        print(f"[Error] 文件不存在: {wav_path}")
        return

    iq_raw, fs_actual = load_wav_iq(wav_path)

    # ── 重采样至 SPS_TARGET ─────────────────────────────────────────────────
    # 接收端 SPS = fs_rx / sym_rate（此例 20）
    # 训练集 SPS = SPS_TARGET = 4
    # 重采样比例 = SPS_TARGET / SPS_RX = 4/20 = 1/5
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
        # 功率归一化 → (2, FRAME_LEN_TGT)
        sig = power_normalize(frame)

        pred_cls, conf, probs = predict_frame(
            model, sig, device, CLASSES, TEMPERATURE)

        prediction_counts[pred_cls] += 1
        prob_accumulator += probs

        # Top-3
        top3_idx  = np.argsort(probs)[::-1][:3]
        top3_str  = '  '.join(
            f"{CLASSES[i]}:{probs[i]*100:.0f}%" for i in top3_idx)

        print(f"  {k+1:>5}  {pred_cls:<12}  {conf*100:>7.1f}%  {top3_str}")

    # ── 汇总 ────────────────────────────────────────────────────────────────
    total = sum(prediction_counts.values())
    avg_probs = prob_accumulator / total

    print(f"\n{'='*60}")
    print(f"  [{MODEL_NAME}]  预测汇总  |  共 {total} 帧")
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
        print(f"\n  ▶ 最终预测: {top[0]}  ({top[1]} 帧, {pct:.1f}%)")

    # 投票置信度（平均后验概率）
    voted_idx = int(np.argmax(avg_probs))
    print(f"  ▶ 软投票结果: {CLASSES[voted_idx]}  "
          f"(平均后验 {avg_probs[voted_idx]*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()