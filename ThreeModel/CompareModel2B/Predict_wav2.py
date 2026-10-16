"""
predict_wav.py  — AMR 实采信号预测
=====================================
核心原则：使 RSPduo 采样率 FS_RX = SYM_RATE × 4，
           从而完全消除重采样，让输入特征与训练数据完全一致。

问题根因说明：
  训练数据 = 仿真符号 → 上采样 → RRC成形(SPS=4) → 存储
  若实测 SPS≠4，则 resample_poly 会额外引入一个低通滤波器，
  其截止频率和相位响应与训练时的 RRC 完全不同，导致特征偏移。
  唯一可靠的方法是让采集时 SPS 直接等于 4。

推荐采集参数：
  符号速率     推荐 RSPduo FS_RX    SPS
  100 kSym/s   400 kHz              4  (最优)
  500 kSym/s   2 MHz                4  (最优，RSPduo标准支持)
  1   MSym/s   4 MHz                4  (最优)
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
    raise ImportError("pip install soundfile")

# ╔══════════════════════════════════════════════════════════════╗
# ║  修改这里切换模型                                              ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME = 'tcn'   # 'transformer' / 'tcn' / 'mamba'

if MODEL_NAME == 'transformer':
    from model_transformer import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn import build_model
elif MODEL_NAME == 'mamba':
    from model_mamba import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# 配置（必须满足 FS_RX = SYM_RATE × 4）
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_SionnaA_6.pt'

WAV_FILE   = 'E:/SUN/Other/64QAM_EE_SDRuno_20260421_055301Z_99980kHz.wav'
FS_RX      = 2000_000    # RSPduo 采样率，必须 = SYM_RATE × 4
SYM_RATE   = 500_000    # 符号速率 (Hz)
SPS        = FS_RX // SYM_RATE   # 自动计算，必须 = 4

SPS_TARGET       = 4
N_SYMS_PER_FRAME = 1024
FRAME_LEN        = SPS_TARGET * N_SYMS_PER_FRAME   # = 4096

NUM_FRAMES   = 200
TEMPERATURE  = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 参数验证
# ─────────────────────────────────────────────────────────────────────────────
def validate_params():
    if SPS != SPS_TARGET:
        print(f"\n{'='*60}")
        print(f"  [ERROR] SPS 不匹配！")
        print(f"  当前 SPS = FS_RX({FS_RX}) / SYM_RATE({SYM_RATE}) = {SPS}")
        print(f"  训练 SPS = {SPS_TARGET}")
        print(f"")
        print(f"  解决方案：将 RSPduo 采样率设置为 {SYM_RATE * SPS_TARGET} Hz")
        print(f"  即 SDRuno 中 Sample Rate = {SYM_RATE * SPS_TARGET / 1000:.0f} kHz")
        print(f"")
        print(f"  常用符号速率对应的最优 FS_RX：")
        rspdup_ok = {200_000, 500_000, 1_000_000, 2_000_000,
                     4_000_000, 6_000_000, 8_000_000, 10_000_000}
        for sr in [25_000, 50_000, 100_000, 200_000, 250_000, 500_000, 1_000_000]:
            needed = sr * 4
            ok = "✓ RSPduo支持" if needed in rspdup_ok else "✗ RSPduo不支持，换符号速率"
            print(f"    {sr/1000:>6.0f} kSym/s  →  FS_RX={needed/1000:.0f} kHz  {ok}")
        print(f"{'='*60}")
        raise SystemExit(1)
    print(f"  [OK] SPS={SPS} = SPS_TARGET，无需重采样 ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_iq(path: str):
    data, fs = sf.read(path, always_2d=True, dtype='float32')
    if data.shape[1] < 2:
        raise ValueError("WAV 为单声道，需要立体声（L=I, R=Q）")
    iq = data[:, 0] + 1j * data[:, 1]
    print(f"  WAV: {os.path.basename(path)}")
    print(f"  fs={fs} Hz  ({len(iq)/fs:.2f} s  /  {len(iq)} 样点)")
    if int(fs) != FS_RX:
        print(f"  ⚠ WAV实际fs={fs}，配置FS_RX={FS_RX}，请确认！")
    return iq, int(fs)


def preprocess_frame(iq_frame: np.ndarray) -> np.ndarray:
    """
    单帧预处理，与训练流水线保持一致。
    输入:  (N,) complex，N = FRAME_LEN = 4096
    输出:  (2, N) float32
    """
    # 步骤1：去除直流偏置（ZIF架构推荐）
    iq_frame = iq_frame - np.mean(iq_frame)

    # 步骤2（可选）：IQ不平衡补偿
    # I, Q = iq_frame.real, iq_frame.imag
    # alpha = np.std(I) / (np.std(Q) + 1e-10)
    # phi   = np.arcsin(np.clip(np.mean(I * Q) / (np.std(I) * np.std(Q) + 1e-10), -1, 1))
    # Q     = (Q - np.sin(phi) * I) / (np.cos(phi) * alpha + 1e-10)
    # iq_frame = I + 1j * Q

    # 步骤3：功率归一化（与训练一致，必须）
    I, Q  = iq_frame.real, iq_frame.imag
    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        return np.stack([I, Q], axis=0).astype(np.float32)
    s = 1.0 / np.sqrt(power)
    return np.stack([I * s, Q * s], axis=0).astype(np.float32)


def segment_frames(iq: np.ndarray, frame_len: int, num_frames: int):
    guard  = frame_len
    usable = iq[guard: len(iq) - guard]
    max_n  = len(usable) // frame_len
    n      = min(num_frames, max_n)
    if n < num_frames:
        need_sec = (num_frames + 2) * frame_len / FS_RX
        print(f"  ⚠ 只能切出 {max_n} 帧（请求 {num_frames}），"
              f"需录制至少 {need_sec:.1f} 秒")
    return [usable[k * frame_len: (k+1) * frame_len] for k in range(n)]


def predict_frame(model, sig_2xN: np.ndarray, device, temperature=1.0):
    model.eval()
    x = torch.from_numpy(sig_2xN).float()
    x = x.permute(1, 0).unsqueeze(0).to(device)   # (1, 4096, 2)
    with torch.no_grad():
        out = model(x)
    probs = F.softmax(out / temperature, dim=1).squeeze(0).cpu().numpy()
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, default=WAV_FILE)
    parser.add_argument('--n',   type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  AMR Predictor  |  Model: {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"  FS_RX    = {FS_RX} Hz")
    print(f"  SYM_RATE = {SYM_RATE} Hz")
    print(f"  SPS      = {SPS}  (需 = {SPS_TARGET})")

    validate_params()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device   = {device}\n")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[Error] 权重不存在: {MODEL_WEIGHTS_PATH}")
        return

    model = build_model(classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    print(f"  [{MODEL_NAME}] 参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    iq_raw, _ = load_wav_iq(args.wav)

    frames = segment_frames(iq_raw, FRAME_LEN, args.n)
    if not frames:
        print("[Error] 帧数为0，请检查信号长度和参数")
        return
    print(f"  切出帧数 : {len(frames)}  (每帧 {FRAME_LEN} 样点 = {N_SYMS_PER_FRAME} 符号)\n")

    counts    = Counter()
    probs_acc = np.zeros(len(CLASSES))

    print(f"  {'#':>4}  {'预测':12}  {'置信度':>7}  {'Top-3'}")
    print(f"  {'-'*58}")

    for k, frame in enumerate(frames):
        sig = preprocess_frame(frame)
        cls, conf, probs = predict_frame(model, sig, device, TEMPERATURE)
        counts[cls]  += 1
        probs_acc    += probs
        top3 = np.argsort(probs)[::-1][:3]
        t3   = '  '.join(f"{CLASSES[i]}:{probs[i]*100:.0f}%" for i in top3)
        print(f"  {k+1:>4}  {cls:<12}  {conf*100:>6.1f}%  {t3}")

    total = sum(counts.values())
    avg   = probs_acc / total

    print(f"\n{'='*60}")
    print(f"  [{MODEL_NAME}] 汇总  ({total} 帧)")
    print(f"{'='*60}")
    for cls in CLASSES:
        c = counts.get(cls, 0)
        p = c / total * 100 if total else 0
        print(f"  {cls:<12}: {c:>4} 帧  ({p:5.1f}%)  {'█' * int(p/2)}")

    if counts:
        top = counts.most_common(1)[0]
        print(f"\n  ▶ 硬投票: {top[0]}  ({top[1]/total*100:.1f}%)")
    vi = int(np.argmax(avg))
    print(f"  ▶ 软投票: {CLASSES[vi]}  (avg_prob={avg[vi]*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()