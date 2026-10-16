"""
统一预测脚本 v2（WAV版）：适配双分支模型
输入：RSPduo IQ-OUT 立体声 WAV（L=I, R=Q）
核心要求：FS_RX = SYM_RATE × 4，使 SPS=4 与训练一致，无需重采样

修改自 .mat 版本，新增：
  - WAV 文件读取（soundfile）
  - SPS 验证（不等于4直接报错并给出指引）
  - 帧切分（每帧 SIGNAL_LENGTH × SPS = 4096 样点）
  - 前 N 帧预测（命令行 --n 指定，默认200）
  - 软/硬 双投票汇总
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
    raise ImportError("请安装: pip install soundfile")

# ╔══════════════════════════════════════════════════════════════╗
# ║  修改这里切换模型和配置                                        ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME    = 'tcn'   # 'transformer' / 'tcn' / 'bilstm'
SIGNAL_LENGTH = 1024       # 与训练时 SIGNAL_LENGTH 一致（符号数）

if MODEL_NAME == 'transformer':
    from model_transformer_v2 import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn_v2 import build_model
elif MODEL_NAME == 'bilstm':
    from test01_v2_model import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# 采集参数：必须满足 FS_RX = SYM_RATE × 4
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_v2_Sionna32_2.pt'

WAV_FILE   = 'E:/SUN/Right/256QAM_01_SDRuno_20260421_081701Z_99980kHz.wav'   # 待预测 WAV 文件路径
FS_RX      = 2000_000         # RSPduo 采样率（必须 = SYM_RATE × 4）
SYM_RATE   = 500_000         # 符号速率（与 VSG25A 设置一致）
SPS        = FS_RX // SYM_RATE          # 必须 = 4
SPS_TARGET = 4
FRAME_LEN  = SPS_TARGET * SIGNAL_LENGTH # 模型输入样点数 = 4 × 1024 = 4096

NUM_FRAMES  = 200    # 默认预测帧数
TEMPERATURE = 1.0    # softmax 温度


# ─────────────────────────────────────────────────────────────────────────────
# 参数验证：SPS 不等于4时立即报错
# ─────────────────────────────────────────────────────────────────────────────
def validate_sps():
    if SPS != SPS_TARGET:
        print(f"\n{'='*62}")
        print(f"  [ERROR] SPS 不匹配！预测将失败。")
        print(f"  当前 SPS = {FS_RX}/{SYM_RATE} = {SPS}，训练 SPS = {SPS_TARGET}")
        print(f"  解决：将 RSPduo 采样率改为 {SYM_RATE * SPS_TARGET} Hz")
        print(f"\n  各符号速率最优 FS_RX（使 SPS=4）：")
        supported = {200_000, 500_000, 1_000_000, 2_000_000,
                     4_000_000, 6_000_000, 8_000_000, 10_000_000}
        for sr in [25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
            fs_need = sr * 4
            ok = "RSPduo 支持" if fs_need in supported else "RSPduo 不支持，换符号速率"
            print(f"    {sr//1000:>4} kSym/s  ->  FS_RX={fs_need//1000} kHz  [{ok}]")
        print(f"\n  推荐方案：SYM_RATE=500k, FS_RX=2MHz → SPS=4 且 RSPduo原生支持")
        print(f"{'='*62}")
        raise SystemExit(1)
    print(f"  [OK] SPS={SPS} = SPS_TARGET，无需重采样 ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_iq(path: str):
    """读取 RSPduo IQ-OUT WAV，返回复数 IQ 数组和实际采样率"""
    data, fs = sf.read(path, always_2d=True, dtype='float32')
    if data.shape[1] < 2:
        raise ValueError("WAV 必须为立体声（L=I, R=Q）")
    iq = data[:, 0] + 1j * data[:, 1]
    print(f"  WAV : {os.path.basename(path)}")
    print(f"  fs  : {fs} Hz  ({len(iq)/fs:.2f} s  /  {len(iq)} 样点)")
    if int(fs) != FS_RX:
        print(f"  ⚠ WAV 实际 fs={fs}，配置 FS_RX={FS_RX}，请核对！")
    return iq, int(fs)


def preprocess_frame(iq_frame: np.ndarray) -> np.ndarray:
    """
    单帧预处理，与训练流水线完全一致。
    输入:  (N,) complex，N = FRAME_LEN = 4096
    输出:  (2, N) float32

    步骤：去直流 → 功率归一化
    可选：IQ 不平衡补偿（见注释）
    """
    # 去直流偏置（ZIF 架构推荐，仿真数据无直流，实测有）
    # iq_frame = iq_frame - np.mean(iq_frame)

    # 可选：IQ 不平衡补偿
    # I, Q  = iq_frame.real, iq_frame.imag
    # alpha = np.std(I) / (np.std(Q) + 1e-10)
    # phi   = np.arcsin(np.clip(np.mean(I*Q)/(np.std(I)*np.std(Q)+1e-10), -1, 1))
    # Q     = (Q - np.sin(phi)*I) / (np.cos(phi)*alpha + 1e-10)
    # iq_frame = I + 1j*Q

    # 功率归一化（与训练一致，必须）
    I, Q  = iq_frame.real, iq_frame.imag
    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        return np.stack([I, Q], axis=0).astype(np.float32)
    s = 1.0 / np.sqrt(power)
    return np.stack([I * s, Q * s], axis=0).astype(np.float32)


def segment_frames(iq: np.ndarray, frame_len: int, num_frames: int):
    """切分 IQ 为等长帧，丢弃两端各 1 帧避免边缘效应"""
    guard  = frame_len
    usable = iq[guard: len(iq) - guard]
    max_n  = len(usable) // frame_len
    n      = min(num_frames, max_n)
    if n < num_frames:
        need = (num_frames + 2) * frame_len / FS_RX
        print(f"  ⚠ 只能切出 {max_n} 帧，需录制至少 {need:.1f} s")
    return [usable[k * frame_len: (k+1) * frame_len] for k in range(n)]


def predict_frame(model, sig_2xN: np.ndarray, device, temperature=1.0):
    """
    sig_2xN: (2, FRAME_LEN) float32
    模型输入: (1, SIGNAL_LENGTH, 2)  —— v2 双分支模型期望格式
    """
    model.eval()
    # (2, 4096) → (4096, 2) → (1, 4096, 2)
    x = torch.from_numpy(sig_2xN).float()
    x = x.permute(1, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    probs = F.softmax(out / temperature, dim=1).squeeze(0).cpu().numpy()
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AMR WAV Predictor v2 (dual-branch)')
    parser.add_argument('--wav', type=str,  default=WAV_FILE,   help='WAV 文件路径')
    parser.add_argument('--n',   type=int,  default=NUM_FRAMES, help='预测帧数（默认200）')
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  AMR Predictor v2  |  双分支  |  Model: {MODEL_NAME}")
    print(f"{'='*62}")
    print(f"  FS_RX       = {FS_RX} Hz")
    print(f"  SYM_RATE    = {SYM_RATE} Hz")
    print(f"  SPS         = {SPS}  （需 = {SPS_TARGET}）")
    print(f"  SIGNAL_LEN  = {SIGNAL_LENGTH} 符号")
    print(f"  FRAME_LEN   = {FRAME_LEN} 样点")

    validate_sps()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device      = {device}\n")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[Error] 权重文件不存在: {MODEL_WEIGHTS_PATH}")
        return

    model = build_model(classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  [{MODEL_NAME}_v2]  参数量: {n_p:,}\n")

    iq_raw, _ = load_wav_iq(args.wav)

    frames = segment_frames(iq_raw, FRAME_LEN, args.n)
    if not frames:
        print("[Error] 帧数为 0，请检查 WAV 长度和参数")
        return
    print(f"  切出帧数 = {len(frames)}  (每帧 {FRAME_LEN} 样点 = {SIGNAL_LENGTH} 符号)\n")

    counts    = Counter()
    probs_acc = np.zeros(len(CLASSES))

    print(f"  {'帧':>4}  {'预测类别':12}  {'置信度':>7}  Top-3")
    print(f"  {'-'*60}")

    for k, frame in enumerate(frames):
        sig  = preprocess_frame(frame)
        cls, conf, probs = predict_frame(model, sig, device, TEMPERATURE)
        counts[cls]  += 1
        probs_acc    += probs
        top3 = np.argsort(probs)[::-1][:3]
        t3   = '  '.join(f"{CLASSES[i]}:{probs[i]*100:.0f}%" for i in top3)
        print(f"  {k+1:>4}  {cls:<12}  {conf*100:>6.1f}%  {t3}")

    total = sum(counts.values())
    avg   = probs_acc / total

    print(f"\n{'='*62}")
    print(f"  [{MODEL_NAME}_v2]  汇总  ({total} 帧)")
    print(f"{'='*62}")
    for cls in CLASSES:
        c = counts.get(cls, 0)
        p = c / total * 100 if total else 0
        print(f"  {cls:<12}: {c:>4} 帧  ({p:5.1f}%)  {'|' * int(p/2)}")

    if counts:
        top = counts.most_common(1)[0]
        print(f"\n  >> 硬投票: {top[0]}  ({top[1]/total*100:.1f}%)")
    vi = int(np.argmax(avg))
    print(f"  >> 软投票: {CLASSES[vi]}  (平均后验 {avg[vi]*100:.1f}%)")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
