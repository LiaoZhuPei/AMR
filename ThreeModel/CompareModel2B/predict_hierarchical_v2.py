"""
predict_hierarchical_v2.py
两阶段层次识别预测脚本

预测流程:
  1. Stage 1 (粗分类): 判断是否属于 QAM 族
  2. Stage 2 (细分类): 若属于 QAM，进一步输出具体阶数

输出示例:
  [ 1] sample_001.mat: [Stage1: QAM(94.2%)] → [Stage2: 64QAM(88.1%)]  最终: 64QAM
  [ 2] sample_002.mat: [Stage1: NonQAM(97.5%)] → BPSK? (仅参考)        最终: NonQAM
"""

import os
import glob
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import h5py

from model_bilstm_v2_hierarchical import (
    build_model, ALL_CLASSES, QAM_CLASSES
)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_NAME        = 'bilstm_hierarchical'
SIGNAL_LENGTH     = 1024
MODEL_WEIGHTS_PATH= f'weights/{MODEL_NAME}_SionnaA_6_II.pt'
TEST_DATA_FOLDER  = 'E:/Project/AMR-Benchmark/output_mat_files_16QAM_DDC'
NUM_SAMPLES       = 100
FILE_EXTENSION    = '.mat'
TEMPERATURE       = 1.0   # softmax 温度（>1 更平滑，<1 更尖锐）

# 粗分类标签
COARSE_LABELS = ['NonQAM', 'QAM']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device     : {device}")
print(f"Model            : {MODEL_NAME}")
print(f"Signal length    : {SIGNAL_LENGTH}")
print(f"QAM classes      : {QAM_CLASSES}")


# ─────────────────────────────────────────────
# 预处理（与训练完全一致）
# ─────────────────────────────────────────────
def power_normalize(signal_data):
    I, Q  = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ─────────────────────────────────────────────
# 单样本两阶段预测
# ─────────────────────────────────────────────
def predict_hierarchical(model, signal_data, device, temperature=1.0):
    """
    Args:
        signal_data: np.ndarray, shape (1, 2, SIGNAL_LENGTH)

    Returns:
        dict with keys:
          stage1_label   : 'QAM' or 'NonQAM'
          stage1_conf    : float  (0-1)
          stage1_probs   : np.ndarray (2,)

          is_qam         : bool

          stage2_label   : str  (QAM 阶数名称, 例如 '64QAM')
                           若 stage1 判断非 QAM，仍会给出细分类得分供参考
          stage2_conf    : float
          stage2_probs   : np.ndarray (5,)  各 QAM 阶数概率
          stage2_ranking : list of (name, prob)  按概率降序

          final_label    : 综合两阶段的最终判断
    """
    model.eval()
    sample = power_normalize(signal_data.squeeze(0))           # (2, L)
    tensor = torch.from_numpy(sample).float().permute(1, 0)   # (L, 2)
    tensor = tensor.unsqueeze(0).to(device)                    # (1, L, 2)

    with torch.no_grad():
        logits_c, logits_f, _ = model(tensor)

    # Stage 1
    probs_c      = F.softmax(logits_c / temperature, dim=1).squeeze(0).cpu().numpy()
    stage1_idx   = int(probs_c.argmax())
    stage1_label = COARSE_LABELS[stage1_idx]
    stage1_conf  = float(probs_c[stage1_idx])
    is_qam       = (stage1_idx == 1)

    # Stage 2
    probs_f       = F.softmax(logits_f / temperature, dim=1).squeeze(0).cpu().numpy()
    stage2_idx    = int(probs_f.argmax())
    stage2_label  = QAM_CLASSES[stage2_idx]
    stage2_conf   = float(probs_f[stage2_idx])
    stage2_ranking= sorted(
        zip(QAM_CLASSES, probs_f.tolist()),
        key=lambda x: x[1], reverse=True
    )

    # 最终结论
    if is_qam:
        final_label = stage2_label          # QAM 族 → 具体阶数
    else:
        final_label = stage1_label          # 直接报告 NonQAM

    return {
        'stage1_label'   : stage1_label,
        'stage1_conf'    : stage1_conf,
        'stage1_probs'   : probs_c,
        'is_qam'         : is_qam,
        'stage2_label'   : stage2_label,
        'stage2_conf'    : stage2_conf,
        'stage2_probs'   : probs_f,
        'stage2_ranking' : stage2_ranking,
        'final_label'    : final_label,
    }


# ─────────────────────────────────────────────
# 批量预测（提升吞吐量）
# ─────────────────────────────────────────────
def predict_batch(model, signals, device, temperature=1.0):
    """
    Args:
        signals: list of np.ndarray, each shape (1, 2, SIGNAL_LENGTH)
    Returns:
        list of result dicts (same structure as predict_hierarchical)
    """
    model.eval()
    tensors = []
    for s in signals:
        sample = power_normalize(s.squeeze(0))
        t = torch.from_numpy(sample).float().permute(1, 0).unsqueeze(0)
        tensors.append(t)
    batch = torch.cat(tensors, dim=0).to(device)   # (B, L, 2)

    with torch.no_grad():
        logits_c, logits_f, _ = model(batch)

    probs_c = F.softmax(logits_c / temperature, dim=1).cpu().numpy()
    probs_f = F.softmax(logits_f / temperature, dim=1).cpu().numpy()

    results = []
    for i in range(len(signals)):
        s1_idx   = int(probs_c[i].argmax())
        s2_idx   = int(probs_f[i].argmax())
        is_qam   = (s1_idx == 1)
        results.append({
            'stage1_label'   : COARSE_LABELS[s1_idx],
            'stage1_conf'    : float(probs_c[i][s1_idx]),
            'stage1_probs'   : probs_c[i],
            'is_qam'         : is_qam,
            'stage2_label'   : QAM_CLASSES[s2_idx],
            'stage2_conf'    : float(probs_f[i][s2_idx]),
            'stage2_probs'   : probs_f[i],
            'stage2_ranking' : sorted(
                zip(QAM_CLASSES, probs_f[i].tolist()),
                key=lambda x: x[1], reverse=True
            ),
            'final_label'    : QAM_CLASSES[s2_idx] if is_qam else COARSE_LABELS[s1_idx],
        })
    return results


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights not found at '{MODEL_WEIGHTS_PATH}'")
        exit()

    # 加载模型
    model = build_model()
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # 找到测试文件
    files = glob.glob(os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}'))
    if not files:
        print(f"No '{FILE_EXTENSION}' files found in {TEST_DATA_FOLDER}")
        exit()
    files_to_test = files[:NUM_SAMPLES]
    print(f"Testing {len(files_to_test)} files...\n")

    # ─── 逐样本预测并打印详细结果 ─────────────
    final_counts  = Counter()
    coarse_counts = Counter()

    print(f"{'Idx':>5}  {'File':<35}  {'Stage1':<20}  {'Stage2 (QAM ranking)'}")
    print("-" * 110)

    for i, fpath in enumerate(files_to_test):
        fname = os.path.basename(fpath)
        try:
            with h5py.File(fpath, 'r') as f:
                signal = f['data_save'][:]

            if signal.shape != (1, 2, SIGNAL_LENGTH):
                print(f"  [Skip] {fname}: shape {signal.shape}")
                continue

            res = predict_hierarchical(model, signal, device, TEMPERATURE)

            # 打印行
            s1_str = f"{res['stage1_label']}({res['stage1_conf']*100:.1f}%)"
            if res['is_qam']:
                # 打印 QAM 内部 top-3 排名
                top3 = res['stage2_ranking'][:3]
                s2_str = "  ".join(f"{n}:{p*100:.1f}%" for n, p in top3)
                final_str = f"→ \033[1m{res['stage2_label']}\033[0m({res['stage2_conf']*100:.1f}%)"
            else:
                # 非 QAM：给出细分类得分供参考（灰色提示）
                top1 = res['stage2_ranking'][0]
                s2_str = f"(ref) {top1[0]}:{top1[1]*100:.1f}%"
                final_str = f"→ NonQAM"

            print(f"  [{i+1:3d}]  {fname:<35}  Stage1:{s1_str:<20}  {s2_str}  {final_str}")

            final_counts[res['final_label']]  += 1
            coarse_counts[res['stage1_label']]+=1

        except Exception as e:
            print(f"  [Err] {fname}: {e}")

    # ─── 汇总统计 ──────────────────────────────
    total = sum(final_counts.values())
    print(f"\n{'='*60}")
    print(f"  PREDICTION SUMMARY  ({total} samples)")
    print(f"{'='*60}")

    print("\n  Stage 1 (Coarse):")
    for label in COARSE_LABELS:
        cnt = coarse_counts.get(label, 0)
        print(f"    {label:<10}: {cnt:>4}  ({cnt/total*100:.1f}%)")

    print("\n  Final Label Distribution:")
    # 先打 NonQAM
    cnt = final_counts.get('NonQAM', 0)
    print(f"    {'NonQAM':<12}: {cnt:>4}  ({cnt/total*100:.1f}%)")
    # 再按 QAM 顺序
    for name in QAM_CLASSES:
        cnt = final_counts.get(name, 0)
        print(f"    {name:<12}: {cnt:>4}  ({cnt/total*100:.1f}%)")

    if final_counts:
        top = final_counts.most_common(1)[0]
        print(f"\n  Most predicted: {top[0]} ({top[1]} times, {top[1]/total*100:.1f}%)")
    print(f"{'='*60}")
