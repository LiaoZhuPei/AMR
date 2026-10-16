"""
step3_evaluate.py
对比预训练模型 vs 微调后模型，在实测 .mat 文件上的表现
同时支持在仿真数据集（.npz）上做消融对比
"""

import os
import glob
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import h5py

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ★ 修改这里                                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
MODEL_WEIGHTS = {
    'pretrained': 'weights/tcn_v2_SionnaA_6.pt',
    'finetuned':  'weights/tcn_v2_finetune_ewc.pt',
}

SIGNAL_LENGTH    = 1024
TEST_MAT_FOLDER  = 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC'
NUM_SAMPLES      = 100       # 每个模型测多少个 .mat 文件
FILE_EXTENSION   = '.mat'

# 也可以同时在仿真npz上验证（留空则跳过）
EVAL_NPZ         = './finetune_v2_dataset/finetune_v2.npz'   # 留空字符串则跳过

CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM',
           '128QAM', '256QAM', 'MSK']

# ════════════════════════════════════════════════════════════════════════
from model_tcn_v2 import build_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# ════════════════════════════════════════════════════════════════════════
#  预处理
# ════════════════════════════════════════════════════════════════════════

def power_normalize(signal_data):
    """功率归一化，与训练时一致"""
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
#  单文件推理
# ════════════════════════════════════════════════════════════════════════

def predict(model, signal_2xL):
    """
    signal_2xL: np.ndarray (2, SIGNAL_LENGTH)
    返回 (class_name, confidence)
    """
    inp = torch.from_numpy(signal_2xL).float()    # (2, L)
    inp = inp.permute(1, 0).unsqueeze(0).to(device)  # (1, L, 2)
    model.eval()
    with torch.no_grad():
        out = model(inp)
    prob, idx = torch.max(F.softmax(out, dim=1), 1)
    return CLASSES[idx.item()], prob.item()


# ════════════════════════════════════════════════════════════════════════
#  在 .mat 文件上评估
# ════════════════════════════════════════════════════════════════════════

def eval_on_mat(model, model_name, folder, n_samples):
    files = glob.glob(os.path.join(folder, f'*{FILE_EXTENSION}'))
    if not files:
        print(f"  未找到 {FILE_EXTENSION} 文件: {folder}")
        return {}

    files = files[:n_samples]
    print(f"\n[{model_name}] 测试 {len(files)} 个 .mat 文件")
    counts = Counter()
    errors = 0

    for i, fpath in enumerate(files):
        try:
            with h5py.File(fpath, 'r') as f:
                signal = f['data_save'][:]

            if signal.shape != (1, 2, SIGNAL_LENGTH):
                print(f"  [Skip] shape={signal.shape}")
                continue

            s = power_normalize(signal.squeeze(0))   # (2, L)
            cls, conf = predict(model, s)
            counts[cls] += 1

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{len(files)}  当前最多: "
                      f"{counts.most_common(1)[0] if counts else ('None',0)}")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [Error] {os.path.basename(fpath)}: {e}")

    # 打印汇总
    total = sum(counts.values())
    print(f"\n  {model_name} 预测分布（共 {total} 个样本）：")
    print(f"  {'类别':<14} {'次数':>6} {'占比':>8}")
    print(f"  {'─'*30}")
    for cls in CLASSES:
        cnt = counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 5)
        print(f"  {cls:<14} {cnt:>6}  {pct:>6.1f}%  {bar}")
    if counts:
        top = counts.most_common(1)[0]
        print(f"\n  最多预测类别: {top[0]}  ({top[1]}/{total}, {top[1]/total*100:.1f}%)")

    return dict(counts)


# ════════════════════════════════════════════════════════════════════════
#  在仿真 npz 上评估（按SNR段）
# ════════════════════════════════════════════════════════════════════════

def eval_on_npz(model, model_name, npz_path, target_class='128QAM'):
    if not npz_path or not os.path.exists(npz_path):
        return

    print(f"\n[{model_name}] 仿真数据评估: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_test   = data['X_test']
    Y_test   = data['Y_test']
    snr_test = data['snr_test']
    cls_names = list(data['classes'])

    # 整体准确率
    from torch.utils.data import DataLoader, TensorDataset
    Xt = torch.from_numpy(X_test).permute(0, 2, 1).float()
    Yt = torch.from_numpy(Y_test)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=512, shuffle=False)

    model.eval()
    correct_per = [0] * len(CLASSES)
    total_per   = [0] * len(CLASSES)
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(1)
            for c in range(len(CLASSES)):
                mask = (Y == c)
                correct_per[c] += (preds[mask] == c).sum().item()
                total_per[c]   += mask.sum().item()

    print(f"\n  每类准确率：")
    print(f"  {'类别':<14} {'准确率':>8}")
    print(f"  {'─'*25}")
    for c, cls in enumerate(CLASSES):
        acc  = correct_per[c] / max(total_per[c], 1)
        flag = ' ←' if cls == target_class else ''
        print(f"  {cls:<14} {acc*100:>7.1f}%{flag}")

    overall = sum(correct_per) / max(sum(total_per), 1)
    print(f"\n  整体准确率: {overall*100:.2f}%")

    # 目标类别按 SNR 段统计
    tgt_idx = CLASSES.index(target_class) if target_class in CLASSES else -1
    if tgt_idx >= 0:
        print(f"\n  {target_class} 按 SNR 段准确率：")
        snr_bins = [(-5, 5), (5, 15), (15, 25), (25, 40)]
        for lo, hi in snr_bins:
            mask = (Y_test == tgt_idx) & (snr_test >= lo) & (snr_test < hi)
            if mask.sum() == 0:
                continue
            Xs = torch.from_numpy(X_test[mask]).permute(0, 2, 1).float().to(device)
            with torch.no_grad():
                preds = model(Xs).argmax(1).cpu().numpy()
            acc = np.mean(preds == tgt_idx)
            print(f"    SNR [{lo:+3d}, {hi:+3d}) dB: {acc*100:.1f}%  ({mask.sum()} 样本)")


# ════════════════════════════════════════════════════════════════════════
#  双模型对比
# ════════════════════════════════════════════════════════════════════════

def compare_models(results_dict):
    """打印两个模型在 .mat 上的预测分布对比"""
    if len(results_dict) < 2:
        return
    names  = list(results_dict.keys())
    counts = list(results_dict.values())

    print("\n" + "═" * 55)
    print("  模型对比（.mat 文件预测分布）")
    print("═" * 55)
    print(f"  {'类别':<14} {names[0]:>12} {names[1]:>12}")
    print("  " + "─" * 40)
    for cls in CLASSES:
        a = counts[0].get(cls, 0)
        b = counts[1].get(cls, 0)
        print(f"  {cls:<14} {a:>12}  {b:>12}")
    print("═" * 55)


# ════════════════════════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════════════════════════

def main():
    mat_results = {}

    for model_name, weights_path in MODEL_WEIGHTS.items():
        if not os.path.exists(weights_path):
            print(f"[跳过] 权重不存在: {weights_path}")
            continue

        print(f"\n{'='*55}")
        print(f"  加载模型: {model_name}  ({weights_path})")
        print(f"{'='*55}")

        model = build_model(classes=len(CLASSES))
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()

        # 在实测 .mat 上测试
        if os.path.exists(TEST_MAT_FOLDER):
            result = eval_on_mat(model, model_name, TEST_MAT_FOLDER, NUM_SAMPLES)
            mat_results[model_name] = result
        else:
            print(f"  .mat 文件夹不存在，跳过实测评估: {TEST_MAT_FOLDER}")

        # 在仿真 npz 上测试
        eval_on_npz(model, model_name, EVAL_NPZ, target_class='128QAM')

    # 对比两个模型
    compare_models(mat_results)


if __name__ == '__main__':
    main()
