"""
统一预测脚本：适配 4800 长度信号，采用滑动窗口 + 投票机制
只需修改 MODEL_NAME 和 MODEL_WEIGHTS_PATH
"""
import os
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

# ╔══════════════════════════════════════════════════════════════╗
# ║  修改这里切换模型                                              ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME = 'cnn'  # 选项: 'transformer', 'tcn', 'mamba'

# 根据你的项目路径导入模型
if MODEL_NAME == 'cnn':
    from cnn_frontend import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn import build_model
elif MODEL_NAME == 'mamba':
    from model_mamba import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# --- 配置 ---
classes = ['OQPSK', 'Pi4DQPSK']
MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_sionna_two_A.pt'
DATASET_PATH = './sionna_dataset/Sionna_head_mid1.npz'  # 我们刚生成的测试集路径

# 滑动窗口参数
SIG_LEN = 4800
WIN_LEN = 1024
STEP = 800
NUM_SAMPLES_TO_TEST = 1000  # 测试前100个样本看看效果

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using model: {MODEL_NAME}")


# --- 预处理（必须与训练一致） ---
def power_normalize(signal_data):
    """功率归一化，针对 (2, 1024) 的窗口数据"""
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


def predict_single_window(model, window_data, device, class_list, temperature=1.0):
    """
    输入: window_data shape (1, 2, 1024)
    输出: 预测类别字符串，置信度
    """
    model.eval()
    sample = window_data.squeeze(0)  # (1, 2, 1024)->(2, 1024)
    input_tensor = torch.from_numpy(sample).float()
    input_tensor = input_tensor.permute(1, 0)  # (1024, 2)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 1024, 2)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output / temperature, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

    return class_list[predicted_idx.item()], confidence.item()


def sliding_window_predict_and_vote(model, signal, device, class_list, win_len, step):
    """
    对单条长信号进行滑动窗口预测并投票
    输入 signal: (2, 4800) numpy array
    返回: 最终投票类别, 投票状态字典, 所有窗口的预测详情
    """
    L = signal.shape[1]
    predictions = []
    details = []

    # 滑动窗口遍历
    for start_idx in range(0, L - win_len + 1, step):
        end_idx = start_idx + win_len

        # 1. 截取窗口
        window_data = signal[:, start_idx:end_idx]  # (2, 1024)

        # 2. 归一化并整理形状为 (1, 2, 1024)
        window_data = power_normalize(window_data)
        window_data = window_data.reshape(1, 2, win_len)

        # 3. 输入模型预测
        pred_class, conf = predict_single_window(model, window_data, device, class_list)

        predictions.append(pred_class)
        details.append(f"{pred_class}({conf * 100:.1f}%)")

    # 4. 根据你的特定规则进行投票
    class_A = class_list[0]  # 'OQPSK'
    class_B = class_list[1]  # 'Pi4DQPSK'

    count_A = predictions.count(class_A)
    count_B = predictions.count(class_B)

    vote_status = {class_A: count_A, class_B: count_B}

    if count_A > count_B:
        final_pred = class_A
    elif count_A < count_B:
        final_pred = class_B
    else:
        final_pred = '无法预测'

    return final_pred, vote_status, details


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 检查权重并加载模型
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights not found at '{MODEL_WEIGHTS_PATH}'")
        exit()

    model = build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{MODEL_NAME}] Parameters: {total_params:,}")

    # 2. 加载 .npz 测试集数据
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at '{DATASET_PATH}'")
        exit()

    print(f"Loading data from {DATASET_PATH}...")
    dataset = np.load(DATASET_PATH)
    X_test = dataset['X_test']  # (N, 2, 4800)
    Y_test = dataset['Y_test']  # (N,)
    true_classes = dataset['classes']

    total_test_samples = len(Y_test)
    num_to_test = min(NUM_SAMPLES_TO_TEST, total_test_samples)
    print(f"Testing {num_to_test} samples with [{MODEL_NAME}]...\n")
    print("-" * 110)
    print(f"{'Idx':<5} | {'真实标签':<10} | {'最终预测':<10} | {'投票状态':<25} | {'各窗口预测详情'}")
    print("-" * 110)

    # 统计信息
    correct_count = 0
    unpredicted_count = 0

    # 3. 循环测试每个信号
    for i in range(num_to_test):
        signal = X_test[i]  # (2, 4800)
        true_label = true_classes[Y_test[i]]

        # 滑动窗口预测
        final_pred, vote_status, details = sliding_window_predict_and_vote(
            model, signal, device, classes, WIN_LEN, STEP
        )

        # 记录准确率
        if final_pred == true_label:
            correct_count += 1
        elif final_pred == '无法预测':
            unpredicted_count += 1

        # 格式化投票状态为字符串
        vote_str = f"A({vote_status[classes[0]]}) : B({vote_status[classes[1]]})"
        details_str = " -> ".join(details)

        # 打印输出 (若预测错误，在行尾加上 '❌' 提醒)
        mark = "✅" if final_pred == true_label else ("⚠️" if final_pred == '无法预测' else "❌")
        print(f"[{i + 1:<3}] | {true_label:<10} | {final_pred:<10} | {vote_str:<25} | {details_str} {mark}")

    # 4. 输出最终准确率汇总
    print(f"\n{'=' * 50}")
    print(f"  [{MODEL_NAME}] 滑动窗口投票预测汇总")
    print(f"{'=' * 50}")
    print(f"  总测试样本数 : {num_to_test}")
    print(f"  预测正确数   : {correct_count}")
    print(f"  无法预测数   : {unpredicted_count}")
    print(f"  错误数       : {num_to_test - correct_count - unpredicted_count}")
    print(f"  最终准确率   : {(correct_count / num_to_test) * 100:.2f} %")
    print(f"{'=' * 50}")