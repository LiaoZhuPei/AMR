import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F

# 导入您自定义的CNN模型结构
import CNN as mcl

# --- 1. 基本配置 (与之前相同) ---
classes_full = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK',
                '4QAM', '8QAM', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
                '2FSK', '4FSK', '8FSK', '16FSK',
                '4PAM', '8PAM', '16PAM',
                'AM-DSB', 'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM']

MODEL_WEIGHTS_PATH = 'weights/weights.pt'
TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/HisarModdataset/Test/test.mat'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- 2. 核心预测函数 (新版本) ---

def predict_single_signal_targeted(model, signal_data, device, all_trained_classes, target_classes):
    """
    对单个信号进行调制识别预测，并将非目标类别归为"Other"。

    Args:
        model (torch.nn.Module): 加载了权重的PyTorch模型。
        signal_data (np.ndarray): 单个信号数据，期望形状为 (2, 1024)。
        device (torch.device): 运行预测的设备 (CPU or CUDA)。
        all_trained_classes (list): 模型训练时使用的完整类别列表 (例如，26个类别)。
        target_classes (list): 您关心的目标类别列表。

    Returns:
        tuple: (最终类别名称 (含"Other"), 对应的置信度)
    """
    # 检查输入信号的形状
    if signal_data.shape != (2, 1024):
        raise ValueError(f"Expected signal_data shape (2, 1024), but got {signal_data.shape}")

    # 1. 预处理和模型推理 (与之前相同)
    model.eval()
    input_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(signal_data, axis=0), axis=0)).float().to(device)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)

    # 2. 获取模型最自信的原始预测
    top_confidence, top_predicted_idx_tensor = torch.max(probabilities, 1)
    top_predicted_idx = top_predicted_idx_tensor.item()
    original_predicted_class = all_trained_classes[top_predicted_idx]

    # 3. 核心逻辑：检查预测是否在目标列表中
    if original_predicted_class in target_classes:
        # 如果预测结果是我们关心的类别，直接返回结果
        final_class = original_predicted_class
        final_confidence = top_confidence.item()
    else:
        # 如果预测结果不是我们关心的类别，将其归为 "Other"
        final_class = "Other"

        # 计算 "Other" 类的置信度。一个好的方法是：
        # 将所有不属于target_classes的类别的概率相加。

        # 创建一个布尔掩码，标记哪些是 "Other" 类
        other_mask = torch.ones(len(all_trained_classes), dtype=torch.bool)
        for target_cls in target_classes:
            try:
                idx = all_trained_classes.index(target_cls)
                other_mask[idx] = False
            except ValueError:
                # 如果目标类别不在训练列表中，忽略它
                pass

        # 使用掩码对概率进行求和
        # probabilities[0] 的形状是 [26]
        other_confidence_sum = torch.sum(probabilities[0][other_mask.to(device)])
        final_confidence = other_confidence_sum.item()

    return final_class, final_confidence


# --- 3. 主执行程序 (展示如何使用新函数) ---

if __name__ == "__main__":
    print(f"Using device: {device}")

    # 定义我们只关心的类别
    TARGET_CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'FM']
    print(f"\nTarget classes of interest: {TARGET_CLASSES}")

    # 检查权重文件是否存在
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
    else:
        # a. 加载模型结构和权重
        print("Loading model and weights...")
        model = mcl.CNN(classes=len(classes_full))
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)

        # b. 加载测试数据
        print(f"Loading test data from '{TEST_DATA_PATH}'...")
        try:
            with h5py.File(TEST_DATA_PATH, 'r') as data_file:
                all_test_data = data_file['data_save'][:].swapaxes(0, 2)

            # --- 示例 1: 预测一个可能在目标列表中的信号 (QPSK, 8PSK等在数据集中较早出现) ---
            sample_index_1 = 500
            signal_1 = all_test_data[sample_index_1]
            print(f"\n--- Predicting for Sample #{sample_index_1} (likely a target class) ---")

            pred_class_1, conf_1 = predict_single_signal_targeted(
                model=model,
                signal_data=signal_1,
                device=device,
                all_trained_classes=classes_full,
                target_classes=TARGET_CLASSES
            )
            print(f"Final Predicted Class: {pred_class_1}")
            print(f"Confidence: {conf_1 * 100:.2f}%")
            if pred_class_1 != "Other":
                print("(The model's top prediction was in our target list.)")
            else:
                print("(The model's top prediction was not in our target list.)")

            # --- 示例 2: 预测一个很可能不在目标列表中的信号 (AM-DSB等在数据集中靠后出现) ---
            # AM-DSB 类别索引为20, 20 * 3000 = 60000, 我们取一个这个范围的索引
            sample_index_2 = 61000
            signal_2 = all_test_data[sample_index_2]
            print(f"\n--- Predicting for Sample #{sample_index_2} (likely an 'Other' class) ---")

            pred_class_2, conf_2 = predict_single_signal_targeted(
                model=model,
                signal_data=signal_2,
                device=device,
                all_trained_classes=classes_full,
                target_classes=TARGET_CLASSES
            )
            print(f"Final Predicted Class: {pred_class_2}")
            print(f"Confidence: {conf_2 * 100:.2f}%")
            if pred_class_2 != "Other":
                print("(The model's top prediction was in our target list.)")
            else:
                print("(The model's top prediction was not in our target list.)")
                print("(Confidence represents the sum of probabilities of all non-target classes.)")


        except FileNotFoundError:
            print(f"Error: Test data file not found at '{TEST_DATA_PATH}'")
        except Exception as e:
            print(f"An error occurred: {e}")
