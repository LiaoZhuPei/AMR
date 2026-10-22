"""
dataset_generator.py

功能：
    读取原始的 CSV 数据，将其整理为三份 .npy 文件：
        1. train_known.npy   —— 包含 Train_known_target 和 Train_known_untarget
        2. test_known.npy    —— 包含 Test_known_target 和 Test_known_untarget
        3. test_unknown.npy  —— 包含 Test_unknown_untarget

    每个 .npy 文件保存为字典：
        {
            "data":   ndarray, shape=(N, 1024),  dtype=float32
            "label":  ndarray, shape=(N,),       dtype=<U32   (字符串标签, 即所在子文件夹名)
        }

    每个 CSV 文件的形状为 (M, 1025)，其中第 1025 列为原始标签，直接丢弃。
    样本的标签统一使用其所在子文件夹的名称。
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================ 路径配置 ============================
ROOT_DIR = "E:/CardRF/SVDD_DATA_CSV"   # 根目录，按需修改
SAVE_DIR = "./npy_dataset"   # 生成的 npy 保存目录

os.makedirs(SAVE_DIR, exist_ok=True)

# 各个子集里包含的文件夹（类别）
SPLIT_FOLDERS = {
    "train_known": {
        "Train_known_target":   ["APPLE_IPAD3", "APPLE_IPHONE6S", "APPLE_IPHONE7", "CISCO_LINKSYS_E3200", "FITBIT_CHARGE3", "MOTOROLA", "TPLINK_TL_WR940N"],
        "Train_known_untarget": ["DJI_M600"],
    },
    "test_known": {
        "Test_known_target":   ["APPLE_IPAD3", "APPLE_IPHONE6S", "APPLE_IPHONE7", "CISCO_LINKSYS_E3200", "FITBIT_CHARGE3", "MOTOROLA", "TPLINK_TL_WR940N"],
        "Test_known_untarget": ["DJI_M600"],
    },
    "test_unknown": {
        "Test_unknown_untarget": ["BEEBEERUN", "DJI_INSPIRE", "DJI_MAVICPRO",
                                   "DJI_PHANTOM"],
    },
}


# ============================ 工具函数 ============================
def load_csv_folder(folder_path, label):
    """
    读取某个类别文件夹下所有 csv 文件的数据。
    返回：
        data:  ndarray, shape=(sum_of_rows, 1024)
        label: ndarray, shape=(sum_of_rows,), dtype=str
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if len(csv_files) == 0:
        print(f"[Warning] 未在 {folder_path} 中找到任何 CSV 文件。")
        return None, None

    data_list = []
    for csv_file in csv_files:
        # header=None 表示 csv 没有列名；如果有列名请去掉该参数
        df = pd.read_csv(csv_file, header=None)
        arr = df.values  # (M, 1025)
        # 只保留前 1024 列
        if arr.shape[1] < 1024:
            print(f"[Warning] {csv_file} 的列数不足 1024，跳过。实际列数={arr.shape[1]}")
            continue
        arr = arr[:, :1024].astype(np.float32)
        data_list.append(arr)

    if len(data_list) == 0:
        return None, None

    data = np.concatenate(data_list, axis=0)
    labels = np.array([label] * data.shape[0], dtype=np.str_)
    return data, labels


def build_split(split_name, split_config):
    """
    构建一个 split（train_known / test_known / test_unknown）
    """
    all_data = []
    all_labels = []

    for parent_folder, class_folders in split_config.items():
        parent_path = os.path.join(ROOT_DIR, parent_folder)
        if not os.path.isdir(parent_path):
            print(f"[Warning] 不存在文件夹: {parent_path}，跳过。")
            continue

        for cls in class_folders:
            cls_path = os.path.join(parent_path, cls)
            if not os.path.isdir(cls_path):
                print(f"[Warning] 不存在类别文件夹: {cls_path}，跳过。")
                continue

            print(f"[{split_name}] 读取: {cls_path}")
            data, labels = load_csv_folder(cls_path, cls)
            if data is None:
                continue
            all_data.append(data)
            all_labels.append(labels)
            print(f"    -> shape={data.shape}, label={cls}")

    if len(all_data) == 0:
        print(f"[Warning] {split_name} 没有读到任何数据。")
        return

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    save_dict = {"data": all_data, "label": all_labels}
    save_path = os.path.join(SAVE_DIR, f"{split_name}.npy")
    np.save(save_path, save_dict, allow_pickle=True)

    print(f"\n[{split_name}] 保存完成 -> {save_path}")
    print(f"    data.shape  = {all_data.shape}")
    print(f"    label.shape = {all_labels.shape}")
    print(f"    类别分布    = {dict(zip(*np.unique(all_labels, return_counts=True)))}\n")


# ============================ 主流程 ============================
def main():
    for split_name, split_config in SPLIT_FOLDERS.items():
        print(f"\n=========== 正在生成 {split_name}.npy ===========")
        build_split(split_name, split_config)

    print("全部数据集生成完毕！")


if __name__ == "__main__":
    main()