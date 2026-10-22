import numpy as np
import os
from pathlib import Path

# 配置
ROOT_DIR = "datasets_validated/datasets_validated"
SEGMENT_LENGTH = 2048
SPECIAL_FOLDER = "FM_broadcast"  # 有200个文件的文件夹

folders = [
    "FM_broadcast",
    "NOAA_weather", 
    "APRS",
    "pager",
    "ISM_sensors",
    "FRS_GMRS",
    "noise"
]

def load_samples_from_npy(file_path):
    """从npy文件中加载samples的IQ数据"""
    data = np.load(file_path, allow_pickle=True).item()
    samples = data['samples'].astype(np.complex64)
    return samples

def split_into_segments(samples, segment_length=2048):
    """将samples切分为固定长度的段，丢弃不足一段的尾部"""
    total_length = len(samples)
    n_segments = total_length // segment_length
    # 只保留完整的段
    samples_trimmed = samples[:n_segments * segment_length]
    segments = samples_trimmed.reshape(n_segments, segment_length)
    return segments

def process_folder(folder_name, root_dir, segment_length=2048):
    """处理单个文件夹"""
    folder_path = os.path.join(root_dir, folder_name)
    
    # 获取所有npy文件，并排序保证顺序一致
    npy_files = sorted([
        f for f in os.listdir(folder_path) 
        if f.endswith('.npy')
    ])
    
    total_files = len(npy_files)
    print(f"\n{'='*50}")
    print(f"处理文件夹: {folder_name}")
    print(f"文件总数: {total_files}")
    
    # 根据文件夹确定分割点
    if folder_name == SPECIAL_FOLDER:
        half = total_files // 2  # 200个文件 -> 前100，后100
    else:
        half = total_files // 2  # 100个文件 -> 前50，后50
    
    print(f"前半部分: {half} 个文件，后半部分: {total_files - half} 个文件")
    
    # 分成前后两组文件列表
    first_half_files = npy_files[:half]
    second_half_files = npy_files[half:]
    
    # ---- 处理前半部分 ----
    print(f"  正在处理前半部分 ({len(first_half_files)} 个文件)...")
    first_half_segments = []
    for fname in first_half_files:
        fpath = os.path.join(folder_path, fname)
        try:
            samples = load_samples_from_npy(fpath)
            segs = split_into_segments(samples, segment_length)
            first_half_segments.append(segs)
        except Exception as e:
            print(f"    警告: 加载 {fname} 失败: {e}")
    
    if first_half_segments:
        first_half_array = np.concatenate(first_half_segments, axis=0).astype(np.complex64)
    else:
        first_half_array = np.empty((0, segment_length), dtype=np.complex64)
    
    print(f"    前半部分切分结果 shape: {first_half_array.shape}")
    
    # ---- 处理后半部分 ----
    print(f"  正在处理后半部分 ({len(second_half_files)} 个文件)...")
    second_half_segments = []
    for fname in second_half_files:
        fpath = os.path.join(folder_path, fname)
        try:
            samples = load_samples_from_npy(fpath)
            segs = split_into_segments(samples, segment_length)
            second_half_segments.append(segs)
        except Exception as e:
            print(f"    警告: 加载 {fname} 失败: {e}")
    
    if second_half_segments:
        second_half_array = np.concatenate(second_half_segments, axis=0).astype(np.complex64)
    else:
        second_half_array = np.empty((0, segment_length), dtype=np.complex64)
    
    print(f"    后半部分切分结果 shape: {second_half_array.shape}")
    
    # ---- 合并全部 ----
    if first_half_array.size > 0 and second_half_array.size > 0:
        all_array = np.concatenate([first_half_array, second_half_array], axis=0).astype(np.complex64)
    elif first_half_array.size > 0:
        all_array = first_half_array
    else:
        all_array = second_half_array
    
    print(f"    全部合并结果 shape: {all_array.shape}")
    
    # ---- 保存结果 ----
    save_dir = folder_path  # 保存到原文件夹下
    
    # 文件1: 所有样本切分后
    all_save_path = os.path.join(save_dir, f"{folder_name}_all_segments.npy")
    np.save(all_save_path, all_array)
    print(f"  已保存: {all_save_path}  shape={all_array.shape} dtype={all_array.dtype}")
    
    # 文件2: 前半部分切分后
    first_save_path = os.path.join(save_dir, f"{folder_name}_first_half_segments.npy")
    np.save(first_save_path, first_half_array)
    print(f"  已保存: {first_save_path}  shape={first_half_array.shape} dtype={first_half_array.dtype}")
    
    # 文件3: 后半部分切分后
    second_save_path = os.path.join(save_dir, f"{folder_name}_second_half_segments.npy")
    np.save(second_save_path, second_half_array)
    print(f"  已保存: {second_save_path}  shape={second_half_array.shape} dtype={second_half_array.dtype}")
    
    return {
        "folder": folder_name,
        "all_shape": all_array.shape,
        "first_half_shape": first_half_array.shape,
        "second_half_shape": second_half_array.shape
    }


def main():
    print("开始处理数据集...")
    print(f"根目录: {ROOT_DIR}")
    print(f"切片长度: {SEGMENT_LENGTH}")
    
    results = []
    for folder in folders:
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"\n警告: 文件夹不存在，跳过: {folder_path}")
            continue
        result = process_folder(folder, ROOT_DIR, SEGMENT_LENGTH)
        results.append(result)
    
    # 汇总打印
    print(f"\n{'='*50}")
    print("处理完成！汇总信息：")
    print(f"{'='*50}")
    print(f"{'文件夹':<20} {'全部shape':<25} {'前半shape':<25} {'后半shape':<25}")
    print("-" * 95)
    for r in results:
        print(f"{r['folder']:<20} {str(r['all_shape']):<25} {str(r['first_half_shape']):<25} {str(r['second_half_shape']):<25}")

if __name__ == "__main__":
    main()