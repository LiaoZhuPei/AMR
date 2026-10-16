import h5py

h5_path = "G1_IN_S1_slice_21.h5"  # 替换为你的 h5 文件路径

def print_dataset_info(name, obj):
    """递归回调函数：如果是数据集则打印详细信息"""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset 名称 : /{name}")
        print(f"  └─ 原始形状 (Python C-Order) : {obj.shape}")
        print(f"  └─ MATLAB 对应尺寸           : {obj.shape[::-1]}")  # 倒序即对应 MATLAB 的尺寸
        print(f"  └─ 数据类型 (dtype)          : {obj.dtype}")
        print(f"  └─ 占用元素总数              : {obj.size}")
        print(f"  └─ 数据占用内存 (MB)         : {obj.nbytes / (1024 * 1024):.2f} MB\n")

# 打开并遍历文件
with h5py.File(h5_path, 'r') as f:
    print(f"========== 正在分析 H5 文件: {h5_path} ==========")
    # 打印根目录下包含的所有组/数据集名称
    print(f"顶层键名 (Keys): {list(f.keys())}\n")
    # 自动递归查看该文件内的每一个具体项
    f.visititems(print_dataset_info)
    print("==================================================")