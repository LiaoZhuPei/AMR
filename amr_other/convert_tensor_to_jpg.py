import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# ================= 配置 =================
SRC_ROOT = "./dataset_tri_RAD_Tensor_VSG60" # 源数据 (.pt)
DST_ROOT = "./dataset_YOLO_VSG60"             # 目标数据 (.jpg)

MODES = ['train', 'val', 'test']
# 对应原本的 sim_train, sim_val, real_test
SRC_DIRS = {
    'train': os.path.join(SRC_ROOT, "sim_train"),
    'val':   os.path.join(SRC_ROOT, "sim_val"),
    'test':  os.path.join(SRC_ROOT, "real_test")
}

def tensor_to_image(tensor):
    # tensor: (3, 64, 64) float [0, 1]
    # 转为 numpy uint8 [0, 255]
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    if os.path.exists(DST_ROOT): shutil.rmtree(DST_ROOT)
    
    print(f"Converting .pt to .jpg for YOLO format at {DST_ROOT}...")
    
    for mode in MODES:
        src_path = SRC_DIRS[mode]
        dst_path = os.path.join(DST_ROOT, mode)
        
        if not os.path.exists(src_path): continue
        
        # 遍历类别文件夹
        for cls_name in os.listdir(src_path):
            cls_src = os.path.join(src_path, cls_name)
            cls_dst = os.path.join(dst_path, cls_name)
            os.makedirs(cls_dst, exist_ok=True)
            
            files = [f for f in os.listdir(cls_src) if f.endswith('.pt')]
            
            for fname in tqdm(files, desc=f"{mode}/{cls_name}"):
                # 读取 pt
                tensor = torch.load(os.path.join(cls_src, fname))
                # 转换
                img = tensor_to_image(tensor)
                # 保存 jpg
                img.save(os.path.join(cls_dst, fname.replace('.pt', '.jpg')))
                
    print("\nConversion Done!")

if __name__ == "__main__":
    main()