from ultralytics import YOLO
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= 配置 =================
# [关键] 指向你刚刚训练好的最佳权重文件
# 根据你的日志，它在 results_yolo/exp1/weights/best.pt
MODEL_PATH = "./results_yolo/exp1/weights/best.pt"

# [关键] 指向转换为 JPG 后的实测数据目录
TEST_DATA_DIR = "./dataset_YOLO_VSG60/test"

# 完整的 12 个类别名称 (顺序必须与训练时一致)
# YOLO 通常按字母顺序排序类别，但为了保险，我们使用模型内部的 names
CLASS_NAMES_12 = [
    "16APSK", "16QAM", "2ASK", "32APSK", "32QAM", "64QAM", 
    "8PSK", "BPSK", "FSK2", "OQPSK", "PI4DQPSK", "QPSK"
]
# 注意：上面的列表只是参考，代码里会直接读取模型里的 names 映射

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # 获取模型内部的类别映射 {0: '16APSK', 1: '16QAM', ...}
    model_names = model.names
    print(f"Model classes: {model_names}")

    # 2. 遍历测试数据
    y_true = []
    y_pred = []
    
    # 遍历 test 文件夹下的所有子文件夹
    subfolders = [f.path for f in os.scandir(TEST_DATA_DIR) if f.is_dir()]
    
    print(f"Found {len(subfolders)} classes in Real Test Data.")
    
    for folder in subfolders:
        class_name = os.path.basename(folder)
        
        # 检查该文件夹是否在模型的类别里
        if class_name not in model_names.values():
            print(f"[Warn] Unknown class folder '{class_name}', skipping...")
            continue
            
        # 获取该类别的 index
        # 这是一个反向查找: name -> index
        class_idx = list(model_names.keys())[list(model_names.values()).index(class_name)]
        
        # 获取该文件夹下所有图片
        imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        
        print(f"Processing {class_name} ({len(imgs)} images)...")
        
        # 批量推理 (Batch inference) 以加速
        # YOLO predict 可以接受列表
        results = model.predict(imgs, verbose=False, stream=True)
        
        for r in results:
            # 记录真实标签
            y_true.append(class_idx)
            # 记录预测标签 (top1)
            y_pred.append(r.probs.top1)

    # 3. 计算准确率
    if len(y_true) == 0:
        print("No images found!")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.mean(y_true == y_pred)
    
    print(f"\n==========================================")
    print(f"REAL DATA ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"==========================================")

    # 4. 绘制混淆矩阵
    # 我们希望混淆矩阵显示所有 12 个类别，即使实测数据只有 9 个
    # 这样能看到有没有把实测数据误判成"不存在的类别"
    
    # 获取排序后的类别名列表
    all_class_names = [model_names[i] for i in range(len(model_names))]
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(model_names)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_class_names, yticklabels=all_class_names)
    plt.title(f"Real Data Confusion Matrix (Acc: {acc:.2%})\nModel: YOLOv8-Cls")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = "./results_yolo/real_data_cm.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()