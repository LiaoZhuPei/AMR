# main_train_yolo.py
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO
import torch

# ================= 配置区域 =================
DATASET_ROOT = os.path.abspath("./dataset_Spectrogram_Image_VSG60")
RESULTS_DIR = os.path.abspath("./training_results_yolo_Spectrogram")

IMAGE_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= 1. 数据集格式适配 =================
def prepare_yolo_dataset(root_dir):
    """
    只需把 sim_train/val 改为 train/val。
    [关键修改]：不要把 real_test 改为 test！以防 YOLO 自动加载报错。
    """
    mapping = {
        "sim_train": "train",
        "sim_val": "val"
    }
    for old_name, new_name in mapping.items():
        old_path = os.path.join(root_dir, old_name)
        new_path = os.path.join(root_dir, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
            
    #[补救措施] 如果上一次运行已经把 real_test 改成了 test，这里自动改回来
    wrong_test_path = os.path.join(root_dir, "test")
    correct_test_path = os.path.join(root_dir, "real_test")
    if os.path.exists(wrong_test_path) and not os.path.exists(correct_test_path):
        os.rename(wrong_test_path, correct_test_path)
        print("Reverted 'test' back to 'real_test' to avoid YOLO internal errors.")

# ================= 2. 可视化辅助函数 =================
def plot_custom_confusion_matrix(y_true, y_pred, title, save_path):
    unique_classes = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ================= 3. 主流程 =================
def main():
    print("Checking and preparing dataset structure for YOLO...")
    prepare_yolo_dataset(DATASET_ROOT)
    
    # ---------------- 训练阶段 ----------------
    print("\n--- Starting YOLOv11 Classification Training ---")
    model = YOLO('yolov8n-cls.pt') 
    
    results = model.train(
        data=DATASET_ROOT,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        project=RESULTS_DIR,
        name="yolo_vsg_run",
        exist_ok=True,
        workers=0,  # <---[关键修改] 解决 Windows 系统下 Dataloader 扫描完后卡死/死锁的问题
        device="0" if torch.cuda.is_available() else "cpu"
    )
    
    # ---------------- 测试/推理阶段 ----------------
    print("\n--- Testing on Real Data ---")
    
    best_model_path = os.path.join(RESULTS_DIR, "yolo_vsg_run", "weights", "best.pt")
    if not os.path.exists(best_model_path):
        print(f"Cannot find best model at {best_model_path}")
        return
        
    infer_model = YOLO(best_model_path)
    
    # [关键修改] 直接读取 real_test 文件夹，而不是 test 文件夹
    test_dir = os.path.join(DATASET_ROOT, "real_test") 
    
    if not os.path.exists(test_dir):
        print("Test directory 'real_test' not found, skipping inference.")
        return

    y_true_str =[]
    y_pred_str =[]
    
    for cls_name in os.listdir(test_dir):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
            
        img_paths = glob.glob(os.path.join(cls_dir, "*.jpg"))
        if not img_paths:
            continue
            
        print(f"Predicting class [{cls_name}] with {len(img_paths)} samples...")
        
        for i in range(0, len(img_paths), BATCH_SIZE):
            batch = img_paths[i : i+BATCH_SIZE]
            preds = infer_model.predict(source=batch, verbose=False)
            
            for res in preds:
                top1_idx = res.probs.top1
                pred_class_name = infer_model.names[top1_idx]
                
                y_true_str.append(cls_name)      
                y_pred_str.append(pred_class_name) 

    if len(y_true_str) > 0:
        y_true_str = np.array(y_true_str)
        y_pred_str = np.array(y_pred_str)
        
        acc = np.mean(y_true_str == y_pred_str)
        print(f"\nReal Data Overall Accuracy: {acc:.2%}")
        
        cm_path = os.path.join(RESULTS_DIR, "yolo_vsg_run", "real_test_confusion_matrix.png")
        plot_custom_confusion_matrix(y_true_str, y_pred_str, f"Real Test Data (Acc: {acc:.2%})", cm_path)
        print(f"Confusion Matrix saved to {cm_path}")
    else:
        print("No valid testing data found in 'real_test' directory.")

if __name__ == "__main__":
    main()