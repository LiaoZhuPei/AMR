from ultralytics import YOLO
import os
import shutil

# ================= 配置 =================
DATA_DIR = os.path.abspath("./dataset_YOLO_VSG60") # 必须绝对路径
PROJECT_DIR = "./results_yolo"
EPOCHS = 50
IMG_SIZE = 64

def main():
    if os.path.exists(PROJECT_DIR): shutil.rmtree(PROJECT_DIR)
    
    # 1. 加载预训练模型 (分类模式)
    # yolov8n-cls.pt 是最小的模型 (nano)，速度最快
    model = YOLO('yolov8n-cls.pt') 
    
    print("=== Starting YOLOv8 Classification Training ===")
    
    # 2. 训练
    # data 参数指向包含 train/val 文件夹的根目录
    results = model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=128,
        project=PROJECT_DIR,
        name='exp1',
        device=0, # 使用 GPU 0
        patience=10, # 早停
        augment=True, # 开启自带的数据增强
        # 关闭一些不适合频谱图的增强 (比如 mixup, mosaic 可能破坏特征)
        # 但 YOLO 分类模式默认增强比较温和，可以直接用
    )
    
    # 3. 验证
    print("\n=== Validating on Sim Data ===")
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    
    # 4. 测试实测数据
    # YOLO 默认没有显式的 'test' 模式，我们可以手动预测 'test' 文件夹
    print("\n=== Testing on Real Data ===")
    test_dir = os.path.join(DATA_DIR, "test")
    
    if os.path.exists(test_dir):
        # 预测并返回结果
        # stream=True 节省内存
        preds = model.predict(test_dir, imgsz=IMG_SIZE, stream=True, verbose=False)
        
        correct = 0
        total = 0
        
        for r in preds:
            # 获取预测类别索引
            pred_cls_idx = r.probs.top1
            pred_name = r.names[pred_cls_idx]
            
            # 获取真实类别 (从文件路径解析)
            # path 格式: .../test/QPSK/real_0000.jpg
            true_name = os.path.basename(os.path.dirname(r.path))
            
            if pred_name == true_name:
                correct += 1
            total += 1
            
        print(f"Real Data Accuracy: {correct/total:.4f} ({correct}/{total})")
    else:
        print("No test data found.")

if __name__ == "__main__":
    # Windows 下多进程保护
    import multiprocessing
    multiprocessing.freeze_support()
    main()