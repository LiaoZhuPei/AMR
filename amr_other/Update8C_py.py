import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ================= 配置区域 =================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATASET_ROOT = "./dataset_tri_RAD_Tensor" # 使用 Tensor 数据集
SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "sim_train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "sim_val")
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")
RESULTS_DIR = "./training_results_ensemble"

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_MODELS = 5  # [关键] 训练 5 个模型进行集成

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= Dataset =================
class TensorDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        if not os.path.exists(root_dir): return
        files = []
        for cls_name in CLASS_NAMES:
            d = Path(root_dir) / cls_name
            if d.exists():
                files.extend([(p, CLASS_NAMES.index(cls_name)) for p in d.glob("*.pt")])
        random.shuffle(files)
        for p, label in files:
            try:
                self.data.append(torch.load(p))
                self.labels.append(label)
            except: pass
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# ================= 模型：TinyCNN (最稳的那个) =================
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.classifier(self.features(x))

# ================= 训练单个模型的函数 =================
def train_one_model(model_idx, train_loader, val_loader):
    print(f"\n>>> Training Model {model_idx+1}/{NUM_MODELS}...")
    
    # 每次重新初始化模型，保证随机性
    model = TinyCNN(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_acc = 0.0
    save_path = os.path.join(RESULTS_DIR, f"model_{model_idx}.pth")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 还是加上裁剪比较稳
            optimizer.step()
            
        # Val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                output = model(imgs)
                _, preds = output.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            
        print(f"   Epoch {epoch+1}: Val Acc = {acc:.4f} (Best: {best_acc:.4f})")
        
    print(f">>> Model {model_idx+1} Finished. Best Acc: {best_acc:.4f}")
    return save_path

# ================= 主逻辑：集成测试 =================
def main():
    print("Loading Data...")
    train_ds = TensorDataset(SIM_TRAIN_DIR)
    val_ds = TensorDataset(SIM_VAL_DIR)
    real_ds = TensorDataset(REAL_TEST_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 1. 训练 N 个模型
    model_paths = []
    for i in range(NUM_MODELS):
        path = train_one_model(i, train_loader, val_loader)
        model_paths.append(path)
        
    # 2. 加载所有模型到内存 (集成准备)
    print("\nLoading all models for Ensemble Inference...")
    models = []
    for path in model_paths:
        m = TinyCNN(NUM_CLASSES).to(DEVICE)
        m.load_state_dict(torch.load(path))
        m.eval()
        models.append(m)
        
    # 3. 集成推理 (Ensemble Inference)
    print("Running Ensemble on Real Data...")
    if len(real_ds) == 0:
        print("No real data found.")
        return

    real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE, shuffle=False)
    y_true = []
    y_pred_ensemble = []
    
    # 用于记录单个模型的准确率，做对比
    single_model_correct = [0] * NUM_MODELS
    total_samples = 0
    
    with torch.no_grad():
        for imgs, labels in real_loader:
            imgs = imgs.to(DEVICE)
            total_samples += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            
            # 收集 5 个模型的输出
            batch_outputs = []
            for i, model in enumerate(models):
                out = model(imgs) # [Batch, 12]
                batch_outputs.append(out)
                
                # 统计单个模型表现
                _, single_pred = out.max(1)
                single_model_correct[i] += single_pred.eq(labels.to(DEVICE)).sum().item()
            
            # --- 核心：软投票 (Soft Voting) ---
            # 将 Logits 叠加 (或者先Softmax再叠加，直接叠加Logits通常效果也很好)
            ensemble_out = torch.stack(batch_outputs).mean(dim=0) # [Batch, 12]
            _, ensemble_pred = ensemble_out.max(1)
            y_pred_ensemble.extend(ensemble_pred.cpu().numpy())

    # 4. 结果对比
    print("\n========== FINAL RESULTS ==========")
    for i in range(NUM_MODELS):
        acc = single_model_correct[i] / total_samples
        print(f"Model {i+1} Accuracy: {acc:.4f}")
        
    ensemble_acc = np.mean(np.array(y_true) == np.array(y_pred_ensemble))
    print(f"-----------------------------------")
    print(f"Ensemble Accuracy: {ensemble_acc:.4f} (Usually higher and stable!)")
    print(f"-----------------------------------")
    
    # 绘制混淆矩阵
    existing_idx = sorted(list(set(y_true)))
    existing_names = [CLASS_NAMES[i] for i in existing_idx]
    
    cm = confusion_matrix(y_true, y_pred_ensemble, labels=existing_idx)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=existing_names, yticklabels=existing_names)
    plt.title(f"Ensemble Confusion Matrix (Acc: {ensemble_acc:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ensemble_cm.png"))
    print(f"Saved confusion matrix to {RESULTS_DIR}")

if __name__ == "__main__":
    main()