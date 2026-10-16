import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# [关键修改] 这里指向你的 4 个消融实验数据集之一
# 例如: ./dataset_Ablation/NewFeat_Rotated
DATASET_ROOT = "./dataset_Ablation/Exp4_New_Rotated" 

SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "val")

# [关键修改] 结果保存目录 (每次实验改名，防覆盖)
RESULTS_DIR = "./results_Exp4_New_Rotated" 

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50 # 消融实验不需要跑太久，50轮足够看趋势

os.makedirs(RESULTS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")

# ================= 1. 数据集定义 =================
class InMemoryDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            print(f"[Error] Path not found: {root_dir}")
            return
            
        print(f"Loading {root_dir}...")
        files = []
        for cls_name in CLASS_NAMES:
            d = Path(root_dir) / cls_name
            if d.exists(): 
                files.extend([(p, CLASS_NAMES.index(cls_name)) for p in d.glob("*.pt")])
        
        with tqdm(total=len(files), unit="file") as pbar:
            for p, label in files:
                try:
                    tensor = torch.load(p) # (3, 64, 64)
                    self.data.append(tensor) 
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading {p}: {e}")
                pbar.update(1)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# ================= 2. 模型定义 (TinyCNN) =================
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            # 输入 3 通道 (适配 Old 和 New 特征)
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# ================= 3. 辅助函数 =================
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val') 
    plt.title('Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val')
    plt.title('Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ================= 4. 主训练逻辑 =================
def main():
    print(f"=== Ablation Experiment: {DATASET_ROOT} ===")
    
    train_ds = InMemoryDataset(SIM_TRAIN_DIR)
    val_ds = InMemoryDataset(SIM_VAL_DIR)
    
    if len(train_ds) == 0:
        print("No data found! Check DATASET_ROOT.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = TinyCNN(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"\nStarting Training ({NUM_EPOCHS} epochs)...")
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * imgs.size(0)
            _, preds = out.max(1)
            correct += preds.eq(labels).sum().item(); total += labels.size(0)
        
        t_loss = r_loss / total
        t_acc = correct / total
        
        # Val
        model.eval()
        correct, total = 0, 0
        v_loss_sum = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                v_loss_sum += loss.item() * imgs.size(0)
                _, preds = out.max(1)
                correct += preds.eq(labels).sum().item(); total += labels.size(0)
        
        v_loss = v_loss_sum / total
        v_acc = correct / total
        
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1:02d} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"\nTraining Finished. Best Val Acc: {best_val_acc:.4f}")
    plot_curves(history, os.path.join(RESULTS_DIR, "curves.png"))
    
    # 简单的验证结果输出
    print(f"Experiment Results saved to: {RESULTS_DIR}")
    with open(os.path.join(RESULTS_DIR, "final_score.txt"), "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

if __name__ == "__main__":
    main()