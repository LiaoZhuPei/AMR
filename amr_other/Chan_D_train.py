# main_train_tensor.py
import os
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
from tqdm import tqdm
import numpy as np

# ================= 配置区域 =================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# [修改] 对应 Tensor 数据集
DATASET_ROOT = "./dataset_tri_RAD_Tensor_VSG60"
SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "sim_train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "sim_val")
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")
RESULTS_DIR = "./training_results_tensor_VSG60" 

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

os.makedirs(RESULTS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_VSG60.pth")

# ================= 1. 数据集定义 (读取 .pt) =================
class InMemoryDataset(Dataset):
    def __init__(self, root_dir, has_snr=False):
        self.data = []
        self.labels = []
        self.snrs = [] 
        
        if not os.path.exists(root_dir):
            print(f"Path not found: {root_dir}")
            return
            
        print(f"Loading {root_dir} tensors into RAM...")
        files = []
        for cls_name in CLASS_NAMES:
            d = Path(root_dir) / cls_name
            if d.exists(): 
                # [关键] 搜索 .pt 文件
                files.extend([(p, CLASS_NAMES.index(cls_name)) for p in d.glob("*.pt")])
        
        with tqdm(total=len(files), unit="file") as pbar:
            for p, label in files:
                try:
                    snr = -999.0
                    if has_snr:
                        try:
                            snr = float(p.stem.split('_')[-1])
                        except: pass
                    
                    # [核心修改] 直接加载张量
                    # torch.load 加载出来就是 (3, 64, 64) 的 float32 Tensor
                    tensor = torch.load(p)
                    
                    self.data.append(tensor) 
                    self.labels.append(label)
                    self.snrs.append(snr)
                except Exception as e:
                    print(f"Error loading {p}: {e}")
                    pass
                pbar.update(1)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx], self.snrs[idx]

# ================= 2. 模型定义 (恢复 3通道输入) =================
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            # [关键] 输入通道恢复为 3 (对应 (3, 64, 64))
            # 这样卷积核就能同时看到 3 个维度的特征，精度最高
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), # 64x64 图片经3次池化变为 8x8
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# ================= 3. 辅助函数 (保持不变) =================
def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val') 
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, save_path, target_names=None, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if target_names is None: target_names = CLASS_NAMES
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ================= 4. 主训练逻辑 =================
def main():
    print("Initializing Tensor Dataset (3x64x64)...")
    train_ds = InMemoryDataset(SIM_TRAIN_DIR, has_snr=False) 
    val_ds = InMemoryDataset(SIM_VAL_DIR, has_snr=True) 
    real_ds = InMemoryDataset(REAL_TEST_DIR, has_snr=False)
    
    if len(train_ds) == 0:
        print("No data found! Run data_generator_tensor.py first.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = TinyCNN(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"\nStarting Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels, _ in train_loader: 
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss_sum += criterion(outputs, labels).item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Epoch {epoch+1} | Train: {epoch_acc:.4f} | Val: {val_acc:.4f} [Saved Best]")
        else:
            print(f"Epoch {epoch+1} | Train: {epoch_acc:.4f} | Val: {val_acc:.4f}")

    plot_training_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))
    
    # --- 测试部分 (包含筛选功能) ---
    print("\nLoading Best Model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    if len(real_ds) > 0:
        print("Testing Real Data...")
        real_loader = DataLoader(real_ds, batch_size=BATCH_SIZE)
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels, _ in real_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                _, preds = model(imgs).max(1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = np.mean(y_true == y_pred)
        print(f"Real Data Accuracy: {acc:.4f}")
        
        # 动态筛选混淆矩阵
        existing_indices = sorted(list(set(y_true)))
        existing_names = [CLASS_NAMES[i] for i in existing_indices]
        print(f"Plotting CM for: {existing_names}")
        
        plot_confusion_matrix(y_true, y_pred, f"Real Data (Acc: {acc:.2%})", 
                              os.path.join(RESULTS_DIR, "real_cm.png"),
                              target_names=existing_names, labels=existing_indices)
    else:
        print("No real data for testing.")

if __name__ == "__main__":
    main()