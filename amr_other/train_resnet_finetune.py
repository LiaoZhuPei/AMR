import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ================= 配置区域 =================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# [修改] 你的数据集路径 (Ultimate)
DATASET_ROOT = "./dataset_tri_RAD_Tensor_ISI"
SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "sim_train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "sim_val")
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")

RESULTS_DIR = "./results_resnet_finetune_ISI"

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 15 # 早停耐心值 (Epochs)

os.makedirs(RESULTS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_resnet.pth")

# ================= 1. 数据集定义 (保持不变) =================
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

# ================= 2. 模型定义 (ResNet18 预训练) =================
def get_resnet_model(num_classes, pretrained=True):
    """
    加载 ResNet18 并修改全连接层
    """
    print(f"Loading ResNet18 (Pretrained={pretrained})...")
    # 加载预训练模型
    # weights='IMAGENET1K_V1' 替代旧版的 pretrained=True
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
        
    # ResNet 的第一层 conv1 是 (3, 64, 7, 7)，正好适配我们的 3通道输入
    # 无需修改 conv1
    
    # 修改最后一层全连接层 (fc)
    # 原始 fc: Linear(in_features=512, out_features=1000)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# ================= 3. 辅助函数 =================
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    plt.close()

def plot_cm(y_true, y_pred, acc):
    cm = confusion_matrix(y_true, y_pred)
    idx = sorted(list(set(y_true) | set(y_pred)))
    names = [CLASS_NAMES[i] for i in idx]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title(f"Real Data CM (Acc: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "real_cm.png"))
    plt.close()

# ================= 4. 主流程 =================
def main():
    print(f"=== Training ResNet18 on {DEVICE} ===")
    
    # 1. 加载数据
    train_ds = InMemoryDataset(SIM_TRAIN_DIR)
    val_ds = InMemoryDataset(SIM_VAL_DIR)
    real_ds = InMemoryDataset(REAL_TEST_DIR)
    
    if len(train_ds) == 0: return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 初始化模型
    model = get_resnet_model(NUM_CLASSES, pretrained=False).to(DEVICE)
    
    # 优化器 & 损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率衰减: 当 val_acc 不提升时，减半 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # 3. 训练循环 (含早停)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    print(f"\nStarting training ({NUM_EPOCHS} epochs, Patience={PATIENCE})...")
    
    for epoch in range(NUM_EPOCHS):
        start_t = time.time()
        
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
        v_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                v_loss_sum += criterion(out, labels).item() * imgs.size(0)
                _, preds = out.max(1)
                v_correct += preds.eq(labels).sum().item(); v_total += labels.size(0)
        
        v_loss = v_loss_sum / v_total
        v_acc = v_correct / v_total
        
        # 记录与调度
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)
        scheduler.step(v_acc)
        
        dur = time.time() - start_t
        print(f"Epoch {epoch+1:02d} | {dur:.1f}s | Train: {t_acc:.4f} | Val: {v_acc:.4f}")
        
        # 早停逻辑
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("  >>> Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                print("\nEarly Stopping Triggered!")
                break

    plot_history(history)
    
    # 4. 实测数据测试
    print("\nTesting Real Data...")
    if len(real_ds) > 0:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval()
        loader = DataLoader(real_ds, batch_size=BATCH_SIZE)
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                _, preds = model(imgs).max(1)
                y_true.extend(labels.cpu().numpy()); y_pred.extend(preds.cpu().numpy())
        
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"\nREAL DATA ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
        plot_cm(y_true, y_pred, acc)
    else:
        print("No real data found.")

if __name__ == "__main__":
    main()