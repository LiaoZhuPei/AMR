import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools

# 导入您的 CNN+LSTM 模型文件
import CNNLSTM as cnn_lstm_model

# ==============================================================================
# --- 1. 全局配置 (Configuration) ---
# ==============================================================================
class Config:
    # --- 设备和路径配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # !! 关键：路径指向纯净的训练数据 和 带噪声的测试数据 !!
    PURE_TRAIN_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data11claI/train/train.csv' # 请替换为您的纯净训练数据CSV路径
    PURE_TRAIN_LABELS_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data11claI/train/train_labels.csv' # 请替换为您的纯净训练标签CSV路径

    TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data11claH/test/test.csv'
    TEST_LABELS_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data11claH/test/test_labels.csv'
    TEST_SNR_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data11claH/test/test_snr.csv'
    
    WEIGHTS_PATH = 'weights/pytorch_cnnlstm_augmented_v1.pt'

    # --- 类别和信号参数 ---
    CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4DQPSK',
               '16QAM', 'ASK2', '16APSK', '32APSK', '64QAM', '32QAM'] # 请根据您的标签文件确认顺序和内容
    IQ_CHANNELS = 2
    SIGNAL_LENGTH = 1024

    # --- 训练超参数 ---
    BATCH_SIZE = 400
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.01
    EARLY_STOPPING_PATIENCE = 20

    # --- 数据增强参数 ---
    AUGMENTATION_PROB = 0.5
    SAMPLE_RATE = 1e6      
    MAX_FREQ_OFFSET_HZ = 5000
    MAX_PHASE_OFFSET_RAD = np.pi / 4
    SNR_RANGE_DB = (-5, 20)
    DC_OFFSET_CLASSES = ['ASK2']
    DC_OFFSET_I_RANGE = (-0.5, -0.3)
    DC_OFFSET_Q_RANGE = (-0.05, 0.05)

# ==============================================================================
# --- 2. 模块化的数据增强函数 ---
# ==============================================================================
def per_sample_normalize(signal):
    min_val, max_val = np.min(signal), np.max(signal)
    range_val = max_val - min_val
    return 2 * (signal - min_val) / range_val - 1 if range_val > 0 else np.zeros_like(signal)

def apply_frequency_offset(signal, fs, max_offset_hz):
    offset = np.random.uniform(-max_offset_hz, max_offset_hz)
    t = np.arange(signal.shape[0]) / fs
    offset_signal = signal * np.exp(1j * 2 * np.pi * offset * t)
    return offset_signal

def apply_phase_offset(signal, max_offset_rad):
    offset = np.random.uniform(-max_offset_rad, max_offset_rad)
    return signal * np.exp(1j * offset)

def apply_dc_offset(signal, i_range, q_range):
    i_offset = np.random.uniform(*i_range)
    q_offset = np.random.uniform(*q_range)
    return signal + (i_offset + 1j * q_offset)
    
def apply_awgn(signal, snr_db_range):
    snr_db = np.random.uniform(*snr_db_range)
    signal_power = np.mean(np.abs(signal)**2)
    if signal_power == 0: return signal
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape) + \
            1j * np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    return signal + noise

def augment_pipeline(signal, label_str, cfg):
    complex_signal = signal[0, :] + 1j * signal[1, :]
    if np.random.rand() < cfg.AUGMENTATION_PROB:
        complex_signal = apply_frequency_offset(complex_signal, cfg.SAMPLE_RATE, cfg.MAX_FREQ_OFFSET_HZ)
    if np.random.rand() < cfg.AUGMENTATION_PROB:
        complex_signal = apply_phase_offset(complex_signal, cfg.MAX_PHASE_OFFSET_RAD)
    if label_str in cfg.DC_OFFSET_CLASSES and np.random.rand() < cfg.AUGMENTATION_PROB:
        complex_signal = apply_dc_offset(complex_signal, cfg.DC_OFFSET_I_RANGE, cfg.DC_OFFSET_Q_RANGE)
    complex_signal = apply_awgn(complex_signal, cfg.SNR_RANGE_DB)
    augmented_signal = np.array([complex_signal.real, complex_signal.imag], dtype=np.float32)
    return augmented_signal

# ==============================================================================
# --- 3. 统一的 PyTorch Dataset ---
# ==============================================================================
class UnifiedIQDataset(Dataset):
    def __init__(self, data, labels, class_names, is_train=False, cfg=None):
        self.data = data
        self.labels = labels
        self.class_names = class_names
        self.is_train = is_train
        self.cfg = cfg

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.data[idx] # shape (2, 1024)
        label_idx = self.labels[idx]
        
        if self.is_train:
            label_str = self.class_names[label_idx]
            augmented_signal = augment_pipeline(signal, label_str, self.cfg)
        else:
            augmented_signal = signal
        
        normalized_signal = per_sample_normalize(augmented_signal)
        
        # !! 关键修改: Dataset的输出形状需要适配CNN+LSTM模型
        # CNN+LSTM模型在forward函数中自己处理permute, 所以这里直接输出 (2, 1024) 的Tensor
        # 或者我们可以保持 (1024, 2) 的输出，然后在训练循环中处理
        # 为了与之前的LSTM代码保持一致，我们继续输出 (1024, 2)
        signal_tensor = torch.from_numpy(normalized_signal).float()
        signal_tensor = signal_tensor.permute(1, 0) # (2, 1024) -> (1024, 2)
        
        return signal_tensor, torch.tensor(label_idx).long()

# ==============================================================================
# --- 4. 数据加载函数 ---
# ==============================================================================
def load_data_from_csv(data_path, labels_path, cfg):
    print(f"Loading data from {data_path}...")
    data_flat = pd.read_csv(data_path, header=None).values
    data_reshaped = data_flat.reshape(-1, cfg.IQ_CHANNELS, cfg.SIGNAL_LENGTH)
    print(f"Loading labels from {labels_path}...")
    labels_idx = pd.read_csv(labels_path, header=None).values.flatten()
    print(f"Loaded {len(labels_idx)} samples. Shape: {data_reshaped.shape}")
    return data_reshaped, labels_idx

# ===================================================================
# --- 5. 核心训练与评估逻辑 ---
# ===================================================================
if __name__ == "__main__":
    config = Config()
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if not os.path.exists('figure'):
        os.makedirs('figure')

    print(f"Using device: {config.DEVICE}")

    # --- Phase 1: Data Loading ---
    print("\n--- Phase 1: Data Loading ---")
    pure_train_raw, pure_train_labels_idx = load_data_from_csv(config.PURE_TRAIN_DATA_PATH, config.PURE_TRAIN_LABELS_PATH, config)
    test_raw, test_labels_idx = load_data_from_csv(config.TEST_DATA_PATH, config.TEST_LABELS_PATH, config)
    Z_test = pd.read_csv(config.TEST_SNR_PATH, header=None).values
    Y_test_onehot = np.eye(len(config.CLASSES), dtype='uint8')[test_labels_idx]


    X_train, X_val, Y_train, Y_val = train_test_split(
        pure_train_raw, pure_train_labels_idx, test_size=0.2, random_state=42, stratify=pure_train_labels_idx
    )

    train_dataset = UnifiedIQDataset(X_train, Y_train, config.CLASSES, is_train=True, cfg=config)
    val_dataset = UnifiedIQDataset(X_val, Y_val, config.CLASSES, is_train=True, cfg=config)
    test_dataset = UnifiedIQDataset(test_raw, test_labels_idx, config.CLASSES, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # --- Phase 2: Model Preparation ---
    print("\n--- Phase 2: Building CNN+LSTM Model ---")
    model = cnn_lstm_model.build_model(classes=len(config.CLASSES)).to(config.DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    # --- Phase 3: Training ---
    print("\n--- Phase 3: Training the CNN+LSTM model with Real-time Augmentation ---")
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': f'{val_loss / val_total:.4f}', 'val_acc': f'{val_correct / val_total:.4f}'})
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ...")
            torch.save(model.state_dict(), config.WEIGHTS_PATH)
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
    
    print("Training finished.")
    mltools.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])

    # --- Phase 4: Final Evaluation ---
    print("\n--- Phase 4: Final Evaluation on Test Set ---")
    def predict_pytorch(model, test_loader, device, cfg):
        model.load_state_dict(torch.load(cfg.WEIGHTS_PATH, map_location=device))
        model.eval()
        all_preds_softmax = []
        predict_pbar = tqdm(test_loader, desc="Predicting on Test Set")
        with torch.no_grad():
            for inputs, _ in predict_pbar:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                all_preds_softmax.append(probabilities.cpu().numpy())
        test_Y_hat = np.concatenate(all_preds_softmax, axis=0)
        cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, cfg.CLASSES)
        acc = round(1.0 * right / (right + wrong), 4)
        print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')
        mltools.plot_confusion_matrix(cm, labels=cfg.CLASSES, save_filename='figure/pytorch_cnnlstm_augmented_confusion.png')
        mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, cfg.CLASSES, min_snr=-18)

    predict_pytorch(model, test_loader, config.DEVICE, config)

    print("Process finished.")