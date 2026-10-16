# train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 导入我们的模块化文件
import config
from dataset import get_dataloaders
#import LSTM as lstm_model
import CNN as cnn_model
import mltools

def main():
    """主执行函数"""
    # --- 1. 初始化和设置 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = config.DEVICE
    print(f"Using device: {device}")

    if not os.path.exists('weights'):
        os.makedirs('weights')
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # --- 2. 数据加载 ---
    data_loaders = get_dataloaders(config)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    Y_test_onehot = data_loaders['Y_test_onehot']
    Z_test = data_loaders['Z_test']
    
    # --- 3. 模型、损失函数、优化器定义 ---
    """
    model = lstm_model.build_model(
        input_size=config.MODEL_PARAMS['input_size'],
        hidden_size=config.MODEL_PARAMS['hidden_size'],
        num_layers=config.MODEL_PARAMS['num_layers'],
        classes=len(config.CLASSES)
    ).to(device)
    """
    model = cnn_model.build_model(
        # CNN的build_model通过kwargs接收参数
        input_shape=config.MODEL_PARAMS['input_shape'],
        classes=len(config.CLASSES)
    ).to(device)
    print(model)

    # #############################################################################
    # --- 修改点: 权重分配 ---
    # #############################################################################
    # 1. 从 config 文件中获取权重列表
    #class_weights = config.CLASS_WEIGHTS
    # 2. 将 Python 列表转换为 PyTorch Tensor，并移动到 GPU
    #class_weights_tensor = torch.tensor(class_weights).float().to(device)

    #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) #权重分配
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN_PARAMS['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=config.TRAIN_PARAMS['lr_scheduler_patience'], verbose=True
    )

    # --- 4. 训练循环 ---
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n--- Starting Training ---")
    for epoch in range(config.TRAIN_PARAMS['num_epochs']):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.TRAIN_PARAMS['num_epochs']} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.TRAIN_PARAMS['gradient_clip_norm'])
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})
        
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ...")
            torch.save(model.state_dict(), config.WEIGHTS_PATH)
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.TRAIN_PARAMS['early_stopping_patience']:
                print(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
                break

    print("Training finished.")
    mltools.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
    
    # --- 5. 评估 ---
    print("\n--- Starting Evaluation ---")
    predict_pytorch(model, test_loader, device, config.WEIGHTS_PATH, Y_test_onehot, Z_test, config.CLASSES, config.FIGURE_PATHS['confusion_matrix'])


def predict_pytorch(model, test_loader, device, model_path, y_true_onehot, z_test, classes, cm_save_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds_softmax = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Predicting on Test Set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds_softmax.append(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
            
    test_y_hat = np.concatenate(all_preds_softmax, axis=0)
    cm, right, wrong = mltools.calculate_confusion_matrix(y_true_onehot, test_y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=cm_save_path)
    mltools.calculate_acc_cm_each_snr(y_true_onehot, test_y_hat, z_test, classes, min_snr=-18)


if __name__ == '__main__':
    main()