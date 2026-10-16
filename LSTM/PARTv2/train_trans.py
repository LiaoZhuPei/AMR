# train.py

import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import model as model_module
import dataset as dataset_module
import utils


def load_data(data_config):
    # ... (此函数无需修改，保持原样) ...
    print("Loading data...")
    p = data_config['root_path']

    train_data_flat = pd.read_csv(os.path.join(p, data_config['train_data']), header=None).values
    train_raw = train_data_flat.reshape(-1, data_config['iq_channels'], data_config['signal_length'])

    test_data_flat = pd.read_csv(os.path.join(p, data_config['test_data']), header=None).values
    test_raw = test_data_flat.reshape(-1, data_config['iq_channels'], data_config['signal_length'])

    train_labels = pd.read_csv(os.path.join(p, data_config['train_labels']), header=None).values.flatten()
    test_labels = pd.read_csv(os.path.join(p, data_config['test_labels']), header=None).values.flatten()
    test_snr = pd.read_csv(os.path.join(p, data_config['test_snr']), header=None).values

    print(f"Loaded {len(train_raw)} training samples and {len(test_raw)} test samples.")
    return train_raw, train_labels, test_raw, test_labels, test_snr


def setup_model_for_transfer_learning(model, config):
    """为迁移学习配置模型：加载权重并冻结层"""
    if not config['transfer_learning']['enabled']:
        print("Transfer learning is disabled. Training from scratch.")
        return

    pretrained_path = config['transfer_learning']['pretrained_path']
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found at {pretrained_path}. Training from scratch.")
        return

    print(f"Loading pretrained weights from: {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. 筛选出预训练模型中与新模型匹配的权重 (主要是lstm层)
    # 我们只加载键名相同且权重形状也相同的层
    weights_to_load = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # 2. 更新当前模型的权重字典
    model_dict.update(weights_to_load)

    # 3. 使用 strict=False 加载权重，因为分类头的权重不存在于weights_to_load中
    model.load_state_dict(model_dict, strict=True)  # 使用更新后的完整字典，可以strict=True

    print(f"Successfully loaded {len(weights_to_load)} layers from the pretrained model.")
    for name, _ in weights_to_load.items():
        print(f"  - Loaded layer: {name}")

    # 4. 根据配置冻结特征提取器 (LSTM层)
    if config['transfer_learning']['freeze_feature_extractor']:
        print("Freezing feature extractor (LSTM layers)...")
        for name, param in model.named_parameters():
            if 'lstm' in name:
                param.requires_grad = False


def main(config):
    """主函数，执行整个训练和评估流程"""
    # --- 1. 环境和设备设置 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # --- 2. 数据加载和准备 ---
    # ... (此部分无需修改) ...
    train_raw, train_labels, test_raw, test_labels, _ = load_data(config['data'])
    X_train, X_val, Y_train, Y_val = train_test_split(
        train_raw, train_labels,
        test_size=config['data']['val_split_ratio'],
        random_state=config['seed'],
        stratify=train_labels
    )

    train_dataset = dataset_module.IQDataset(X_train, Y_train, config['data']['apply_normalization'])
    val_dataset = dataset_module.IQDataset(X_val, Y_val, config['data']['apply_normalization'])
    test_dataset = dataset_module.IQDataset(test_raw, test_labels, config['data']['apply_normalization'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                              **config['dataloader'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                            **config['dataloader'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                             **config['dataloader'])

    # --- 3. 模型、损失函数、优化器定义 ---
    num_classes = len(config['data']['classes'])
    model = model_module.build_model(config['model'], num_classes)  # 首先创建新模型

    # --- 核心修改：应用迁移学习设置 ---
    setup_model_for_transfer_learning(model, config)
    model.to(device)
    print(model)

    # 优化器现在需要能处理冻结层的情况
    # 只将需要更新的参数传递给优化器
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_update, lr=config['training']['optimizer']['lr'])

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config['training']['scheduler'])

    # --- 4. 训练循环 ---
    output_dir = os.path.join(config['output']['directory'], config['run_name'])
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'best_model.pt')

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config['training']['num_epochs']):
        # --- 核心修改：微调逻辑 ---
        # 检查是否到达微调的epoch
        if config['transfer_learning']['enabled'] and epoch == config['training']['fine_tune_after_epoch']:
            print("\n--- Starting Fine-tuning phase ---")
            print("Unfreezing all layers and reducing learning rate.")
            for param in model.parameters():
                param.requires_grad = True

            # 更新优化器以包含所有参数，并使用新的学习率
            optimizer = optim.Adam(model.parameters(), lr=config['training']['optimizer']['fine_tune_lr'])
            # 为新的优化器重置学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config['training']['scheduler'])

        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']} [Train]")
        for inputs, labels in train_pbar:
            # ... (训练循环内部逻辑无需修改) ...
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['grad_clip_norm'])
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix(
                {'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # ... (验证循环和模型保存逻辑无需修改) ...
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix(
                    {'val_loss': f'{val_loss / val_total:.4f}', 'val_acc': f'{val_correct / val_total:.4f}'})

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ...")
            torch.save(model.state_dict(), filepath)
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping']['patience']:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    utils.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])

    # --- 5. 评估 ---
    evaluate(model, test_loader, device, filepath, config)
    print("Process finished.")


# evaluate函数和主执行部分 if __name__ == '__main__': ... 无需修改

def evaluate(model, test_loader, device, model_path, config):
    # ... (此函数无需修改，保持原样) ...
    print("\n--- Starting Evaluation ---")
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    _, _, _, test_labels, test_snr = load_data(config['data'])
    classes = config['data']['classes']
    y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels]

    cm, right, wrong = utils.calculate_confusion_matrix(y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')

    if not os.path.exists('figure'):
        os.makedirs('figure')
    cm_filename = f'figure/{config["run_name"]}_confusion_matrix.png'
    utils.plot_confusion_matrix(cm, labels=classes, save_filename=cm_filename)

    utils.calculate_acc_cm_each_snr(y_test_onehot, test_Y_hat, test_snr, classes,
                                    min_snr=config['evaluation']['min_snr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PyTorch model based on a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)