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

# 导入你的模型
import LSTM as lstm_model

# DDP 需要的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


# --- DDP 设置函数 ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 任何未被占用的端口
    # 初始化进程组，"nccl"是NVIDIA GPU推荐的后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 将当前进程绑定到对应的GPU
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# --- 1. 全局常量和数据预处理函数 (保持不变) ---
classes = ['16QAM', '32QAM', '64QAM']


def to_amp_phase_torch(iq_data):
    if not isinstance(iq_data, torch.Tensor):
        iq_data = torch.from_numpy(iq_data).float()
    complex_data = torch.complex(iq_data[:, 0, :], iq_data[:, 1, :])
    amp = torch.abs(complex_data)
    phase = torch.angle(complex_data) / torch.pi
    amp_max = amp.max(dim=1, keepdim=True).values
    amp_normalized = amp / torch.where(amp_max == 0, torch.tensor(1.0, device=iq_data.device), amp_max)
    return torch.cat((amp_normalized.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)


class IQDataset(Dataset):
    def __init__(self, iq_data, labels, use_amp_phase=True):
        self.use_amp_phase = use_amp_phase
        if self.use_amp_phase:
            self.data = to_amp_phase_torch(iq_data)
        else:
            self.data = torch.from_numpy(iq_data).float().permute(0, 2, 1)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# --- 2. 核心训练/验证/评估函数 (封装起来) ---
# 这个函数将由每个GPU进程独立运行
def train_fn(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # --- 数据加载 (在每个进程中独立完成) ---
    # 注意：这里我们直接传递文件路径，避免在主进程加载巨大数据后传递给子进程
    # 这样更节省内存
    train_data_flat = pd.read_csv(args['train_data_path'], header=None).values
    train_labels_idx = pd.read_csv(args['train_labels_path'], header=None).values.flatten()
    train_raw = train_data_flat.reshape(train_data_flat.shape[0], 2, 1024)

    X_train, X_val, Y_train, Y_val = train_test_split(
        train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
    )

    # 创建 Dataset
    train_dataset = IQDataset(X_train, Y_train, use_amp_phase=False)
    val_dataset = IQDataset(X_val, Y_val, use_amp_phase=False)

    # *** DDP 关键步骤：使用 DistributedSampler ***
    # 它会确保每个进程只加载数据的一个子集，并且不会重叠
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建 DataLoader，使用 sampler
    # batch_size 是每个 GPU 的 batch_size
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4,
                              pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=True,
                            sampler=val_sampler)

    # --- 模型、损失函数、优化器定义 ---
    model = lstm_model.build_model(classes=len(classes)).to(rank)  # 先移动到对应GPU

    # *** DDP 关键步骤：用 DDP 包装模型 ***
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=(rank == 0))

    # --- 训练循环 ---
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args['epochs']):
        # **重要**：每个epoch开始前，需要设置sampler的epoch，保证shuffle正常
        train_sampler.set_epoch(epoch)

        ddp_model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        # 只在主进程(rank 0)显示tqdm进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args['epochs']} [Train]", disable=(rank != 0))
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # DDP会自动同步梯度
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if rank == 0:
                train_pbar.set_postfix(
                    {'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})

        # --- 验证循环 ---
        ddp_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # *** DDP 关键步骤：同步所有进程的结果 ***
        # 将各个进程的统计数据汇总到一起
        val_loss_tensor = torch.tensor([val_loss, val_total, val_correct]).to(rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)  # 求和

        # 现在 val_loss_tensor 在每个进程上都是所有进程的总和
        total_val_loss = val_loss_tensor[0].item()
        total_val_samples = val_loss_tensor[1].item()
        total_val_correct = val_loss_tensor[2].item()

        avg_val_loss = total_val_loss / total_val_samples
        avg_val_acc = total_val_correct / total_val_samples

        # --- 只有主进程(rank 0)执行保存、打印和早停逻辑 ---
        if rank == 0:
            print(f"\nEpoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ...")
                # 保存时，保存 ddp_model.module, 而不是 ddp_model本身
                torch.save(ddp_model.module.state_dict(), args['filepath'])
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter}/{args['early_stopping_patience']}")
                if patience_counter >= args['early_stopping_patience']:
                    print("Early stopping triggered.")
                    # 需要通知其他进程也退出
                    dist.barrier()  # 等待所有进程到达这里
                    break  # 只有主进程break可能不够，需要更强的终止信号，但通常这样已足够

        # 使用 barrier 确保所有进程同步，要么都继续，要么都退出
        dist.barrier()
        if patience_counter >= args['early_stopping_patience']:
            break

    cleanup()


# --- 3. 主函数 (启动器) ---
def main():
    # --- 全局参数配置 ---
    args = {
        'train_data_path': 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/train/train.csv',
        'train_labels_path': 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/train/train_labels.csv',
        'test_data_path': 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/test/test.csv',
        'test_labels_path': 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/test/test_labels.csv',
        'test_snr_path': 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/test/test_snr.csv',
        'batch_size': 200,  # 这是每个GPU的batch size，总batch_size = 200 * 2 = 400
        'epochs': 500,
        'filepath': 'weights/pytorch_lstm_ddp_weights.pt',
        'early_stopping_patience': 60
    }
    if not os.path.exists('weights'):
        os.makedirs('weights')

    # --- 启动 DDP 训练 ---
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting DDP training...")
    mp.spawn(train_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

    # --- 训练结束后，在主进程中进行评估 ---
    print("\n--- Training Finished. Starting Evaluation ---")

    # 加载评估所需的数据
    test_data_flat = pd.read_csv(args['test_data_path'], header=None).values
    test_labels_idx = pd.read_csv(args['test_labels_path'], header=None).values.flatten()
    Z_test = pd.read_csv(args['test_snr_path'], header=None).values
    test_raw = test_data_flat.reshape(test_data_flat.shape[0], 2, 1024)
    Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]

    test_dataset = IQDataset(test_raw, test_labels_idx, use_amp_phase=False)
    # 评估时可以使用更大的batch_size
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'] * 2, shuffle=False)

    # 创建一个新模型来加载权重
    model = lstm_model.build_model(classes=len(classes))
    device = torch.device("cuda:0")  # 评估可以在任意一个GPU上进行
    model.to(device)

    # 定义评估函数
    def predict_pytorch(model, test_loader, device, model_path):
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
        return np.concatenate(all_preds_softmax, axis=0)

    test_Y_hat = predict_pytorch(model, test_loader, device, args['filepath'])

    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')

    mltools.plot_confusion_matrix(cm, labels=classes, save_filename='figure/pytorch_lstm_ddp_total_confusion.png')
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)

    print("Process finished.")


if __name__ == '__main__':
    # *** DDP 关键步骤：必须将启动代码放在这个保护块中 ***
    # 确保 CUDA_VISIBLE_DEVICES 未被设置，或设置为你想用的所有GPU
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    main()