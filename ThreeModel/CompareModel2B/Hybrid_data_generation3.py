import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from tqdm import tqdm

# 禁用 cuDNN 的算法自动寻优，防止 Windows 端在第一步时卡死
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. SYSTEM DIAGNOSTIC — 显卡状态诊断                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
print("=" * 60)
print("  系统运行环境诊断:")
cuda_available = torch.cuda.is_available()
print(f"    PyTorch 能否检测到 GPU (CUDA):  {cuda_available}")
if cuda_available:
    print(f"    当前显卡设备名称:                {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. CONFIGURATION — 参数配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    'sim_npz_path': './sionna_dataset/Sionna0_b.npz',
    'real_dirs_map': {
        '16QAM': 'E:/Project/AMR-Benchmark/output_mat_files_16QAM_DDC',
        '32QAM': 'E:/Project/AMR-Benchmark/output_mat_files_32QAM_DDC',
        '64QAM': 'E:/Project/AMR-Benchmark/output_mat_files_64QAM_DDC',
        '128QAM': 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC',
    },
    'mat_dataset_name': 'data_save',

    # 调整 Batch Size 以彻底释放显卡压力，避免 Shared Memory 交换
    'batch_size': 32,  # 降至 32，大幅降低显存占用
    'lr': 2e-4,
    'beta1': 0.5,
    'beta2': 0.999,
    'epochs': 40,
    'steps_per_epoch': 200,
    'device': 'cuda' if cuda_available else 'cpu',

    'lambda_cycle': 10.0,
    'lambda_id': 5.0,
    'lambda_spec': 2.0,

    'checkpoint_dir': './checkpoints_cyclegan_fast',
    'output_synthetic_dir': './synthetic_dataset_fast'
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. RAM-GPU HYBRID MANAGER — 内存-显存混合托管管理器                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
class HybridDatasetManager:
    """
    内存-显存混合托管：
    1. 将静态大矩阵完整保留在 CPU 内存（RAM）中，占用 0 显存。
    2. 每次迭代时，仅在内存中提取 256 KB 的小 Batch 发送至 GPU。
    """

    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config['device'])

        # 1. 载入仿真数据
        print("[Manager] 正在载入仿真数据...")
        sim_data = np.load(config['sim_npz_path'], allow_pickle=True)
        raw_sim_signals = sim_data['X_train']
        raw_sim_labels = sim_data['Y_train']
        self.classes_list = list(sim_data['classes'])

        print("[Manager] 正在进行整块矩阵归一化...")
        max_vals_sim = np.max(np.abs(raw_sim_signals), axis=(1, 2), keepdims=True) + 1e-8
        norm_sim_signals = (raw_sim_signals / max_vals_sim).astype(np.float32)

        # 2. 载入实测数据
        self.sim_tensors_cpu = {}
        self.real_tensors_cpu = {}
        self.available_classes = []

        for cls_name, folder_path in config['real_dirs_map'].items():
            if cls_name not in self.classes_list:
                continue

            files = glob.glob(os.path.join(folder_path, '*.mat'))
            if len(files) == 0:
                continue

            cls_idx = self.classes_list.index(cls_name)

            # 读取该类别所有的 mat
            temp_list = []
            for mat_path in files:
                with h5py.File(mat_path, 'r') as f:
                    arr = np.array(f[config['mat_dataset_name']], dtype=np.float32)[0]
                    temp_list.append(arr)

            # 转换为整块矩阵并进行归一化
            stacked_real = np.stack(temp_list, axis=0)
            max_vals_real = np.max(np.abs(stacked_real), axis=(1, 2), keepdims=True) + 1e-8
            norm_real = (stacked_real / max_vals_real).astype(np.float32)

            # 保持在 CPU 内存上（不占用显卡显存）
            self.real_tensors_cpu[cls_idx] = torch.from_numpy(norm_real)

            mask = (raw_sim_labels == cls_idx)
            self.sim_tensors_cpu[cls_idx] = torch.from_numpy(norm_sim_signals[mask])

            self.available_classes.append(cls_idx)
            print(
                f"  -> {cls_name} 已载入 RAM: 仿真 {self.sim_tensors_cpu[cls_idx].shape} | 实测 {self.real_tensors_cpu[cls_idx].shape}")

        print("[Manager] 内存初始化完毕，显存占用保持在极低水平。")

    def get_batch(self, batch_size):
        """在内存中随机切片，然后将极小的数据包发送至 GPU"""
        cls_idx = np.random.choice(self.available_classes)

        sim_t = self.sim_tensors_cpu[cls_idx]
        real_t = self.real_tensors_cpu[cls_idx]

        # 内存快速索引
        idx_sim = torch.randint(0, sim_t.size(0), (batch_size,))
        idx_real = torch.randint(0, real_t.size(0), (batch_size,))

        # 切片提取，并发送至 GPU
        sim_batch_gpu = sim_t[idx_sim].to(self.device, non_blocking=True)
        real_batch_gpu = real_t[idx_real].to(self.device, non_blocking=True)

        return sim_batch_gpu, real_batch_gpu


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. NETWORK ARCHITECTURE — 1D 网络定义                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
class ResNetBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm1d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_resnet_blocks=6):
        super(Generator1D, self).__init__()
        model = [
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=1, padding=7, padding_mode='reflect'),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        ]
        curr_channels = 64
        for _ in range(2):
            model += [
                nn.Conv1d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm1d(curr_channels * 2),
                nn.ReLU(inplace=True)
            ]
            curr_channels *= 2
        for _ in range(num_resnet_blocks):
            model += [ResNetBlock1D(curr_channels)]
        for _ in range(2):
            model += [
                nn.ConvTranspose1d(curr_channels, curr_channels // 2, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.InstanceNorm1d(curr_channels // 2),
                nn.ReLU(inplace=True)
            ]
            curr_channels //= 2
        model += [
            nn.Conv1d(curr_channels, out_channels, kernel_size=15, stride=1, padding=7, padding_mode='reflect'),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator1D(nn.Module):
    def __init__(self, in_channels=2):
        super(Discriminator1D, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv1d(in_filters, out_filters, kernel_size=15, stride=2, padding=7)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv1d(512, 1, kernel_size=15, stride=1, padding=7)
        )

    def forward(self, x):
        return self.model(x)


class SpectralConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpectralConsistencyLoss, self).__init__()

    def forward(self, x, y):
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])
        y_complex = torch.complex(y[:, 0, :], y[:, 1, :])
        spec_x = torch.abs(torch.fft.fft(x_complex, dim=-1))
        spec_y = torch.abs(torch.fft.fft(y_complex, dim=-1))
        return nn.functional.l1_loss(spec_x, spec_y)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. SOLVER — 混合管理训练循环                                         ║
# ╚══════════════════════════════════════════════════════════════════════╗
class CycleGANTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config['device'])

        self.G_S2R = Generator1D().to(self.device)
        self.G_R2S = Generator1D().to(self.device)
        self.D_R = Discriminator1D().to(self.device)
        self.D_S = Discriminator1D().to(self.device)

        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_spectral = SpectralConsistencyLoss()

        self.optimizer_G = optim.Adam(
            list(self.G_S2R.parameters()) + list(self.G_R2S.parameters()),
            lr=config['lr'], betas=(config['beta1'], config['beta2'])
        )
        self.optimizer_D = optim.Adam(
            list(self.D_R.parameters()) + list(self.D_S.parameters()),
            lr=config['lr'], betas=(config['beta1'], config['beta2'])
        )
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train_epoch(self, manager, epoch):
        self.G_S2R.train()
        self.G_R2S.train()
        self.D_R.train()
        self.D_S.train()

        steps = self.cfg['steps_per_epoch']
        pbar = tqdm(range(steps), desc=f"Epoch {epoch + 1}/{self.cfg['epochs']}", ncols=100)

        for step in pbar:
            # 获取已经推送到 GPU 上的极小 Batch 数据
            sim_signals, real_signals = manager.get_batch(self.cfg['batch_size'])

            # ---------------------------
            # 1. 训练生成器
            # ---------------------------
            self.optimizer_G.zero_grad(set_to_none=True)

            loss_id_R = self.criterion_identity(self.G_S2R(real_signals), real_signals)
            loss_id_S = self.criterion_identity(self.G_R2S(sim_signals), sim_signals)
            loss_identity = (loss_id_R + loss_id_S) * self.cfg['lambda_id']

            fake_real = self.G_S2R(sim_signals)
            pred_fake_real = self.D_R(fake_real)
            loss_gan_S2R = self.criterion_gan(pred_fake_real, torch.ones_like(pred_fake_real))

            fake_sim = self.G_R2S(real_signals)
            pred_fake_sim = self.D_S(fake_sim)
            loss_gan_R2S = self.criterion_gan(pred_fake_sim, torch.ones_like(pred_fake_sim))
            loss_gan = loss_gan_S2R + loss_gan_R2S

            reconstructed_sim = self.G_R2S(fake_real)
            loss_cycle_sim = self.criterion_cycle(reconstructed_sim, sim_signals)

            reconstructed_real = self.G_S2R(fake_sim)
            loss_cycle_real = self.criterion_cycle(reconstructed_real, real_signals)
            loss_cycle = (loss_cycle_sim + loss_cycle_real) * self.cfg['lambda_cycle']

            loss_spec_S2R = self.criterion_spectral(fake_real, sim_signals)
            loss_spec_R2S = self.criterion_spectral(fake_sim, real_signals)
            loss_spectral = (loss_spec_S2R + loss_spec_R2S) * self.cfg['lambda_spec']

            loss_G = loss_identity + loss_gan + loss_cycle + loss_spectral
            loss_G.backward()
            self.optimizer_G.step()

            # ---------------------------
            # 2. 训练判别器
            # ---------------------------
            self.optimizer_D.zero_grad(set_to_none=True)

            pred_real = self.D_R(real_signals)
            loss_real_D_R = self.criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake_R = self.D_R(fake_real.detach())
            loss_fake_D_R = self.criterion_gan(pred_fake_R, torch.zeros_like(pred_fake_R))
            loss_D_R = (loss_real_D_R + loss_fake_D_R) * 0.5

            pred_sim = self.D_S(sim_signals)
            loss_real_D_S = self.criterion_gan(pred_sim, torch.ones_like(pred_sim))
            pred_fake_S = self.D_S(fake_sim.detach())
            loss_fake_D_S = self.criterion_gan(pred_fake_S, torch.zeros_like(pred_fake_S))
            loss_D_S = (loss_real_D_S + loss_fake_D_S) * 0.5

            loss_D = loss_D_R + loss_D_S
            loss_D.backward()
            self.optimizer_D.step()

            pbar.set_postfix({
                'G_L': f"{loss_G.item():.2f}",
                'D_L': f"{loss_D.item():.2f}"
            })

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.cfg['checkpoint_dir'], f"cyclegan_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch,
            'G_S2R_state_dict': self.G_S2R.state_dict(),
        }, checkpoint_path)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. INFERENCE — 批量导出转换                                         ║
# ╚══════════════════════════════════════════════════════════════════════╗
def generate_synthetic_dataset(trainer, config):
    print("\n--- 开始批量转换仿真信号为高保真合成信号 ---")
    trainer.G_S2R.eval()

    data = np.load(config['sim_npz_path'], allow_pickle=True)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_test, Y_test = data['X_test'], data['Y_test']

    def convert_batch(signals):
        converted_signals = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i + batch_size]
                max_vals = np.max(np.abs(batch), axis=(1, 2), keepdims=True) + 1e-8
                norm_batch = batch / max_vals

                tensor_batch = torch.from_numpy(norm_batch).float().to(trainer.device)
                fake_real_tensor = trainer.G_S2R(tensor_batch)

                fake_real_numpy = fake_real_tensor.cpu().numpy() * max_vals
                converted_signals.append(fake_real_numpy)
        return np.concatenate(converted_signals, axis=0)

    print("转换 Train 集...")
    X_train_synthetic = convert_batch(X_train)
    print("转换 Test 集...")
    X_test_synthetic = convert_batch(X_test)

    os.makedirs(config['output_synthetic_dir'], exist_ok=True)
    out_path = os.path.join(config['output_synthetic_dir'], 'Sionna_Synthetic_Fast.npz')

    np.savez_compressed(
        out_path,
        X_train=X_train_synthetic, Y_train=Y_train,
        X_test=X_test_synthetic, Y_test=Y_test,
        snr_test=data['snr_test'],
        classes=data['classes'],
        snr_list=data['snr_list']
    )
    print(f"\n[Done] 转换完毕！合成数据集已保存至 -> {out_path}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. MAIN — 执行主入口                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    # 1. 初始化内存管理（占用 0 显存，极安全）
    manager = HybridDatasetManager(CONFIG)

    # 2. 初始化训练器
    trainer = CycleGANTrainer(CONFIG)

    # 3. 开启极速训练
    print(f"\n开始极速联合训练... (运行设备: {CONFIG['device']})")
    for epoch in range(CONFIG['epochs']):
        trainer.train_epoch(manager, epoch)
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)

    # 4. 执行数据生成
    generate_synthetic_dataset(trainer, CONFIG)