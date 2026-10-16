import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION — 参数配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    'sim_npz_path':         './sionna_dataset/Sionna0_b.npz',
    'real_dirs_map': {
        '16QAM':  'E:/Project/AMR-Benchmark/output_mat_files_16QAM_DDC',
        '32QAM':  'E:/Project/AMR-Benchmark/output_mat_files_32QAM_DDC',
        '64QAM':  'E:/Project/AMR-Benchmark/output_mat_files_64QAM_DDC',
        '128QAM': 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC',
    },
    'mat_dataset_name':     'data_save',
    
    # 优化后的训练控制
    'batch_size':           128,        # 增大 Batch Size (1D 信号较小，128 可显著提升 GPU 并行度)
    'lr':                   2e-4,
    'beta1':                0.5,
    'beta2':                0.999,
    'epochs':               40,
    'steps_per_epoch':      200,        # 每个 Epoch 固定的迭代次数 (200 steps * 128 batch = 25,600 样本/epoch)
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    
    'lambda_cycle':         10.0,   
    'lambda_id':            5.0,    
    'lambda_spec':          2.0,    
    
    'checkpoint_dir':       './checkpoints_cyclegan_fast',
    'output_synthetic_dir': './synthetic_dataset_fast'
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. OPTIMIZED DATASET — 内存预载与极速加载器                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
class FastMultiClassAlignedDataset(Dataset):
    """
    重构后的极速加载器：
    1. 一次性将实测数据和仿真数据载入内存并完成归一化。
    2. 训练过程中纯内存读取，彻底消除磁盘 I/O 瓶颈。
    """
    def __init__(self, sim_npz_path, real_dirs_map, mat_dataset_name='data_save', steps_per_epoch=200, batch_size=128):
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.total_samples_per_epoch = steps_per_epoch * batch_size

        # 1. 加载并缓存仿真数据到内存
        print("[Dataset] 正在预载仿真数据到内存...")
        sim_data = np.load(sim_npz_path, allow_pickle=True)
        raw_sim_signals = sim_data['X_train']  # Shape: (N, 2, 1024)
        raw_sim_labels = sim_data['Y_train']    # Shape: (N,)
        self.classes_list = list(sim_data['classes'])

        # 2. 加载、归一化并缓存所有实测数据到内存
        print("[Dataset] 正在预载实测 .mat 数据并执行内存归一化...")
        self.real_data_by_class = {}
        self.available_classes = []

        for cls_name, folder_path in real_dirs_map.items():
            if cls_name not in self.classes_list:
                continue
            
            files = glob.glob(os.path.join(folder_path, '*.mat'))
            if len(files) == 0:
                continue
            
            cls_idx = self.classes_list.index(cls_name)
            self.real_data_by_class[cls_idx] = []
            
            # 一次性读取该类别下所有文件至内存
            for mat_path in tqdm(files, desc=f"预载 {cls_name}", leave=False):
                with h5py.File(mat_path, 'r') as f:
                    arr = np.array(f[mat_dataset_name], dtype=np.float32)[0] # (2, 1024)
                    # 提前做好归一化
                    arr = arr / (np.max(np.abs(arr)) + 1e-8)
                    self.real_data_by_class[cls_idx].append(torch.from_numpy(arr))
            
            if len(self.real_data_by_class[cls_idx]) > 0:
                self.available_classes.append(cls_idx)

        # 3. 按参与训练的类别，过滤并提前归一化仿真数据
        print("[Dataset] 正在预处理仿真数据并建立快速索引...")
        self.sim_data_by_class = {c: [] for c in self.available_classes}
        for sig, lbl in zip(raw_sim_signals, raw_sim_labels):
            if lbl in self.available_classes:
                # 提前归一化
                norm_sig = sig / (np.max(np.abs(sig)) + 1e-8)
                self.sim_data_by_class[lbl].append(torch.from_numpy(norm_sig.astype(np.float32)))

        print(f"[Dataset] 内存预载完成。参与训练的类别: {[self.classes_list[c] for c in self.available_classes]}")

    def __len__(self):
        # 限制 Epoch 长度，使得训练进度能够频繁反馈与保存
        return self.total_samples_per_epoch

    def __getitem__(self, idx):
        # 1. 随机选定一个类别
        cls_idx = np.random.choice(self.available_classes)

        # 2. 从内存直接获取对应的仿真和实测张量（避免任何磁盘读取或归一化计算）
        sim_list = self.sim_data_by_class[cls_idx]
        real_list = self.real_data_by_class[cls_idx]

        x_sim = sim_list[np.random.randint(0, len(sim_list))]
        x_real = real_list[np.random.randint(0, len(real_list))]

        return x_sim, x_real


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. NETWORK ARCHITECTURE — 保持原一维卷积网络                        ║
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
                nn.ConvTranspose1d(curr_channels, curr_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
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
# ║  4. SOLVER / TRAINER — 核心训练循环                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
class CycleGANTrainer1D:
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

    def train_epoch(self, dataloader, epoch):
        self.G_S2R.train()
        self.G_R2S.train()
        self.D_R.train()
        self.D_S.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']}", ncols=100)
        
        for i, (sim_signals, real_signals) in enumerate(pbar):
            # 将张量快速移动到 GPU
            sim_signals = sim_signals.to(self.device, non_blocking=True)
            real_signals = real_signals.to(self.device, non_blocking=True)

            # ---------------------------
            # 1. 训练生成器
            # ---------------------------
            self.optimizer_G.zero_grad(set_to_none=True) # set_to_none=True 略微加速梯度归零

            # 恒等损失
            loss_id_R = self.criterion_identity(self.G_S2R(real_signals), real_signals)
            loss_id_S = self.criterion_identity(self.G_R2S(sim_signals), sim_signals)
            loss_identity = (loss_id_R + loss_id_S) * self.cfg['lambda_id']

            # 对抗损失
            fake_real = self.G_S2R(sim_signals)
            pred_fake_real = self.D_R(fake_real)
            loss_gan_S2R = self.criterion_gan(pred_fake_real, torch.ones_like(pred_fake_real))

            fake_sim = self.G_R2S(real_signals)
            pred_fake_sim = self.D_S(fake_sim)
            loss_gan_R2S = self.criterion_gan(pred_fake_sim, torch.ones_like(pred_fake_sim))
            loss_gan = loss_gan_S2R + loss_gan_R2S

            # 循环损失
            reconstructed_sim = self.G_R2S(fake_real)
            loss_cycle_sim = self.criterion_cycle(reconstructed_sim, sim_signals)

            reconstructed_real = self.G_S2R(fake_sim)
            loss_cycle_real = self.criterion_cycle(reconstructed_real, real_signals)
            loss_cycle = (loss_cycle_sim + loss_cycle_real) * self.cfg['lambda_cycle']

            # 频域物理约束
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

            # D_R 损失
            pred_real = self.D_R(real_signals)
            loss_real_D_R = self.criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake_R = self.D_R(fake_real.detach())
            loss_fake_D_R = self.criterion_gan(pred_fake_R, torch.zeros_like(pred_fake_R))
            loss_D_R = (loss_real_D_R + loss_fake_D_R) * 0.5

            # D_S 损失
            pred_sim = self.D_S(sim_signals)
            loss_real_D_S = self.criterion_gan(pred_sim, torch.ones_like(pred_sim))
            pred_fake_S = self.D_S(fake_sim.detach())
            loss_fake_D_S = self.criterion_gan(pred_fake_S, torch.zeros_like(pred_fake_S))
            loss_D_S = (loss_real_D_S + loss_fake_D_S) * 0.5

            loss_D = loss_D_R + loss_D_S
            loss_D.backward()
            self.optimizer_D.step()

            pbar.set_postfix({
                'G': f"{loss_G.item():.2f}",
                'D': f"{loss_D.item():.2f}",
                'Spec': f"{loss_spectral.item():.2f}"
            })

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.cfg['checkpoint_dir'], f"cyclegan_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'G_S2R_state_dict': self.G_S2R.state_dict(),
        }, checkpoint_path)
        print(f"  [Checkpoint] Saved -> {checkpoint_path}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. INFERENCE — 快速生成批处理                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def generate_synthetic_dataset(trainer, config):
    print("\n--- 开始批量泛化生成高保真合成数据集 ---")
    trainer.G_S2R.eval()
    
    data = np.load(config['sim_npz_path'], allow_pickle=True)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_test, Y_test = data['X_test'], data['Y_test']
    
    def convert_batch(signals):
        converted_signals = []
        batch_size = 512 # 推理阶段不需要梯度，进一步调大 Batch
        with torch.no_grad():
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i+batch_size]
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
        X_test=X_test_synthetic,   Y_test=Y_test,
        snr_test=data['snr_test'],
        classes=data['classes'],
        snr_list=data['snr_list']
    )
    print(f"\n[Done] 转换完毕！合成数据集已保存至 -> {out_path}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. MAIN — 启动入口                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    # 1. 内存预载数据集
    dataset = FastMultiClassAlignedDataset(
        sim_npz_path=CONFIG['sim_npz_path'],
        real_dirs_map=CONFIG['real_dirs_map'],
        mat_dataset_name=CONFIG['mat_dataset_name'],
        steps_per_epoch=CONFIG['steps_per_epoch'],
        batch_size=CONFIG['batch_size']
    )

    # 2. 启用多线程并行，使用 pin_memory 加快 CPU->GPU 拷贝
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=0,          # 启用多线程
        pin_memory=True,        # 锁页内存，加速数据搬运
        drop_last=True
    )

    # 3. 初始化训练器
    trainer = CycleGANTrainer1D(CONFIG)

    # 4. 极速训练循环
    print(f"\n开始极速联合训练... (当前设备: {CONFIG['device']})")
    for epoch in range(CONFIG['epochs']):
        trainer.train_epoch(dataloader, epoch)
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)

    # 5. 执行数据生成
    generate_synthetic_dataset(trainer, CONFIG)