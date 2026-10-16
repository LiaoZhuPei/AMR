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
# ║  1. CONFIGURATION — 多类路径与参数集中配置                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # 仿真数据集路径
    'sim_npz_path':         './sionna_dataset/Sionna0_b.npz',
    
    # 实测数据集字典：将调制名称映射到对应的实测数据文件夹
    'real_dirs_map': {
        '16QAM':  'E:/Project/AMR-Benchmark/output_mat_files_16QAM_DDC',
        '32QAM':  'E:/Project/AMR-Benchmark/output_mat_files_32QAM_DDC',
        '64QAM':  'E:/Project/AMR-Benchmark/output_mat_files_64QAM_DDC',
        '128QAM': 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC',
    },
    'mat_dataset_name':     'data_save',
    
    # 训练超参数
    'batch_size':           32,
    'lr':                   2e-4,
    'beta1':                0.5,
    'beta2':                0.999,
    'epochs':               50,
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 损失权重
    'lambda_cycle':         10.0,   
    'lambda_id':            5.0,    
    'lambda_spec':          2.0,    # 频域一致性损失权重 (物理先验)
    
    # 输出保存
    'checkpoint_dir':       './checkpoints_cyclegan_multiclass',
    'output_synthetic_dir': './synthetic_dataset_multiclass'
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. DATASET DEFINITION — 类对齐多路加载数据集                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
class MultiClassAlignedUnpairedDataset(Dataset):
    """
    支持多类对齐的非配对数据加载器。
    确保每次返回的仿真信号和实测信号属于相同的调制类别。
    """
    def __init__(self, sim_npz_path, real_dirs_map, mat_dataset_name='data_save'):
        self.mat_dataset_name = mat_dataset_name
        self.real_dirs_map = real_dirs_map

        # 1. 加载仿真数据并建立类别索引
        if not os.path.exists(sim_npz_path):
            raise FileNotFoundError(f"Simulated dataset not found at {sim_npz_path}")
        
        sim_data = np.load(sim_npz_path, allow_pickle=True)
        self.sim_signals = sim_data['X_train'] # Shape: (N, 2, 1024)
        self.sim_labels = sim_data['Y_train']   # Shape: (N,)
        self.classes_list = list(sim_data['classes']) # 仿真中的所有类别名称列表
        
        # 2. 解析实测数据路径并与仿真类别建立映射
        self.real_files_by_class = {}
        self.available_train_classes = [] # 既有仿真又有实测的类别

        for cls_name, folder_path in real_dirs_map.items():
            if cls_name not in self.classes_list:
                print(f"[Warning] 实测文件夹对应的类别 {cls_name} 不在仿真的 classes 列表中，将跳过。")
                continue
            
            files = glob.glob(os.path.join(folder_path, '*.mat'))
            if len(files) > 0:
                # 获取该类别在仿真中的整数标签
                cls_idx = self.classes_list.index(cls_name)
                self.real_files_by_class[cls_idx] = files
                self.available_train_classes.append(cls_idx)
                print(f"[Dataset] 类别 {cls_name} (ID: {cls_idx}): 找到 {len(files)} 个实测 .mat 文件")
            else:
                print(f"[Warning] 文件夹 {folder_path} 内未找到 .mat 文件，跳过该类。")

        if len(self.available_train_classes) == 0:
            raise ValueError("没有找到任何可用于对齐训练的实测类别。请检查 CONFIG 中的文件夹路径。")

        # 3. 过滤出仅包含可用训练类别的仿真数据索引，用于构建训练循环
        self.train_sim_indices = []
        for idx, lbl in enumerate(self.sim_labels):
            if lbl in self.available_train_classes:
                self.train_sim_indices.append(idx)
        
        print(f"[Dataset] 共有 {len(self.available_train_classes)} 个类别参与域转换训练。")
        print(f"参与训练的仿真样本数: {len(self.train_sim_indices)}")

    def __len__(self):
        return len(self.train_sim_indices)

    def __getitem__(self, idx):
        # 1. 获取仿真信号及对应的类别标签
        sim_data_idx = self.train_sim_indices[idx]
        x_sim = self.sim_signals[sim_data_idx].astype(np.float32)
        cls_idx = self.sim_labels[sim_data_idx]

        # 2. 从同类别的实测数据中随机抽取一个样本
        real_files_list = self.real_files_by_class[cls_idx]
        random_real_idx = np.random.randint(0, len(real_files_list))
        mat_path = real_files_list[random_real_idx]

        with h5py.File(mat_path, 'r') as f:
            x_real = np.array(f[self.mat_dataset_name], dtype=np.float32)[0]

        # 3. 归一化
        x_sim = x_sim / (np.max(np.abs(x_sim)) + 1e-8)
        x_real = x_real / (np.max(np.abs(x_real)) + 1e-8)

        return torch.from_numpy(x_sim), torch.from_numpy(x_real)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. NETWORK ARCHITECTURE — 1D-CycleGAN 模型定义                      ║
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
        # 下采样
        curr_channels = 64
        for _ in range(2):
            model += [
                nn.Conv1d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm1d(curr_channels * 2),
                nn.ReLU(inplace=True)
            ]
            curr_channels *= 2
        # 残差网络
        for _ in range(num_resnet_blocks):
            model += [ResNetBlock1D(curr_channels)]
        # 上采样
        for _ in range(2):
            model += [
                nn.ConvTranspose1d(curr_channels, curr_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm1d(curr_channels // 2),
                nn.ReLU(inplace=True)
            ]
            curr_channels //= 2
        # 输出
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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. PHYSICS PRIOR LOSS — 频域物理一致性约束                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
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
# ║  5. SOLVER / TRAINER — 训练控制类                                     ║
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
            sim_signals = sim_signals.to(self.device)
            real_signals = real_signals.to(self.device)

            # ---------------------------
            # 1. 训练生成器
            # ---------------------------
            self.optimizer_G.zero_grad()

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
            self.optimizer_D.zero_grad()

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
                'G_Loss': f"{loss_G.item():.3f}",
                'D_Loss': f"{loss_D.item():.3f}",
                'Cycle': f"{loss_cycle.item():.3f}"
            })

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.cfg['checkpoint_dir'], f"cyclegan_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'G_S2R_state_dict': self.G_S2R.state_dict(),
        }, checkpoint_path)
        print(f"  已保存 Checkpoint -> {checkpoint_path}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. INFERENCE — 全类别泛化转换（包括无实测数据的类别）                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def generate_synthetic_dataset(trainer, config):
    """
    使用训练好的生成器，将所有仿真类别（包含 QPSK、FSK 等未参与训练的类别）
    全部转换为实测退化风格，并完整保存。
    """
    print("\n--- 开始将全部仿真信号转换（泛化）为实测风格信号 ---")
    trainer.G_S2R.eval()
    
    data = np.load(config['sim_npz_path'], allow_pickle=True)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_test, Y_test = data['X_test'], data['Y_test']
    
    def convert_batch(signals):
        converted_signals = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i+batch_size]
                # 记录原始最大值以保持功率电平
                max_vals = np.max(np.abs(batch), axis=(1, 2), keepdims=True) + 1e-8
                norm_batch = batch / max_vals

                tensor_batch = torch.from_numpy(norm_batch).float().to(trainer.device)
                fake_real_tensor = trainer.G_S2R(tensor_batch)
                
                fake_real_numpy = fake_real_tensor.cpu().numpy() * max_vals
                converted_signals.append(fake_real_numpy)
        return np.concatenate(converted_signals, axis=0)

    print("正在泛化转换 Train 集...")
    X_train_synthetic = convert_batch(X_train)
    print("正在泛化转换 Test 集...")
    X_test_synthetic = convert_batch(X_test)

    os.makedirs(config['output_synthetic_dir'], exist_ok=True)
    out_path = os.path.join(config['output_synthetic_dir'], 'Sionna_Synthetic_MultiClass.npz')
    
    np.savez_compressed(
        out_path,
        X_train=X_train_synthetic, Y_train=Y_train,
        X_test=X_test_synthetic,   Y_test=Y_test,
        snr_test=data['snr_test'],
        classes=data['classes'],
        snr_list=data['snr_list']
    )
    print(f"\n[Done] 转换完毕！混合生成数据集已保存至 -> {out_path}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. MAIN — 执行主程序                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    # 1. 初始化多类对齐加载器
    try:
        dataset = MultiClassAlignedUnpairedDataset(
            sim_npz_path=CONFIG['sim_npz_path'],
            real_dirs_map=CONFIG['real_dirs_map'],
            mat_dataset_name=CONFIG['mat_dataset_name']
        )
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    except Exception as e:
        print(f"初始化数据集失败，请检查配置。详情: {e}")
        exit()

    # 2. 创建训练器
    trainer = CycleGANTrainer1D(CONFIG)

    # 3. 联合训练
    print(f"开始联合训练多类别 1D-CycleGAN...")
    for epoch in range(CONFIG['epochs']):
        trainer.train_epoch(dataloader, epoch)
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch)

    # 4. 执行全调制格式转换
    generate_synthetic_dataset(trainer, CONFIG)