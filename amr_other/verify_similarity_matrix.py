import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# ================= 依赖 =================
# 假设你已经有了 data_generator_tensor_vsg_ultimate.py
from data_generator_tensor_vsg_filtered import LabDeviceSimulator, generate_tensor_data

# ================= 配置 =================
REAL_FILE = "E:/Project/AMR-Benchmark/QPSK-03-24-25-10h53m20s46.iq"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAMPLES_PER_ANCHOR = 50 # 每个锚点生成多少个样本

def main():
    print("=== Generating Similarity Matrix ===")
    sim = LabDeviceSimulator(carrier_freq=2.4e9)
    
    features = []
    labels = []
    
    # -------------------------------------------------
    # 1. 生成 4 个象限的锚点数据 (Anchors)
    # -------------------------------------------------
    anchors = [
        # Name,       SNR, PPM, Color
        ("Ideal",     25,  1,   "green"),  # 高信噪比, 低频偏
        ("Noisy",     5,   1,   "orange"), # 低信噪比, 低频偏
        ("Drift",     25,  20,  "purple"), # 高信噪比, 高频偏 (重点关注!)
        ("Worst",     5,   20,  "red")     # 低信噪比, 高频偏
    ]
    
    for name, snr, ppm, color in anchors:
        print(f"Generating Anchor: {name} (SNR={snr}, PPM={ppm})...")
        for i in range(SAMPLES_PER_ANCHOR):
            # 固定随机种子确保这50个样本内部具有多样性但统计特征一致
            # i+seed_offset 防止不同锚点生成的随机数序列完全重叠
            sig = sim.generate_packet(
                   mod_type='QPSK', 
                   num_samples=10240, # 注意长度要一致
                   ppm_error=ppm, 
                   iq_gain_db=0,      # 默认为0
                   iq_phase_deg=0,    # 默认为0
                   snr_db=snr)
            
            # 生成 RAD 张量
            tensor = generate_tensor_data(torch.from_numpy(sig).to(DEVICE), 64)
            
            # 展平为一维向量 (3*64*64 = 12288)
            # 直接使用像素特征做 t-SNE，这最能反映"图长得像不像"
            features.append(tensor.cpu().numpy().flatten())
            labels.append(name)

    # -------------------------------------------------
    # 2. 读取实测数据 (Target)
    # -------------------------------------------------
    print(f"Loading Real Data: {REAL_FILE.split('/')[-1]}...")
    if REAL_FILE.endswith('.dat'):
        raw = np.fromfile(REAL_FILE, dtype='>i2')
        if len(raw) > 1024: raw = raw[1024:]
    else:
        raw = np.fromfile(REAL_FILE, dtype=np.int16)
        
    if len(raw) % 2 != 0: raw = raw[:-1]
    
    I = raw[::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    iq_complex = torch.complex(torch.from_numpy(I), torch.from_numpy(Q))
    iq_complex = iq_complex / 32767.0
    iq_complex = iq_complex.to(DEVICE)
    
    # 切片取前 50 个
    for i in range(min(50, len(iq_complex)//10240)):
        chunk = iq_complex[i*10240 : (i+1)*10240]
        tensor = generate_tensor_data(chunk, 64)
        features.append(tensor.cpu().numpy().flatten())
        labels.append("Real_Data")

    # -------------------------------------------------
    # 3. t-SNE 降维与绘图
    # -------------------------------------------------
    print("Running t-SNE (this may take a moment)...")
    features = np.array(features)
    # perplexity 设为 30 比较适合几百个样本
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    
    # 绘制锚点
    for name, _, _, color in anchors:
        idx = [i for i, x in enumerate(labels) if x == name]
        plt.scatter(embedded[idx, 0], embedded[idx, 1], c=color, label=name, alpha=0.6, s=50)
        
    # 绘制实测数据 (大大的蓝色五角星)
    idx = [i for i, x in enumerate(labels) if x == "Real_Data"]
    plt.scatter(embedded[idx, 0], embedded[idx, 1], c='blue', marker='*', s=150, label='Real Data', edgecolors='black')
    
    plt.title("Signal Similarity Analysis (t-SNE)\nDetermine the dominant impairment factor")
    plt.legend(loc='best', fontsize='large')
    plt.grid(True, alpha=0.3)
    
    # 添加辅助说明文本
    plt.figtext(0.15, 0.02, "Cluster Meaning:\n"
                           "Ideal (Green): Perfect Signal\n"
                           "Noisy (Orange): Low SNR only\n"
                           "Drift (Purple): High CFO only\n"
                           "Worst (Red): Both Bad", fontsize=10, bbox={"facecolor":"white", "alpha":0.5})
    
    save_path = "similarity_matrix_tsne.png"
    plt.savefig(save_path)
    print(f"Done! Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()