import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.spatial.distance import cdist, pdist

from models import STFTH5Dataset, FullFeatureExtractor

# 设置绘图风格 (无中文警告)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ==========================================
# 1. 特征提取函数
# ==========================================
@torch.no_grad()
def extract_all_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    print("-> Extracting Sequence Embeddings...")
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        feats = model(x)
        embeddings.append(feats.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

# ==========================================
# 2. 统计与特征空间量化指标分析
# ==========================================
def perform_feature_analysis(embeddings, labels, class_names):
    num_classes = len(class_names)
    stats = {}

    # (1) 样本分布与长尾分析 (Long-tail Check)
    class_counts = {class_names[i]: np.sum(labels == i) for i in range(num_classes)}
    counts = np.array(list(class_counts.values()))
    imbalance_ratio = np.max(counts) / (np.min(counts) + 1e-6)
    stats['counts'] = class_counts
    stats['imbalance_ratio'] = imbalance_ratio

    # (2) 类内距离与类中心计算 (Intra-class Distance & Centroids)
    centroids = []
    intra_distances = {}
    cov_traces = {}
    for c in range(num_classes):
        c_feats = embeddings[labels == c]
        if len(c_feats) > 1:
            # 组内样本点两两欧氏距离平均值
            intra_d = np.mean(pdist(c_feats, metric='euclidean'))
            # 协方差迹 (Trace of Covariance) 衡量类内离散方差
            cov_matrix = np.cov(c_feats, rowvar=False)
            cov_trace = np.trace(cov_matrix)
        else:
            intra_d, cov_trace = 0.0, 0.0
        
        intra_distances[class_names[c]] = intra_d
        cov_traces[class_names[c]] = cov_trace
        centroids.append(np.mean(c_feats, axis=0))

    centroids = np.array(centroids)
    stats['intra_distances'] = intra_distances
    stats['cov_traces'] = cov_traces

    # (3) 类间距离矩阵 (Inter-class Distance Between Centroids)
    inter_dist_matrix = cdist(centroids, centroids, metric='euclidean')
    stats['inter_dist_matrix'] = inter_dist_matrix

    # (4) 聚类质量与类别重叠分析 (Clustering & Overlap Diagnosis)
    if num_classes > 1 and len(embeddings) > num_classes:
        sil_score = silhouette_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        sample_sils = silhouette_samples(embeddings, labels)
        # 负 Silhouette 系数代表该样本更靠近其他类，属于重叠/错分样本
        overlap_rate = np.mean(sample_sils < 0.0)
    else:
        sil_score, db_score, sample_sils, overlap_rate = 0.0, 0.0, np.zeros(len(labels)), 0.0

    stats['silhouette_score'] = sil_score
    stats['davies_bouldin_score'] = db_score
    stats['sample_silhouettes'] = sample_sils
    stats['overlap_rate'] = overlap_rate

    # 控制台打印统计报告
    print("\n" + "="*50)
    print("      FEATURE SPACE ANALYSIS REPORT")
    print("="*50)
    print(f"Total Samples: {len(labels)}, Num Classes: {num_classes}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    print(f"Long-tail Warning: {'YES (Severe Imbalance)' if imbalance_ratio > 3.0 else 'NO (Balanced)'}")
    print(f"Global Silhouette Score (Clustering Quality): {sil_score:.4f} (Higher is better, [-1, 1])")
    print(f"Davies-Bouldin Score: {db_score:.4f} (Lower is better)")
    print(f"Estimated Overlap Ratio (Negative Silhouettes): {overlap_rate * 100:.2f}%")
    print("-" * 50)
    for c_name in class_names:
        print(f"Class [{c_name}] -> Samples: {class_counts[c_name]:<4} | "
              f"Intra-Dist: {intra_distances[c_name]:.4f} | "
              f"Cov Trace: {cov_traces[c_name]:.4f}")
    print("="*50 + "\n")

    return stats

# ==========================================
# 3. 嵌入空间可视化
# ==========================================
def plot_analysis_dashboard(embeddings, labels, class_names, stats, save_path="feature_space_analysis.png"):
    fig = plt.figure(figsize=(20, 12))
    cmap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")
    colors = [cmap(i) for i in range(len(class_names))]

    # 1. 样本分布柱状图 (检查长尾)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(class_names, list(stats['counts'].values()), color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("Class Sample Counts (Long-tail Diagnostic)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")

    # 2. 类间距离热力图 (Inter-class Distance Matrix)
    ax2 = fig.add_subplot(2, 3, 2)
    sns.heatmap(stats['inter_dist_matrix'], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax2, cbar=False)
    ax2.set_title("Inter-Class Centroid Distance", fontsize=13, fontweight='bold')

    # 3. 类内方差/协方差迹 (Covariance Trace & Intra-dist)
    ax3 = fig.add_subplot(2, 3, 3)
    x_idx = np.arange(len(class_names))
    ax3.bar(x_idx - 0.2, list(stats['intra_distances'].values()), width=0.4, label='Intra-Dist', color='teal', alpha=0.85)
    ax3.bar(x_idx + 0.2, list(stats['cov_traces'].values()), width=0.4, label='Cov Trace', color='orange', alpha=0.85)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(class_names)
    ax3.set_title("Intra-Class Spread & Covariance Trace", fontsize=13, fontweight='bold')
    ax3.legend()

    # 4. Silhouette 分布 (重叠度诊断: <0 的样本属于重叠)
    ax4 = fig.add_subplot(2, 3, 4)
    sns.boxplot(x=[class_names[l] for l in labels], y=stats['sample_silhouettes'], ax=ax4, palette=colors)
    ax4.axhline(0.0, color='red', linestyle='--', linewidth=1.2, label='Overlap Threshold')
    ax4.set_title("Class Silhouette Coefficients (Overlap Check)", fontsize=13, fontweight='bold')
    ax4.set_ylabel("Silhouette Score")
    ax4.legend(loc='lower right')

    # 5. PCA 2D 降维可视化
    ax5 = fig.add_subplot(2, 3, 5)
    pca = PCA(n_components=2)
    embeds_pca = pca.fit_transform(embeddings)
    for i, c_name in enumerate(class_names):
        mask = (labels == i)
        ax5.scatter(embeds_pca[mask, 0], embeds_pca[mask, 1], label=c_name, color=colors[i], alpha=0.8, edgecolors='none', s=45)
    ax5.set_title(f"PCA 2D Projection (EVR: {np.sum(pca.explained_variance_ratio_):.2f})", fontsize=13, fontweight='bold')
    ax5.set_xlabel("PC 1")
    ax5.set_ylabel("PC 2")
    ax5.legend()

    # 6. t-SNE 2D 流形降维可视化
    ax6 = fig.add_subplot(2, 3, 6)
    perplexity = min(30, max(5, len(labels) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    embeds_tsne = tsne.fit_transform(embeddings)
    for i, c_name in enumerate(class_names):
        mask = (labels == i)
        ax6.scatter(embeds_tsne[mask, 0], embeds_tsne[mask, 1], label=c_name, color=colors[i], alpha=0.85, edgecolors='none', s=45)
    ax6.set_title("t-SNE 2D Manifold Space", fontsize=13, fontweight='bold')
    ax6.set_xlabel("t-SNE Dim 1")
    ax6.set_ylabel("t-SNE Dim 2")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Feature visualization dashboard saved to: {save_path}")
    plt.show()

# ==========================================
# 4. 主执行入口
# ==========================================
def main():
    # 路径设置：请根据实际放置修改
    TRAIN_DIR = "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4  # 一个 Batch: 4 * 127 个 blocks，根据显存调节

    print(f"Using Device: {DEVICE}")

    # 1. 准备数据集
    dataset = STFTH5Dataset(data_dir=TRAIN_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Loaded {len(dataset)} files spanning {len(dataset.classes)} classes: {dataset.classes}")

    # 2. 构建模型 (CNN + Temporal Transformer)
    model = FullFeatureExtractor(win_size=32, stride=16, d_model=512)
    
    # 若有预训练权重，可在此加载:
    # model.load_state_dict(torch.load("pretrained_extractor.pth", map_location=DEVICE))
    model = model.to(DEVICE)

    # 3. 提取特征
    embeddings, labels = extract_all_embeddings(model, dataloader, DEVICE)

    # 4. 统计与量化指标评估
    stats = perform_feature_analysis(embeddings, labels, dataset.classes)

    # 5. 空间可视化
    plot_analysis_dashboard(embeddings, labels, dataset.classes, stats, save_path="embedding_space_analysis.png")

if __name__ == "__main__":
    main()