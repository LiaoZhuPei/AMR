import os
import glob
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 纯净原始 STFT 数据集读取 (无任何有损变换)
# =========================================================================
class STFTSplitDataset(Dataset):
    """
    原始双通道 IQ 时频矩阵读取类
    输出张量维度: [2, 2048, 2055]
    通道 0: Real (实部 I)
    通道 1: Imag (虚部 Q)
    """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(h5_path, 'r') as f:
            # 读取原始实虚部并转置对齐 [2048, 2055]
            real = f['real'][:].T
            imag = f['imag'][:].T
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# =========================================================================
# 2. 严格文件级数据集划分 (杜绝任何跨切片数据泄露)
# =========================================================================
def split_datasets_by_files(train_dir, known_classes=['A', 'D', 'E', 'G'], unknown_classes=['B', 'C', 'F']):
    all_files = glob.glob(os.path.join(train_dir, "*.h5"))
    if len(all_files) == 0:
        raise ValueError(f"未在目录 {train_dir} 中检索到 .h5 文件，请检查路径！")

    class_to_files = {}
    for f in all_files:
        c = os.path.basename(f)[0].upper()
        class_to_files.setdefault(c, []).append(f)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_known_files, test_known_labels = [], []
    test_unknown_files, test_unknown_labels = [], []

    known_map = {c: i for i, c in enumerate(sorted(known_classes))}

    print("\n" + "="*70)
    print("                 数据集文件级划分概览 (DATASET PARTITION)")
    print("="*70)

    for c in known_classes:
        files = class_to_files.get(c, [])
        random.shuffle(files)
        n = len(files)
        if n == 0:
            raise ValueError(f"已知类 [{c}] 样本数为 0，请检查文件名！")

        n_train = int(n * 0.8)
        n_val = max(1, int(round(n * 0.1)))
        n_test = n - n_train - n_val

        c_train = files[:n_train]
        c_val = files[n_train:n_train + n_val]
        c_test = files[n_train + n_val:]

        train_files.extend(c_train)
        train_labels.extend([known_map[c]] * len(c_train))
        val_files.extend(c_val)
        val_labels.extend([known_map[c]] * len(c_val))
        test_known_files.extend(c_test)
        test_known_labels.extend([known_map[c]] * len(c_test))

        print(f"已知类 (Known)   [{c}] -> 总数: {n:<4} | 训练集: {len(c_train):<4} | 验证集: {len(c_val):<3} | 测试集: {len(c_test):<3}")

    for c in unknown_classes:
        files = class_to_files.get(c, [])
        test_unknown_files.extend(files)
        test_unknown_labels.extend([c] * len(files))
        print(f"未知类 (Unknown) [{c}] -> 总数: {len(files):<4} (100% 投入测试评估)")

    print("-" * 70)
    print(f"汇总统计: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试已知={len(test_known_files)}, 测试未知={len(test_unknown_files)}")
    print("="*70 + "\n")

    return (train_files, train_labels), (val_files, val_labels), \
           (test_known_files, test_known_labels), (test_unknown_files, test_unknown_labels), known_map

# =========================================================================
# 3. 实验历程与失效机理系统性复盘总结 (自动输出)
# =========================================================================
def print_experiment_summary():
    summary_text = """
========================================================================================
           无人机射频开放集识别 (UAV-OSR) 历次探索方案与失效机理系统性总结报告
========================================================================================

【核心问题背景定义】
* 已知类别: A, D, E, G（参与训练与验证）
* 未知类别: B（背景噪声）, C（高频跳频无人机）, F（低频跳频无人机）
* 核心物理矛盾:
  1. 所有无人机在 2.46 ~ 2.47 GHz 均存在强连续图传信号（公共大带宽底色）。
  2. A, D, G 与未知类 F 的跳频集中在 < 2.46 GHz。
  3. 整个训练集中，有且仅有 E 在 > 2.47 GHz 存在大能量；而未知类 C 恰好也在 > 2.47 GHz。
  4. 现象: B 与 F 极易拒识，但 C 极易被误判为 E（高频垄断捷径导致特征坍缩）。

----------------------------------------------------------------------------------------
【第一维度：输入表征与信号变换的探索与失效机理】
----------------------------------------------------------------------------------------
1. 单通道能量谱（I^2 + Q^2 / 对数功率谱）：
   - 做法：将双通道 IQ 压缩为标量能量。
   - 结果：显存降低，但分类与拒识性能无显著提升。
   - 失效机理：丢失了正交相位信息；纯能量矩阵依然保留了高频频段的绝对能量占满，没有打破
     “只要高频亮就是 E”的网络捷径依赖。

2. 循环谱密度（CSD / 谱相关函数 SCF）：
   - 做法：试图利用循环平稳理论，将信号映射到 [谱频率 f, 循环频率 alpha] 空间。
   - 结果：拒识率依然极低，计算开销大。
   - 失效机理：无人机跳频脉冲是非同步的、短驻留的非协作信号。快速分帧平均算法（X * X'）
     在非同步条件下，非对角线上的循环相位因子全部随机相消，退化为普通的功率谱，未能提取
     出码元速率谱线。

3. STFT 参数调整（nfft=1024, noverlap=0.5 -> [1024, 7811]）：
   - 做法：试图通过减小窗口、提高重叠率来增加时间分辨率。
   - 结果：不仅 C 的拒识没有改善，原本表现极佳的 F 和 C 拒识率同时出现断崖式下跌。
   - 失效机理：
     a) nfft 从 2048 降为 1024，频域分辨率从 39 kHz 粗化为 78 kHz，导致 A, D, G 细微的
        子信道被模糊，连带击穿了原本对 F 的低频防御。
     b) 50% 汉宁窗大重叠引入了严重的时间过渡带伪影，平滑了跳频脉冲的上升沿瞬态。
   - 结论：原始参数（nfft=2048, noverlap=0.05, 尺寸 [2, 2048, 2055]）是最佳物理观测基准。

----------------------------------------------------------------------------------------
【第二维度：网络架构的演进与失效机理】
----------------------------------------------------------------------------------------
1. 时频分块 Transformer（AST / Spectrogram ViT）：
   - 做法：纯 ViT 结构，将大图打碎为 32x32 的二维网格 Patch。
   - 结果：严重过拟合（训练集 Acc 以 10% 狂飙，验证集仅以 1%~6% 缓慢爬升），收敛极慢。
   - 失效机理：
     a) 缺乏 CNN 的局部感应偏置（Inductive Bias），在从零训练（From Scratch）时，
        Transformer 无法理解频域相邻与时间相继的物理规则。
     b) 输入序列长达 4160 个 Token，自注意力矩阵极其庞大，小 Batch 下梯度噪声剧烈。
     c) 即使加入 SpecAugment 和 DropPath 缓解了过拟合，对高频 C 的拒识依然没有突破。

2. 借鉴现代目标检测骨干（YOLOv8/YOLO11: C2f + SPPF）：
   - 做法：引入 YOLO 的跨阶段局部残差与快速空间金字塔池化（SPPF）。
   - 结果：C 类别的拒识率直接腰斩，性能严重倒退。
   - 失效机理：
     a) YOLO 的 SPPF 包含 5x5, 9x9, 13x13 的超大 MaxPool，专为宏观刚性目标设计。
     b) 射频跳频脉冲是微秒级断续脉冲，大核池化直接把离散跳频之间的空白间隙强行填满，
        把断续脉冲融合成了一根实心粗线，彻底摧毁了 C 与 E 的脉冲驻留时间差异。

3. 频域过度下采样陷阱（经典 ResNet 4级降采样的致命伤）：
   - 做法：标准 ResNet 对时间与频率同等进行 128 倍下采样（2048 -> 16）。
   - 发现的致命数学硬伤：
     总带宽 80 MHz，高频段（> 2.47 GHz）仅占 10 MHz（12.5%）。在最终 16 行的特征图上，
     整个高频段仅占 2 个像素点（16 * 12.5% = 2）！
     网络把所有时间平均后，高频段信息退化成 2 个标量值。C 与 E 细微的 2 MHz 信道差异
     被数学强制四舍五入抹杀，输入决策层时两者向量完全重合。

----------------------------------------------------------------------------------------
【第三维度：损失函数与决策边界建模的探索与失效机理】
----------------------------------------------------------------------------------------
1. 对抗互易点（ARPLayer / ARPLoss）：
   - 做法：为每个已知类设立互易点，并采用相对 Margin（(d2 - d1) / d2）判定未知。
   - 结果：F 表现良好，但 C 几乎 100% 误判进 E。
   - 失效机理（单邻居开放集盲区）：
     低频有 A, D, G 三强竞争，F 进来时 d1≈d2，Margin 趋近 0 成功拒识；但高频只有 E 独占，
     C 进来时距 A, D, G 极远（d2 极大），导致即使 C 离 E 不近，其相对 Margin 也接近满分，
     Softmax 给出 99% 的错误高置信度。

2. 伪未知类暴露（Time-Scrambled POE）：
   - 做法：训练时动态将 E 的时间轴切片打乱，保留频段、粉碎时序，作为未知类排斥。
   - 结果：C 没有改善，F 的拒识率反而暴跌近一半。
   - 失效机理：切片重排引入了非自然的拼接毛刺，网络学会了“有毛刺的是未知”，而真实的 C 
     是平滑的真实信号，依然无法防范；同时破坏了特征流形的平滑性，冲击了低频 A, D, G 边界。

3. ArcFace 角度超锥加严（增大角度惩罚 m 从 0.50 到 0.65）：
   - 做法：期望通过更严苛的向心力迫使 E 聚拢。
   - 结果：性能不升反降。
   - 失效机理（过犹不及的向心坍缩）：
     在单类垄断的高频段，过度的角度压缩逼迫网络放弃微观时序特征，退化为一个极端的
     “高频能量开关”，反而加剧了与 C 的混淆。

----------------------------------------------------------------------------------------
【第四维度：当前已被验证的最佳性能基线（有效配置）】
----------------------------------------------------------------------------------------
* 输入特征：纯净 STFT [2, 2048, 2055]（nfft=2048, noverlap=0.05）。
* 数据增强：训练期 50% 概率垂直频率镜像翻转（打破高频 E 的绝对垄断，迫使提取脉冲指纹）。
* 骨干架构：频域保真非对称 ResNet（FPA-ResNet，Stage 2~4 锁住 128 个频带，不压缩频率）。
* 损失模型：ArcFace 角度空间建模（尺度 s=32.0, 角度 margin m=0.50）。
* 判定准则：稳健高斯尾部门限（tau_k = mean_k - 1.0 * std_k，废除小样本极值分位点）。
* 达成的最佳测试性能：
  - 未知类 B (背景噪声) 拒识率: 100.00% (37/37)
  - 未知类 F (低频无人机) 拒识率: 92.76% (282/304)
  - 未知类 C (高频冒充者) 拒识率: 64.72% (244/377，从最初的接近 0% 取得实质突破)
  - 整体未知类综合拒识率: 78.41%
  - 已知类闭集 Top-1 准确率: 91.84%
========================================================================================
"""
    print(summary_text)

# =========================================================================
# 4. 主程序：验证数据集读取流程并输出技术总结
# =========================================================================
def main():
    # 填入你原始提取的 STFT H5 目录
    TRAIN_DIR = "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms"

    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    KNOWN_CLASSES = ['A', 'D', 'E', 'G']
    UNKNOWN_CLASSES = ['B', 'C', 'F']

    # 1. 执行文件级划分
    tr_data, val_data, te_known, te_unknown, known_map = split_datasets_by_files(
        TRAIN_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES
    )

    # 2. 实例化纯净 STFT 数据集对象
    train_ds  = STFTSplitDataset(tr_data[0], tr_data[1])
    val_ds    = STFTSplitDataset(val_data[0], val_data[1])
    test_k_ds = STFTSplitDataset(te_known[0], te_known[1])
    test_u_ds = STFTSplitDataset(te_unknown[0], te_unknown[1])

    # 3. 构造 DataLoader 并抽取一个批次进行数据健康度检查
    check_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    
    print("\n" + "="*50)
    print("      正在验证原始时频图数据读取管道...")
    print("="*50)
    for sample_x, sample_y, sample_name in check_loader:
        print(f"-> 成功加载一个 Batch:")
        print(f"   * 张量形状 (Tensor Shape): {sample_x.shape} (预期: [4, 2, 2048, 2055])")
        print(f"   * 数据类型 (Data Type):    {sample_x.dtype}")
        print(f"   * 样本标签 (Labels):       {sample_y.tolist()}")
        print(f"   * 文件样本名 (Filenames):  {sample_name[:2]} ...")
        print(f"   * 数值统计: Min={sample_x.min():.4f}, Max={sample_x.max():.4f}, Mean={sample_x.mean():.4f}")
        break
    print("="*50 + "\n")

    # 4. 打印完整技术复盘报告
    print_experiment_summary()

if __name__ == "__main__":
    main()