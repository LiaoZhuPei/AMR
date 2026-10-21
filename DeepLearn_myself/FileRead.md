在 PyTorch 深度学习中，不同后缀的数据文件对应不同的底层存储结构、读取库以及性能考量。

下面针对 **`npy`、`pkl`、`csv`、`wav`、`dat`、`h5`** 这 6 种最常见的数据格式，按 **「核心库与读取方式」**、**「如何查看详细信息」** 以及 **「在 Dataset/DataLoader 中的标准化接入模板」** 进行详细补充。

---

### 一览速查表

| 后缀 | 常用库 | 特点 | 推荐加载策略 |
| :--- | :--- | :--- | :--- |
| **`.npy`** | `numpy` | 二进制矩阵，读取极快，支持内存映射 | 内存够大直接载入；超大文件用 `mmap_mode='r'` |
| **`.pkl`** | `pickle` / `joblib` | 任意 Python 对象序列化 | 在 `__init__` 中一次性反序列化 |
| **`.csv`** | `pandas` / `csv` | 纯文本表格，可读性好，大文件读较慢 | 在 `__init__` 读入，提取特征列和标签列 |
| **`.wav`** | `torchaudio` / `scipy.io.wavfile` / `librosa` | 音频时域信号 | 在 `__getitem__` 动态按路径读取，避免内存爆炸 |
| **`.dat`** | `numpy.fromfile` / `open` | 纯二进制流或定制文本 | 必须知道存储的 `dtype` 和 `shape`，支持 `memmap` |
| **`.h5`** | `h5py` | 分级存储，支持切片随机访问 | **必须在 `__getitem__` 延迟打开**，解决多进程锁冲突 |

---

### 1. `.npy` 文件 (NumPy 数组)
- **读取与查看信息**：
  ```python
  import numpy as np

  arr = np.load("data.npy")
  print(f"形状: {arr.shape}, 类型: {arr.dtype}, 元素总数: {arr.size}")
  
  # 若文件超大（例如几十GB），用内存映射（不占用物理内存）：
  # arr = np.load("data.npy", mmap_mode='r')
  ```
- **Dataset 接入模板**：
  ```python
  import torch
  from torch.utils.data import Dataset

  class NpyDataset(Dataset):
      def __init__(self, data_path, label_path):
          # 内存够大直接读；文件超大可加 mmap_mode='r'
          self.data = np.load(data_path)
          self.labels = np.load(label_path)

      def __len__(self):
          return len(self.labels)

      def __getitem__(self, idx):
          # 转换为 Tensor
          x = torch.from_numpy(self.data[idx]).float()
          y = torch.tensor(self.labels[idx], dtype=torch.long)
          return x, y
  ```

---

### 2. `.pkl` 文件 (Python Pickle 序列化)
- **读取与查看信息**：
  ```python
  import pickle

  with open("data.pkl", "rb") as f:
      obj = pickle.load(f)
  print(f"数据类型: {type(obj)}")
  if isinstance(obj, dict):
      print(f"包含的键: {obj.keys()}")
  ```
- **Dataset 接入模板**（常用于将字典格式的特征+标签打包保存）：
  ```python
  import torch
  import pickle
  from torch.utils.data import Dataset

  class PickleDataset(Dataset):
      def __init__(self, pkl_path):
          with open(pkl_path, "rb") as f:
              # 假设保存的是 dict: {'features': [...], 'targets': [...]}
              self.payload = pickle.load(f)
          self.features = self.payload['features']
          self.targets = self.payload['targets']

      def __len__(self):
          return len(self.targets)

      def __getitem__(self, idx):
          x = torch.tensor(self.features[idx], dtype=torch.float32)
          y = torch.tensor(self.targets[idx], dtype=torch.long)
          return x, y
  ```

---

### 3. `.csv` 文件 (逗号分隔文本)
- **读取与查看信息**：
  ```python
  import pandas as pd

  df = pd.read_csv("data.csv")
  print("前5行:\n", df.head())
  print("数据摘要:\n", df.info())
  print("数值统计:\n", df.describe())
  ```
- **Dataset 接入模板**：
  ```python
  import torch
  import pandas as pd
  from torch.utils.data import Dataset

  class CsvDataset(Dataset):
      def __init__(self, csv_path, target_col="label"):
          df = pd.read_csv(csv_path)
          
          # 处理缺失值、编码等（按需），最后转为 numpy 矩阵
          features = df.drop(columns=[target_col]).values
          labels = df[target_col].values

          self.features = torch.tensor(features, dtype=torch.float32)
          self.labels = torch.tensor(labels, dtype=torch.long)

      def __len__(self):
          return len(self.labels)

      def __getitem__(self, idx):
          return self.features[idx], self.labels[idx]
  ```

---

### 4. `.wav` 文件 (音频数据)
- **读取与查看信息**：
  推荐优先使用官方音频库 `torchaudio`（底层基于 SoX / FFmpeg）。
  ```python
  import torchaudio

  # 获取详细元信息（不把音频解码进内存）
  meta = torchaudio.info("sample.wav")
  print(f"采样率: {meta.sample_rate}, 声道数: {meta.num_channels}, 采样点数: {meta.num_frames}, 时长(秒): {meta.num_frames / meta.sample_rate:.2f}")

  # 读取音频波形为 Tensor
  waveform, sample_rate = torchaudio.load("sample.wav")
  # waveform 形状为 [声道数, 采样点数]
  print("音频张量形状:", waveform.shape)
  ```
- **Dataset 接入模板**（音频文件通常只保存路径列表，在 `__getitem__` 中按需读取）：
  ```python
  import torch
  import torchaudio
  from torch.utils.data import Dataset

  class WavAudioDataset(Dataset):
      def __init__(self, file_paths, labels, target_sample_rate=16000, fixed_length=16000 * 3):
          self.file_paths = file_paths
          self.labels = labels
          self.target_sr = target_sample_rate
          self.fixed_length = fixed_length  # 例如固定为 3 秒

      def __len__(self):
          return len(self.file_paths)

      def __getitem__(self, idx):
          # 1. 动态读取音频
          waveform, sr = torchaudio.load(self.file_paths[idx])
          
          # 2. 重采样到目标采样率（如果需要）
          if sr != self.target_sr:
              resampler = torchaudio.transforms.Resample(sr, self.target_sr)
              waveform = resampler(waveform)

          # 3. 单声道合并 / 统一形状截断或填充
          if waveform.shape[0] > 1:
              waveform = torch.mean(waveform, dim=0, keepdim=True)
              
          if waveform.shape[1] < self.fixed_length:
              pad_size = self.fixed_length - waveform.shape[1]
              waveform = torch.nn.functional.pad(waveform, (0, pad_size))
          else:
              waveform = waveform[:, :self.fixed_length]

          label = torch.tensor(self.labels[idx], dtype=torch.long)
          return waveform, label
  ```

---

### 5. `.dat` 文件 (裸二进制 / 自定义格式数据)
- **读取与查看信息**：
  `.dat` 常见于仿真、传感器导出的无头（Raw binary）数据。读取前**必须明确数据类型（如 float32、int16）和形状（shape）**。
  ```python
  import os
  import numpy as np

  file_size = os.path.getsize("data.dat")
  print(f"文件大小: {file_size} 字节")

  # 假设已知按 float32 (每个占 4 字节) 存储
  data = np.fromfile("data.dat", dtype=np.float32)
  print(f"解析出的总数据量: {len(data)}")
  ```
- **Dataset 接入模板**（推荐用 `np.memmap` 实现按需磁盘读取）：
  ```python
  import torch
  import numpy as np
  from torch.utils.data import Dataset

  class DatDataset(Dataset):
      def __init__(self, dat_path, num_samples, feature_dim, dtype=np.float32):
          # 内存映射方式读取，不把大二进制文件直接塞进内存
          self.data = np.memmap(
              dat_path, 
              dtype=dtype, 
              mode='r', 
              shape=(num_samples, feature_dim)
          )

      def __len__(self):
          return len(self.data)

      def __getitem__(self, idx):
          # 按索引切片读取
          sample = np.array(self.data[idx])  # 读入内存
          return torch.from_numpy(sample).float()
  ```

---

### 6. `.h5` / `.hdf5` 文件 (层次化数据格式)
- **读取与查看信息**：
  `.h5` 类似于一个包含树状文件夹的文件系统，最适合存放几百 GB 的复杂多维数据（图像集合、医学切片、物理场数据）。
  ```python
  import h5py

  with h5py.File("data.h5", "r") as f:
      # 查看根目录下的 keys（组或数据集名称）
      print("Keys:", list(f.keys()))
      # 获取特定 dataset 的元信息
      dset = f["features"]
      print(f"形状: {dset.shape}, 类型: {dset.dtype}")
  ```
- **Dataset 接入模板（极度关键坑点！）**：
  > ⚠️ **避坑说明**：`h5py.File` 对象**不能跨多进程共享**。如果将 `h5py.File` 实例放在 `__init__` 中，当 `DataLoader(num_workers > 0)` 启动子进程时，会导致死锁或抛出 HDF5 错误。  
  > **正确做法**：在 `__getitem__` 或 worker 初始化时动态打开文件句柄（懒加载 Lazy Loading）。

  ```python
  import torch
  import h5py
  from torch.utils.data import Dataset, DataLoader

  class H5Dataset(Dataset):
      def __init__(self, h5_path):
          self.h5_path = h5_path
          # 初始化时仅获取长度，不持久持有文件句柄
          with h5py.File(self.h5_path, 'r') as f:
              self.length = len(f['labels'])
          self.file = None  # 句柄占位

      def __len__(self):
          return self.length

      def __getitem__(self, idx):
          # 懒加载：每个 DataLoader 的 worker 子进程只在需要时打开一次自己的句柄
          if self.file is None:
              self.file = h5py.File(self.h5_path, 'r')

          # 支持高速切片读取
          x = torch.from_numpy(self.file['features'][idx]).float()
          y = torch.tensor(self.file['labels'][idx], dtype=torch.long)
          return x, y

  # 测试多进程读取安全
  if __name__ == '__main__':
      # dataset = H5Dataset("data.h5")
      # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
      pass
  ```

---

### 总结设计原则

1. **小文件（小于可用内存）**：在 `__init__` 中直接读取并转为内存变量，最大化训练吞吐速度（如普通 `csv`, 几百 MB 的 `npy`, `pkl`）。
2. **大文件（几十上百 GB）**：
   - 连续单文件：用 `np.memmap` 或 `h5py` 按需在 `__getitem__` 中切片加载。
   - 散文件（如百万张 `wav` 或图片）：`__init__` 中只保存文件路径列表，在 `__getitem__` 里动态通过路径载入单个文件。