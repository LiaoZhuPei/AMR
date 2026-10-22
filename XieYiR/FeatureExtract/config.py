# config.py
import numpy as np

class Config:
    # 数据路径
    DATA_PATH = "dataset_panoradio_hf.npy"
    TAGS_PATH = "dataset_panoradio_hf_tags.csv"
    
    # 信号参数
    FS = 6000          # 采样率 Hz
    N_SAMPLES = 2048   # IQ采样点数
    
    # 目标类别（Link11代替品）
    TARGET_MODE = "mt63_1000"
    
    # SNR设置
    ALL_SNR = [25, 20, 15, 10, 5, 0, -5, -10]
    TRAIN_SNR = [25, 20, 15, 10, 5]   # 训练时使用的SNR
    TEST_SNR  = [25, 20, 15, 10, 5, 0, -5, -10]  # 测试全SNR
    
    # 特征提取参数
    N_FFT = 512
    HOP_LENGTH = 128
    N_MFCC = 20
    
    # 模型参数
    OCSVM_NU = 0.01
    OCSVM_KERNEL = "rbf"
    OCSVM_GAMMA = "scale"
    
    IF_CONTAMINATION = 0.05
    IF_N_ESTIMATORS = 200
    
    # 开放集阈值
    THRESHOLD_PERCENTILE = 5   # 训练集得分的5th percentile作为阈值
    
    # 随机种子
    SEED = 42
    
    # 输出路径
    MODEL_SAVE_PATH = "saved_models/"
    RESULT_SAVE_PATH = "results/"