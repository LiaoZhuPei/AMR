# main.py
import numpy as np
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data.load_data import DataLoader
from features.extractor import FeatureExtractor
from models.one_class_detector import OneClassDetector
from evaluation.metrics import OpenSetEvaluator

def main():
    print("="*60)
    print("    Link11 开放集协议识别系统")
    print("    (使用 mt63_1000 作为 Link11 代替)")
    print("="*60)
    
    cfg = Config()
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(cfg.RESULT_SAVE_PATH, exist_ok=True)
    
    # ------------------------------------------------------------------ #
    # Step 1: 加载数据
    # ------------------------------------------------------------------ #
    loader = DataLoader(cfg)
    loader.load()
    
    X_train_raw, y_train, X_test_raw, y_test, meta_modes = \
        loader.get_split(
            train_snr=cfg.TRAIN_SNR,
            test_snr=cfg.TEST_SNR
        )
    
    # 获取测试集SNR信息（用于分析）
    # 需要从tags中获取对应SNR
    tags = loader.tags
    pos_mask = tags['mode'] == cfg.TARGET_MODE
    pos_mask &= tags['snr'].isin(cfg.TEST_SNR)
    neg_mask = tags['mode'] != cfg.TARGET_MODE
    neg_mask &= tags['snr'].isin(cfg.TEST_SNR)
    
    test_snr_pos = tags[pos_mask]['snr'].tolist()
    test_snr_neg = tags[neg_mask]['snr'].tolist()
    meta_snr = test_snr_pos + test_snr_neg
    
    # ------------------------------------------------------------------ #
    # Step 2: 特征提取
    # ------------------------------------------------------------------ #
    extractor = FeatureExtractor(cfg)
    
    print("\n[Step 2] 提取训练集特征...")
    t0 = time.time()
    X_train_feat = extractor.extract_batch(X_train_raw, verbose=True)
    print(f"  完成! 耗时 {time.time()-t0:.1f}s, "
          f"特征维度: {X_train_feat.shape}")
    
    print("\n[Step 2] 提取测试集特征...")
    t0 = time.time()
    X_test_feat = extractor.extract_batch(X_test_raw, verbose=True)
    print(f"  完成! 耗时 {time.time()-t0:.1f}s")
    
    # 保存特征（避免重复提取）
    np.save(os.path.join(cfg.RESULT_SAVE_PATH, 'X_train_feat.npy'), 
            X_train_feat)
    np.save(os.path.join(cfg.RESULT_SAVE_PATH, 'X_test_feat.npy'), 
            X_test_feat)
    np.save(os.path.join(cfg.RESULT_SAVE_PATH, 'y_test.npy'), y_test)
    print("[INFO] 特征已保存")
    
    # ------------------------------------------------------------------ #
    # Step 3: 训练检测器
    # ------------------------------------------------------------------ #
    print("\n[Step 3] 训练开放集检测器...")
    detector = OneClassDetector(cfg)
    detector.fit(X_train_feat)
    detector.save(cfg.MODEL_SAVE_PATH)
    
    # ------------------------------------------------------------------ #
    # Step 4: 评估
    # ------------------------------------------------------------------ #
    print("\n[Step 4] 评估检测性能...")
    
    scores_dict = detector.score(X_test_feat)
    fusion_scores = scores_dict['fusion']
    
    evaluator = OpenSetEvaluator()
    results = evaluator.evaluate(
        y_true=y_test,
        scores=fusion_scores,
        meta_modes=meta_modes,
        meta_snr=meta_snr,
        plot=True
    )
    
    # 各方法对比
    print("\n[Step 4] 各模型对比:")
    for method in ["ocsvm", "iforest", "fusion"]:
        preds = detector.predict(X_test_feat, method=method)
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == -1))
        fn = np.sum((preds == -1) & (y_test == 1))
        tn = np.sum((preds == -1) & (y_test == -1))
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        print(f"  {method:10s}: TPR={tpr:.4f}, FPR={fpr:.4f}, "
              f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # ------------------------------------------------------------------ #
    # Step 5: 推理示例
    # ------------------------------------------------------------------ #
    print("\n[Step 5] 推理示例 (前5个测试样本):")
    sample_signals = X_test_raw[:5]
    sample_feats   = extractor.extract_batch(sample_signals, verbose=False)
    sample_scores  = detector.predict_proba(sample_feats)
    sample_preds   = detector.predict(sample_feats, method='fusion')
    
    for i in range(5):
        label = "Link11 ✓" if sample_preds[i] == 1 else "Unknown ✗"
        true  = "Link11" if y_test[i] == 1 else "Unknown"
        print(f"  样本{i}: 得分={sample_scores[i]:.4f}, "
              f"预测={label}, 真实={true}, 类型={meta_modes[i]}")
    
    print("\n[DONE] 全部完成！")
    return results, detector

# ------------------------------------------------------------------ #
# 推理接口（实际部署使用）
# ------------------------------------------------------------------ #
def inference(iq_signal: np.ndarray, 
              model_path: str = "saved_models/",
              cfg: Config = None) -> dict:
    """
    对单个陌生信号进行Link11检测
    
    输入:
        iq_signal: (2048,) complex, 功率归一化到1
    输出:
        {
          "is_link11": bool,
          "confidence": float (0~1),
          "scores": dict
        }
    """
    cfg = cfg or Config()
    
    # 加载模型
    detector  = OneClassDetector.load(model_path)
    extractor = FeatureExtractor(cfg)
    
    # 特征提取
    feat = extractor.extract(iq_signal).reshape(1, -1)
    
    # 检测
    score  = detector.predict_proba(feat)[0]
    pred   = detector.predict(feat, method='fusion')[0]
    scores = detector.score(feat)
    
    return {
        "is_link11":  bool(pred == 1),
        "confidence": float(score),
        "scores": {
            "ocsvm":   float(scores['ocsvm_norm'][0]),
            "iforest": float(scores['iforest_norm'][0]),
            "fusion":  float(scores['fusion'][0])
        }
    }

if __name__ == "__main__":
    main()