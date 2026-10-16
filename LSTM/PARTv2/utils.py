# utils.py

import matplotlib
# matplotlib.use('Tkagg') # 根据您的运行环境和需求，可以取消此行注释
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os

# --- 全局设置 ---
# 确保所有绘图和结果都保存在'figure'文件夹中
if not os.path.exists('figure'):
    os.makedirs('figure')


# --- 训练历史可视化 ---
def show_history(train_loss, val_loss, train_acc, val_acc):
    """
    接收训练和验证的loss和accuracy列表来绘图，并将历史数据保存到文件。

    Args:
        train_loss (list): 每个epoch的训练损失列表。
        val_loss (list): 每个epoch的验证损失列表。
        train_acc (list): 每个epoch的训练准确率列表。
        val_acc (list): 每个epoch的验证准确率列表。
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.title('Training and Validation Loss')
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure/loss_history.png')
    plt.close()

    plt.figure()
    plt.title('Training and Validation Accuracy')
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure/accuracy_history.png')
    plt.close()

    # 将训练历史数据保存到文本文件，方便后续分析
    print("Saving training history to text files...")
    np.savetxt('figure/train_acc.txt', np.array(train_acc))
    np.savetxt('figure/train_loss.txt', np.array(train_loss))
    np.savetxt('figure/val_acc.txt', np.array(val_acc))
    np.savetxt('figure/val_loss.txt', np.array(val_loss))


# --- 混淆矩阵和评估 ---
def calculate_confusion_matrix(Y, Y_hat, classes):
    """
    计算混淆矩阵。

    Args:
        Y (np.ndarray): 真实的 one-hot 编码标签, shape (N, num_classes)。
        Y_hat (np.ndarray): 模型的 softmax 输出预测, shape (N, num_classes)。
        classes (list): 类别名称列表。

    Returns:
        tuple: (归一化的混淆矩阵, 正确样本数, 错误样本数)
    """
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    # 将 one-hot 编码和 softmax 输出转换为类别索引
    Y_true_indices = np.argmax(Y, axis=1)
    Y_pred_indices = np.argmax(Y_hat, axis=1)

    for i in range(0, Y.shape[0]):
        true_label = Y_true_indices[i]
        pred_label = Y_pred_indices[i]
        conf[true_label, pred_label] += 1

    # 按行进行归一化 (计算每个类别的召回率)
    for i in range(0, n_classes):
        row_sum = np.sum(conf[i, :])
        if row_sum > 0:
            confnorm[i, :] = conf[i, :] / row_sum

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    """
    绘制并保存混淆矩阵热力图。

    Args:
        cm (np.ndarray): 归一化的混淆矩阵。
        title (str): 图像标题。
        cmap (matplotlib.colors.Colormap): 颜色映射。
        labels (list): 类别标签。
        save_filename (str, optional): 保存图片的文件名。
    """
    plt.figure(figsize=(10, 8))
    # 将归一化的矩阵乘以100以百分比显示
    plt.imshow(cm * 100, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    # 在格子上显示数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(np.around(cm[i, j] * 100))
            # 根据背景色选择文字颜色以提高可读性
            color = 'white' if cm[i, j] > 0.5 else 'black'
            plt.text(j, i, value, ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_filename}")
    plt.close()


def calculate_acc_at1snr_from_cm(cm):
    """从混淆矩阵中计算每个类别的准确率"""
    diag = np.diag(cm)
    row_sums = np.sum(cm, axis=1)
    # 避免除以零的警告
    return np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums != 0)


def calculate_acc_cm_each_snr(Y, Y_hat, Z, classes, save_figure=True, min_snr=-20):
    """
    按信噪比(SNR)分别计算准确率和混淆矩阵，并生成可视化图表。

    Args:
        Y (np.ndarray): 真实的 one-hot 标签。
        Y_hat (np.ndarray): 模型的 softmax 预测。
        Z (np.ndarray): 信噪比(SNR)数据, shape (N, 1)。
        classes (list): 类别列表。
        save_figure (bool): 是否保存每个SNR的混淆矩阵和准确率图。
        min_snr (int): 绘制混淆矩阵的最低SNR阈值。
    """
    Z_array = Z[:, 0]
    snrs = sorted(list(set(Z_array)))

    acc_overall_snr = np.zeros(len(snrs))
    acc_mod_snr = np.zeros((len(classes), len(snrs)))

    # 清空或创建之前的结果文件
    csv_filename = 'figure/accuracy_per_snr.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR (dB)', 'Accuracy'])

    print("\nCalculating accuracy for each SNR...")
    for i, snr in enumerate(snrs):
        # 找到当前SNR对应的所有样本索引
        indices = np.where(Z_array == snr)[0]

        if len(indices) == 0:
            continue

        # 提取当前SNR的标签和预测
        Y_snr = Y[indices]
        Y_hat_snr = Y_hat[indices]

        # 计算混淆矩阵和准确率
        cm, right, wrong = calculate_confusion_matrix(Y_snr, Y_hat_snr, classes)
        total = right + wrong
        accuracy = round(right / total, 4) if total > 0 else 0
        acc_overall_snr[i] = accuracy

        print(f'Accuracy at {int(snr):>3} dB: {accuracy * 100:6.2f}% ({int(right):>5} / {int(total):>5} samples)')

        # 记录到CSV文件
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([snr, accuracy])

        # 为高于阈值的SNR绘制并保存混淆矩阵
        if snr >= min_snr and save_figure:
            cm_filename = f'figure/confusion_matrix_snr_{int(snr)}dB.png'
            plot_confusion_matrix(cm, title=f'Confusion Matrix at {int(snr)}dB', labels=classes,
                                  save_filename=cm_filename)

        # 计算当前SNR下每个调制类型的准确率
        # 注意: calculate_acc_at1snr_from_cm需要的是原始计数的混淆矩阵，而非归一化矩阵
        # 因此我们重新计算一个非归一化的混淆矩阵
        _, non_norm_cm, _, _ = calculate_confusion_matrix(Y_snr, Y_hat_snr,
                                                          classes)  # 这里需要修改calculate_confusion_matrix来返回原始矩阵，或者在这里重算
        # 为了不修改已有函数，我们在这里重算一个原始计数的混淆矩阵
        conf_raw = np.zeros([len(classes), len(classes)])
        y_true_idx_snr = np.argmax(Y_snr, axis=1)
        y_pred_idx_snr = np.argmax(Y_hat_snr, axis=1)
        for idx in range(len(y_true_idx_snr)):
            conf_raw[y_true_idx_snr[idx], y_pred_idx_snr[idx]] += 1
        acc_mod_snr[:, i] = calculate_acc_at1snr_from_cm(conf_raw)

    # --- 绘制总体准确率 vs SNR ---
    if save_figure:
        plt.figure(figsize=(10, 7))
        plt.plot(snrs, acc_overall_snr, marker='o', linestyle='-')
        for x, y in zip(snrs, acc_overall_snr):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        plt.xlabel("Signal to Noise Ratio (dB)")
        plt.ylabel("Overall Classification Accuracy")
        plt.title("Classification Accuracy vs. SNR")
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.savefig('figure/accuracy_vs_snr_overall.png')
        plt.close()

    # --- 绘制每个调制类型的准确率 vs SNR ---
    if save_figure:
        dis_num = 6  # 每张图显示的调制类型数量
        num_plots = int(np.ceil(acc_mod_snr.shape[0] / dis_num))

        for g in range(num_plots):
            beg_index = g * dis_num
            end_index = min((g + 1) * dis_num, acc_mod_snr.shape[0])

            plt.figure(figsize=(12, 8))
            plt.xlabel("Signal to Noise Ratio (dB)")
            plt.ylabel("Classification Accuracy")
            plt.title("Per-Modulation Classification Accuracy vs. SNR")
            plt.grid(True)
            plt.ylim(0, 1.05)

            for j in range(beg_index, end_index):
                plt.plot(snrs, acc_mod_snr[j], label=classes[j], marker='.')

            plt.legend(loc='lower right')
            plt.savefig(f'figure/accuracy_vs_snr_mod_part_{g + 1}.png')
            plt.close()


# --- 用于模型分析的特定绘图函数（可选） ---
def plot_lstm2layer_output(a, modulation_type=None, save_filename=None):
    """绘制LSTM层输出"""
    plt.figure(figsize=(4, 3), dpi=600)
    plt.plot(range(a.shape[1]), a[0], label=modulation_type)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    if save_filename:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.close()


def plot_conv4layer_output(a, modulation_type=None):
    """绘制卷积层输出"""
    output_dir = './figure_conv4_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(4, 3), dpi=600)
    for i in range(100):
        plt.plot(range(a.shape[2]), a[0, 0, :, i])
        plt.xticks([])
        plt.yticks(size=20)
        save_filename = os.path.join(output_dir, f'output_{i}.png')
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.close()