# 文件名: CNNtrain/mltools.py

# "Adapted from the code (https://github.com/leena201818/radiom) contributed by leena201818"
import matplotlib
# matplotlib.use('Tkagg') # 根据需要取消注释
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import os

# 确保保存图片的文件夹存在
if not os.path.exists('figure'):
    os.makedirs('figure')


# Show loss curves
def show_history(train_loss, val_loss, train_acc, val_acc):
    """
    PyTorch版本：接收训练和验证的loss和accuracy列表来绘图。
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.title('Training loss performance')
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.legend()
    plt.savefig('figure/total_loss.png')
    plt.close()

    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(epochs, train_acc, label='train acc')
    plt.plot(epochs, val_acc, label='val acc')
    plt.legend()
    plt.savefig('figure/total_acc.png')
    plt.close()

    # 保存数据到文本文件
    np.savetxt('train_acc.txt', np.array(train_acc))
    np.savetxt('train_loss.txt', np.array(train_loss))
    np.savetxt('val_acc.txt', np.array(val_acc))
    np.savetxt('val_loss.txt', np.array(val_loss))


# 其他函数基本保持不变，因为它们处理的是numpy数组
# ----------------------------------------------------

def plot_lstm2layer_output(a, modulation_type=None, save_filename=None):
    plt.figure(figsize=(4, 3), dpi=600)
    plt.plot(range(128), a[0], label=modulation_type)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    if save_filename:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.close()


def plot_conv4layer_output(a, modulation_type=None):
    if not os.path.exists('./figure_conv4_output'):
        os.makedirs('./figure_conv4_output')
    plt.figure(figsize=(4, 3), dpi=600)
    for i in range(100):
        plt.plot(range(a.shape[2]), a[0, 0, :, i])  # 适应PyTorch的维度
        plt.xticks([])
        plt.yticks(size=20)
        save_filename = './figure_conv4_output/output%d.png' % i
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.close()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[], save_filename=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm * 100, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, size=12)
    plt.yticks(tick_marks, labels, size=12)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i != j:
                text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10)
            elif i == j:
                color = 'darkorange' if int(np.around(cm[i, j] * 100)) < 100 else 'white'
                text = plt.text(j, i, int(np.around(cm[i, j] * 100)), ha="center", va="center", fontsize=10,
                                color=color)
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()


def calculate_confusion_matrix(Y, Y_hat, classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])
    confnorm = np.zeros([n_classes, n_classes])

    # Y是one-hot编码, Y_hat是模型的softmax输出
    Y_true_indices = np.argmax(Y, axis=1)
    Y_pred_indices = np.argmax(Y_hat, axis=1)

    for i in range(0, Y.shape[0]):
        true_label = Y_true_indices[i]
        pred_label = Y_pred_indices[i]
        conf[true_label, pred_label] += 1

    for i in range(0, n_classes):
        row_sum = np.sum(conf[i, :])
        if row_sum > 0:
            confnorm[i, :] = conf[i, :] / row_sum

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong


def calculate_accuracy_each_snr(Y, Y_hat, Z, classes=None):
    Z_array = Z[:, 0]
    snrs = sorted(list(set(Z_array)))
    acc = np.zeros(len(snrs))
    Y_index = np.argmax(Y, axis=1)
    Y_index_hat = np.argmax(Y_hat, axis=1)
    i = 0
    for snr in snrs:
        indices = np.where(Z_array == snr)
        Y_snr = Y_index[indices]
        Y_hat_snr = Y_index_hat[indices]
        if Y_snr.shape[0] > 0:
            acc[i] = np.sum(Y_snr == Y_hat_snr) / Y_snr.shape[0]
        i += 1
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, acc, label='test_acc')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2018.01")
    plt.legend()
    plt.grid()
    plt.savefig('figure/snr_accuracy.png')  # 保存而不是显示
    plt.close()


def calculate_acc_at1snr_from_cm(cm):
    diag = np.diag(cm)
    row_sums = np.sum(cm, axis=1)
    # 避免除以零
    return np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums != 0)


def calculate_acc_cm_each_snr(Y, Y_hat, Z, classes=None, save_figure=True, min_snr=0):
    Z_array = Z[:, 0]
    snrs = sorted(list(set(Z_array)))
    acc = np.zeros(len(snrs))
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0
    for snr in snrs:
        indices = np.where(Z_array == snr)
        Y_snr = Y[indices]
        Y_hat_snr = Y_hat[indices]
        if Y_snr.shape[0] == 0:
            i += 1
            continue

        cm, right, wrong = calculate_confusion_matrix(Y_snr, Y_hat_snr, classes)

        if snr >= min_snr and save_figure:
            plot_confusion_matrix(cm, title=f'Confusion matrix at {snr}dB', cmap=plt.cm.Blues, labels=classes,
                                  save_filename=f'figure/cm_snr{snr}.png')

        total = right + wrong
        acc[i] = round(1.0 * right / total, 4) if total > 0 else 0

        with open('acc_per_snr.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([snr, acc[i]])

        print(f'Accuracy at {int(snr)}db: {acc[i] * 100:.2f}% ({int(right)} / {int(total)})')
        acc_mod_snr[:, i] = calculate_acc_at1snr_from_cm(cm)
        i += 1

    plt.figure(figsize=(8, 6))
    plt.plot(snrs, acc, label='test_acc', marker='o')
    for x, y in zip(snrs, acc):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on All Test Data")
    plt.legend()
    plt.grid()
    plt.savefig('figure/acc_overall.png')
    plt.close()

    dis_num = 6
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        beg_index = g * dis_num
        end_index = np.min([(g + 1) * dis_num, acc_mod_snr.shape[0]])
        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")
        for j in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[j], label=classes[j], marker='.')
        plt.legend()
        plt.grid()
        if save_figure:
            plt.savefig(f'figure/acc_with_mod_{g + 1}.png')
        plt.close()
