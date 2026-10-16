import csv
import os

# labels parameters
La_NUMBERS = 14  # MATLAB仿真的类别的最大序列数
La_PER_NUMBERS = 4000  # 每个类别的总样本数量

# snr parameters
sn_REPEAT = 14   # 多少个类别具有信噪比
# range中的stop不包含在内
sn_VALUE = range(-2, 38, 2)  # 信噪比范围
sn_PER_NUMBERS = 500          # 单类别每个信噪比的样本数量

def labels(file_path, numbers, rows_per_number):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for num in range(numbers):
            for _ in range(rows_per_number):
                writer.writerow([num])

def snr(file_path, repeat, snr_value, rows_per_value):
    with open(file_path,'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _ in range(repeat):
            for value in snr_value:
                for _ in range(rows_per_value):
                    writer.writerow([value])

def main():
    output_dir = 'E:/Project/AMR-Benchmark/HisarMod/dataset/DataCla14_A/train'
    os.makedirs(output_dir, exist_ok=True)

    creat_snr = False

    if not creat_snr:
        file_path = os.path.join(output_dir, 'train_labels.csv')
        print("正在写入labels.")
        labels(file_path, La_NUMBERS, La_PER_NUMBERS)
    else:
        file_path = os.path.join(output_dir, 'train_snr.csv')
        print("正在写入snr.")
        snr(file_path, sn_REPEAT, sn_VALUE, sn_PER_NUMBERS)

if __name__ == '__main__':
    main()
