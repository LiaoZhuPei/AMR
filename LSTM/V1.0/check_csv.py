import pandas as pd

# --- 修改这里，指向你的 .csv 文件 ---
csv_file_path = 'E:/Project/AMR-Benchmark/HisarMod/dataset/HisarModdataset/Test/test_labels1.csv'

try:
    # 只读取前几行来查看列名，速度很快
    df = pd.read_csv(csv_file_path, nrows=5)

    print(f"Columns in '{csv_file_path}':")
    print(list(df.columns))

    print("\nFirst 5 rows of the file:")
    print(df)

except Exception as e:
    print(f"An error occurred: {e}")