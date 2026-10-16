import random

# ================= 配置区 =================
# 设置生成的总比特数。
# 120,000 个比特足够长，能让 256QAM 星座图的每一个点平均被击中近 500 次。
# 且 120000 是 24 (6和8的最小公倍数) 的整数倍。
TOTAL_BITS = 120000

# 设置输出的文件名
OUTPUT_FILENAME = "vsg25a_test_data.txt"


# ==========================================

def generate_vsg_data():
    print(f"[*] 开始生成 {TOTAL_BITS} 位的纯随机比特序列...")

    # 检查是否为 24 的倍数，给出温馨提示
    if TOTAL_BITS % 24 != 0:
        print("[!] 警告: 建议将总比特数设置为 24 的倍数，以便完美兼容 64QAM 和 256QAM。")

    # 使用 random.choices 快速生成由 '0' 和 '1' 组成的列表，并拼接成字符串
    bit_list = random.choices(['0', '1'], k=TOTAL_BITS)
    bit_string = "".join(bit_list)

    print(f"[*] 序列生成完毕，正在写入文件: {OUTPUT_FILENAME}")

    # 将字符串写入文件，不添加任何换行符、空格或制表符
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(bit_string)

    print("[+] 写入成功！")
    print("-" * 40)
    print("【信号验证信息】")
    print(f"文件大小: {len(bit_string)} bits")
    print(f"测试 64QAM (6 bits/symbol) 时，包含 {TOTAL_BITS / 6:.0f} 个完整符号，余 {TOTAL_BITS % 6} 个比特。")
    print(f"测试 256QAM (8 bits/symbol) 时，包含 {TOTAL_BITS / 8:.0f} 个完整符号，余 {TOTAL_BITS % 8} 个比特。")
    print("-" * 40)
    print("现在你可以打开 VSG25A 的 Symbol Editor，点击 'Load Data' 导入该文件了。")


if __name__ == "__main__":
    generate_vsg_data()