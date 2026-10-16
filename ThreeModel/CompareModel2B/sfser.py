import numpy as np


def generate_data_A(total_len=4800, ones_len=768):
    """生成情况A的数据：开头768个为1，其余为5"""
    data = np.full(total_len, 5)
    data[:ones_len] = 1
    return data


def generate_data_B(total_len=4800, ones_len=768):
    """生成情况B的数据：中间768个为1，前后为5"""
    data = np.full(total_len, 5)
    # 计算中间位置的起始索引: (4800 - 768) / 2 = 2016
    start_idx = (total_len - ones_len) // 2
    data[start_idx: start_idx + ones_len] = 1
    return data


def analyze_sliding_window(data, window_size=1024, step=16):
    """使用滑动窗口遍历数据并统计"""
    total_windows = 0

    # 统计指标
    contains_1_count = 0  # 包含数字1的窗口数
    ratio_less_50_count = 0  # 包含数字1，且比例 < 50%
    ratio_greater_50_count = 0  # 包含数字1，且比例 > 50%
    ratio_equal_50_count = 0  # 包含数字1，且比例 == 50% (边界情况补充)
    no_1_count = 0  # 完全不包含数字1的窗口数 (比例=0%)

    # 滑动窗口遍历
    for i in range(0, len(data) - window_size + 1, step):
        total_windows += 1
        window = data[i: i + window_size]

        # 统计窗口内1的个数
        count_1 = np.sum(window == 1)

        if count_1 > 0:
            contains_1_count += 1
            ratio = count_1 / window_size

            if ratio < 0.5:
                ratio_less_50_count += 1
            elif ratio > 0.5:
                ratio_greater_50_count += 1
            else:
                ratio_equal_50_count += 1
        else:
            no_1_count += 1

    return {
        "total_windows": total_windows,
        "contains_1": contains_1_count,
        "contains_1_and_less_50": ratio_less_50_count,
        "greater_than_50": ratio_greater_50_count,
        "equal_50": ratio_equal_50_count,
        "no_1": no_1_count,
        "all_less_than_50": ratio_less_50_count + no_1_count  # 包含0%的情况
    }


def print_report(case_name, stats):
    print(f"========== {case_name} 统计报告 ==========")
    print(f"滑动窗口总数: {stats['total_windows']}")
    print(f"1. 包含数字1的窗口数: {stats['contains_1']}")
    print(f"2. 数字1比例低于50%的窗口数:")
    print(f"   -> (仅看包含1的窗口) 0% < 比例 < 50%: {stats['contains_1_and_less_50']}")
    print(f"   -> (含完全没有1的窗口) 比例 < 50%的总数: {stats['all_less_than_50']}")
    print(f"3. 数字1比例高于50%的窗口数: {stats['greater_than_50']}")
    print(f"*. 补充: 比例刚好等于50%的窗口数: {stats['equal_50']}")
    print(f"*. 补充: 完全没有数字1的窗口数: {stats['no_1']}")
    print("\n")


if __name__ == "__main__":
    # 参数设置
    TOTAL_LEN = 4800
    WINDOW_SIZE = 1024
    STEP = 800  # 你可以在这里自定义步长

    print(f"【全局配置】数据总长: {TOTAL_LEN}, 窗口长度: {WINDOW_SIZE}, 步长: {STEP}\n")

    # 情况 A
    data_A = generate_data_A(total_len=TOTAL_LEN, ones_len=768)
    stats_A = analyze_sliding_window(data_A, window_size=WINDOW_SIZE, step=STEP)
    print_report("情况 A (开头为1)", stats_A)

    # 情况 B
    data_B = generate_data_B(total_len=TOTAL_LEN, ones_len=768)
    stats_B = analyze_sliding_window(data_B, window_size=WINDOW_SIZE, step=STEP)
    print_report("情况 B (中间为1)", stats_B)