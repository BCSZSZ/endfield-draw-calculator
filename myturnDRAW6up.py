import numpy as np


def calculate_exact_full_potential(
    target_copies: int = 6, max_pity: int = 80, max_sim_pulls: int = 1500,
    initial_pity: int = 0,
) -> dict:
    from draw_model import character_states, distribution_statistics

    cdf = np.array([
        state[:, :, target_copies].sum()
        for state in character_states(
            max_sim_pulls, initial_pity, copies_cap=target_copies, max_pity=max_pity,
        )
    ])
    return distribution_statistics(cdf)


def print_result(target_copies, result):
    print(f"=== 满潜能 ({target_copies}个UP) 精确计算结果 ===")
    print(f"总计算覆盖率: {result['check_sum'] * 100:.6f}%")
    print(f"数学期望 (Mean): {result['mean']:.4f} 抽")
    print(f"标准差 (Std Dev): {result['std_dev']:.4f}")
    print(f"方差 (Variance): {result['variance']:.4f}")
    print(f"一倍标准差内: {result['within_one_std'] * 100:.6f}%")
    print("-" * 30)
    print(f"欧皇线 (Top 25%): {result['p25']} 抽")
    print(f"中位数 (Top 50%): {result['p50']} 抽")
    print(f"普通人 (Top 75%): {result['p75']} 抽")
    print(f"非酋线 (Top 99%): {result['p99']} 抽")


if __name__ == "__main__":
    result_1 = calculate_exact_full_potential(1)
    print_result(1, result_1)

    result_6 = calculate_exact_full_potential(6)
    print_result(6, result_6)
