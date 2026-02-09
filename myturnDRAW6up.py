import math

import numpy as np


def calculate_exact_full_potential(
    target_copies=6, max_pity=80, max_sim_pulls=1500, initial_pity=0
):
    # --- 配置参数 ---

    # 概率函数 (返回 UP概率, 歪概率, 不出概率)
    def get_rates(pity_counter):
        if pity_counter < 66:
            base = 0.008
        elif pity_counter < 80:
            base = 0.008 + (pity_counter - 65) * 0.05
        else:
            base = 1.0  # 第80抽必出

        base = min(1.0, base)
        return base * 0.5, base * 0.5, 1.0 - base

    # --- 初始化 DP ---
    # dp[pity][copies]
    # pity: 0~80, copies: 0~6
    dp = np.zeros((max_pity + 1, target_copies + 1))
    # 初始状态：初始水位，0个角色
    init_pity = max(0, min(int(initial_pity), max_pity))
    dp[init_pity][0] = 1.0

    # 结果记录：finish_probs[cost] = 恰好在 cost 抽达成满潜的概率
    finish_probs = np.zeros(max_sim_pulls + 1)

    # --- 逐抽计算 (Cost 1 to Max) ---
    for cost in range(1, max_sim_pulls + 1):
        new_dp = np.zeros((max_pity + 1, target_copies + 1))

        # 1. 执行抽卡逻辑
        for p in range(max_pity + 1):
            # 获取当前水位下的概率
            p_up, p_other, p_fail = get_rates(p)

            # 对当前持有数量 c 进行遍历 (如果已经6个了就不需要算了，那是结束状态)
            for c in range(target_copies):
                prob = dp[p][c]
                if prob == 0:
                    continue

                # 分支 A: 抽到 UP (水位清零, copies+1)
                if c + 1 == target_copies:
                    finish_probs[cost] += prob * p_up  # 达成目标，记录到结果
                else:
                    new_dp[0][c + 1] += prob * p_up

                # 分支 B: 抽到 歪 (水位清零, copies不变)
                new_dp[0][c] += prob * p_other

                # 分支 C: 没抽到 (水位+1, copies不变)
                if p + 1 <= max_pity:
                    new_dp[p + 1][c] += prob * p_fail
                else:
                    # 理论上 p=80 时 p_fail=0，不会进这里，但为了严谨处理边界：
                    new_dp[0][c] += prob * p_fail  # 实际上是0

        # --- 2. 处理一次性特殊规则 ---

        # [规则3] 30抽赠送10连 (一次性)
        if cost == 30:
            # 这是一个二项分布：10次试验，每次成功率0.004 (UP率)
            # 为了DP的连续性，我们将 new_dp 中的状态进行分裂
            temp_dp = np.zeros_like(new_dp)

            # 计算获得 k 个 UP 的概率 (k=0到10)
            # binom_probs[k]
            binom_probs = [0.0] * 11
            p_bonus = 0.004
            for k in range(11):
                # C(10, k) * p^k * (1-p)^(10-k)
                coef = math.comb(10, k)
                binom_probs[k] = coef * (p_bonus**k) * ((1 - p_bonus) ** (10 - k))

            # 将 binom_probs 应用到当前所有状态
            for c in range(target_copies):
                current_mass = np.sum(
                    new_dp[:, c]
                )  # 当前持有 c 个的总概率质量 (pity分布不影响赠送)
                if current_mass == 0:
                    continue

                # 注意：我们要保留 pity 分布，不能简单 sum。
                # 正确做法：对每个 (p, c) 状态应用转移
                for p in range(max_pity + 1):
                    val = new_dp[p][c]
                    if val == 0:
                        continue

                    for k in range(11):
                        prob_k = binom_probs[k]
                        final_c = c + k

                        if final_c >= target_copies:
                            # 直接在第30抽毕业
                            # 注意：这里我们算作在第30抽这一瞬间毕业
                            finish_probs[cost] += val * prob_k
                        else:
                            temp_dp[p][final_c] += val * prob_k

            new_dp = temp_dp

        # [规则4] 120抽首发硬保底 (一次性)
        if cost == 120:
            # 只有当 copies == 0 时触发
            # 将所有 new_dp[p][0] 的概率移动到 new_dp[p][1]
            shift_mass = new_dp[:, 0].copy()
            new_dp[:, 0] = 0.0  # 清空 0 命状态

            # 移动到 1 命
            # 注意：如果目标就是1命，则算毕业
            if target_copies == 1:
                finish_probs[cost] += np.sum(shift_mass)
            else:
                # 120抽必出视为6星出货，重置水位
                total_shift = np.sum(shift_mass)
                if total_shift > 0:
                    new_dp[0][1] += total_shift

        # [规则5] 240抽里程碑赠送 (循环)
        if cost % 240 == 0:
            # 所有未毕业的状态，copies + 1
            temp_dp = np.zeros_like(new_dp)
            for c in range(target_copies):
                # 移动 c -> c+1
                if c + 1 >= target_copies:
                    # 直接毕业
                    # 将这一列的所有概率加到 finish_probs
                    finish_probs[cost] += np.sum(new_dp[:, c])
                else:
                    temp_dp[:, c + 1] += new_dp[:, c]
            new_dp = temp_dp

        # 更新 DP 表
        dp = new_dp

        # 优化：如果剩余概率极小，提前结束 (精度截断)
        if np.sum(dp) < 1e-9:
            break

    # --- 统计计算 ---
    total_prob = np.sum(finish_probs)

    # 归一化 (理论上 total_prob 应该接近 1.0)
    # 如果 max_sim_pulls 设得太小，可能会小于1
    costs = np.arange(max_sim_pulls + 1)

    expected_value = np.sum(costs * finish_probs)
    expected_sq = np.sum((costs**2) * finish_probs)
    variance = expected_sq - (expected_value**2)
    std_dev = np.sqrt(variance)

    lower = max(0, int(np.ceil(expected_value - std_dev)))
    upper = min(max_sim_pulls, int(np.floor(expected_value + std_dev)))
    within_one_std = np.sum(finish_probs[lower : upper + 1])

    # 累积分布 (CDF) 用于计算分位点
    cdf = np.cumsum(finish_probs)

    def get_percentile(p):
        idx = np.searchsorted(cdf, p)
        return idx

    return {
        "costs": costs,
        "finish_probs": finish_probs,
        "mean": expected_value,
        "variance": variance,
        "std_dev": std_dev,
        "p25": get_percentile(0.25),
        "p50": get_percentile(0.50),
        "p75": get_percentile(0.75),
        "p99": get_percentile(0.99),
        "check_sum": total_prob,
        "within_one_std": within_one_std,
        "one_std_range": (lower, upper),
    }


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


# --- 执行并输出 ---
result_1 = calculate_exact_full_potential(1)
print_result(1, result_1)

result_6 = calculate_exact_full_potential(6)
print_result(6, result_6)
