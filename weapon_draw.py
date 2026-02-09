import math

import numpy as np


def calculate_weapon_full_potential(target_copies=6, max_sim_pulls=70):
    # 单抽概率：6星4%，UP占25% => UP 1%，其他6星 3%
    p_up = 0.01
    p_six = 0.04
    p_no_six = 1.0 - p_six

    # 10连分布：k个UP
    binom10 = [
        math.comb(10, k) * (p_up**k) * ((1.0 - p_up) ** (10 - k)) for k in range(11)
    ]
    binom9 = [
        math.comb(9, k) * (p_up**k) * ((1.0 - p_up) ** (9 - k)) for k in range(10)
    ]

    p_no_up_10 = (1.0 - p_up) ** 10
    p_no_six_10 = p_no_six**10
    p_other_six_10 = p_no_up_10 - p_no_six_10

    # 正常10连：k>=1为UP命中；k=0再区分无6星/有6星无UP
    normal_k_probs = {k: binom10[k] for k in range(1, 11)}

    # 6星保底10连：1个必出6星 + 9次正常
    # 25%必出UP；75%必出非UP
    g6_k_probs = {
        k: 0.25 * (binom9[k - 1] if k - 1 >= 0 and k - 1 <= 9 else 0.0)
        + 0.75 * (binom9[k] if k <= 9 else 0.0)
        for k in range(0, 11)
    }

    # UP保底10连：1个必出UP + 9次正常
    gup_k_probs = {
        k: binom9[k - 1] if k - 1 >= 0 and k - 1 <= 9 else 0.0 for k in range(1, 11)
    }

    # pity6: 连续未出6星次数(0-3)，3表示下一次必出6星
    # up_pity: 仅统计“前7次十连是否出UP”，0-7为计数，8为规则已失效/已触发
    up_pity_max = 7
    up_pity_disabled = 8
    dp = np.zeros((4, 9, target_copies + 1))
    dp[0, 0, 0] = 1.0

    finish_probs = np.zeros(max_sim_pulls + 1)

    for cost in range(1, max_sim_pulls + 1):
        new_dp = np.zeros_like(dp)

        for p6 in range(4):
            for pu in range(9):
                for c in range(target_copies):
                    prob = dp[p6, pu, c]
                    if prob == 0:
                        continue

                    if pu == up_pity_max:
                        # 仅触发一次的UP保底10连
                        for k, pk in gup_k_probs.items():
                            final_c = c + k
                            if final_c >= target_copies:
                                finish_probs[cost] += prob * pk
                            else:
                                new_dp[0, up_pity_disabled, final_c] += prob * pk
                        continue

                    if p6 == 3:
                        # 6星保底10连
                        # k>=1为UP命中
                        for k in range(1, 11):
                            pk = g6_k_probs[k]
                            if pk == 0:
                                continue
                            final_c = c + k
                            if final_c >= target_copies:
                                finish_probs[cost] += prob * pk
                            else:
                                next_pu = (
                                    up_pity_disabled
                                    if pu != up_pity_disabled
                                    else up_pity_disabled
                                )
                                new_dp[0, next_pu, final_c] += prob * pk

                        # k==0：必有6星但非UP
                        pk0 = g6_k_probs[0]
                        if pk0 > 0:
                            if pu == up_pity_disabled:
                                next_pu = up_pity_disabled
                            else:
                                next_pu = min(pu + 1, up_pity_max)
                            new_dp[0, next_pu, c] += prob * pk0
                        continue

                    # 正常10连
                    for k, pk in normal_k_probs.items():
                        final_c = c + k
                        if final_c >= target_copies:
                            finish_probs[cost] += prob * pk
                        else:
                            next_pu = (
                                up_pity_disabled
                                if pu != up_pity_disabled
                                else up_pity_disabled
                            )
                            new_dp[0, next_pu, final_c] += prob * pk

                    # k==0：分成无6星/有6星无UP
                    if p_no_six_10 > 0:
                        if pu == up_pity_disabled:
                            next_pu = up_pity_disabled
                        else:
                            next_pu = min(pu + 1, up_pity_max)
                        new_dp[min(p6 + 1, 3), next_pu, c] += prob * p_no_six_10
                    if p_other_six_10 > 0:
                        if pu == up_pity_disabled:
                            next_pu = up_pity_disabled
                        else:
                            next_pu = min(pu + 1, up_pity_max)
                        new_dp[0, next_pu, c] += prob * p_other_six_10

        # 赠送规则：10连箱子忽略；18连送UP；之后每8连交替送箱子和UP
        if cost == 18 or (cost > 18 and (cost - 18) % 16 == 0):
            temp_dp = np.zeros_like(new_dp)
            for c in range(target_copies):
                if c + 1 >= target_copies:
                    finish_probs[cost] += np.sum(new_dp[:, :, c])
                else:
                    temp_dp[:, :, c + 1] += new_dp[:, :, c]
            new_dp = temp_dp

        dp = new_dp

        if np.sum(dp) < 1e-9:
            break

    total_prob = np.sum(finish_probs)
    costs = np.arange(max_sim_pulls + 1)

    expected_value = np.sum(costs * finish_probs)
    expected_sq = np.sum((costs**2) * finish_probs)
    variance = expected_sq - (expected_value**2)
    std_dev = np.sqrt(variance)

    lower = max(0, int(np.ceil(expected_value - std_dev)))
    upper = min(max_sim_pulls, int(np.floor(expected_value + std_dev)))
    within_one_std = np.sum(finish_probs[lower : upper + 1])

    cdf = np.cumsum(finish_probs)

    def get_percentile(p):
        return int(np.searchsorted(cdf, p))

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
    print(f"=== 武器满潜能 ({target_copies}个UP) 精确计算结果 ===")
    print(f"总计算覆盖率: {result['check_sum'] * 100:.6f}%")
    print(f"数学期望 (Mean): {result['mean']:.4f} 次10连")
    print(f"标准差 (Std Dev): {result['std_dev']:.4f}")
    print(f"方差 (Variance): {result['variance']:.4f}")
    print(f"一倍标准差内: {result['within_one_std'] * 100:.6f}%")
    print("-" * 30)
    print(f"欧皇线 (Top 25%): {result['p25']} 次10连")
    print(f"中位数 (Top 50%): {result['p50']} 次10连")
    print(f"普通人 (Top 75%): {result['p75']} 次10连")
    print(f"非酋线 (Top 99%): {result['p99']} 次10连")


if __name__ == "__main__":
    for target in range(1, 7):
        res = calculate_weapon_full_potential(target_copies=target, max_sim_pulls=70)
        print_result(target, res)
