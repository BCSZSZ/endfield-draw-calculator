import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from myturnDRAW6up import calculate_exact_full_potential
from streamlit_theme import THEME, sync_state_value
from weapon_draw import calculate_weapon_full_potential


def render_distribution_page(mode: str) -> None:
    with st.sidebar:
        if mode == "干员抽卡":
            target_copies = st.slider(
                "目标UP数量", min_value=1, max_value=6, value=6, step=1
            )
            if "initial_pity_input" not in st.session_state:
                st.session_state.initial_pity_input = 0
            if "initial_pity_slider" not in st.session_state:
                st.session_state.initial_pity_slider = 0

            st.slider(
                "初始水位(滑块)",
                min_value=0,
                max_value=79,
                step=1,
                key="initial_pity_slider",
                on_change=sync_state_value,
                args=("initial_pity_input", "initial_pity_slider"),
            )
            initial_pity_input = st.number_input(
                "初始水位(输入)",
                min_value=0,
                max_value=79,
                step=1,
                key="initial_pity_input",
                on_change=sync_state_value,
                args=("initial_pity_slider", "initial_pity_input"),
            )

            if "max_sim_pulls_input" not in st.session_state:
                st.session_state.max_sim_pulls_input = 800
            if "max_sim_pulls_slider" not in st.session_state:
                st.session_state.max_sim_pulls_slider = 800

            st.slider(
                "最大模拟抽数(滑块)",
                min_value=1,
                max_value=1200,
                step=30,
                key="max_sim_pulls_slider",
                on_change=sync_state_value,
                args=("max_sim_pulls_input", "max_sim_pulls_slider"),
            )
            max_sim_pulls_input = st.number_input(
                "最大模拟抽数(输入)",
                min_value=1,
                max_value=1200,
                step=1,
                key="max_sim_pulls_input",
                on_change=sync_state_value,
                args=("max_sim_pulls_slider", "max_sim_pulls_input"),
            )
            max_sim_pulls = int(max_sim_pulls_input)
            result = calculate_exact_full_potential(
                target_copies=target_copies,
                max_sim_pulls=max_sim_pulls,
                initial_pity=int(initial_pity_input),
            )
            unit_label = "抽数"
            xaxis_label = "抽数(Pulls)"
            yaxis_pmf = "恰好在该抽完成的概率(P exactly at pulls)"
            yaxis_cdf = "在该抽前完成的概率(P finish by pulls)"
        else:
            target_copies = st.slider(
                "目标UP数量", min_value=1, max_value=6, value=6, step=1
            )
            if "max_sim_pulls10_input" not in st.session_state:
                st.session_state.max_sim_pulls10_input = 60
            if "max_sim_pulls10_slider" not in st.session_state:
                st.session_state.max_sim_pulls10_slider = 60

            st.slider(
                "最大10连次数(滑块)",
                min_value=1,
                max_value=100,
                step=1,
                key="max_sim_pulls10_slider",
                on_change=sync_state_value,
                args=("max_sim_pulls10_input", "max_sim_pulls10_slider"),
            )
            max_sim_pulls_input = st.number_input(
                "最大10连次数(输入)",
                min_value=1,
                max_value=100,
                step=1,
                key="max_sim_pulls10_input",
                on_change=sync_state_value,
                args=("max_sim_pulls10_slider", "max_sim_pulls10_input"),
            )
            max_sim_pulls = int(max_sim_pulls_input)
            result = calculate_weapon_full_potential(
                target_copies=target_copies, max_sim_pulls=max_sim_pulls
            )
            unit_label = "10连次数"
            xaxis_label = "10连次数"
            yaxis_pmf = "恰好在该次完成的概率"
            yaxis_cdf = "在该次前完成的概率"

    costs = result["costs"]
    finish_probs = result["finish_probs"]
    cdf = finish_probs.cumsum()

    mean = result["mean"]
    std_dev = result["std_dev"]
    lower, upper = result["one_std_range"]

    def cdf_at(pulls: int) -> float:
        idx = max(0, min(int(pulls), len(costs) - 1))
        return float(cdf[idx])

    def estimate_pulls_for_prob(target_prob: float) -> float:
        prev = 0.0
        for i, value in enumerate(cdf):
            if value >= target_prob:
                if value == prev:
                    return float(i)
                frac = (target_prob - prev) / (value - prev)
                return float(i - 1) + frac
            prev = float(value)
        return float(len(cdf) - 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("均值(Mean)", f"{mean:.2f}")
    col2.metric("标准差(Std Dev)", f"{std_dev:.2f}")
    col3.metric("1σ内", f"{result['within_one_std'] * 100:.2f}%")
    col4.metric("覆盖率", f"{result['check_sum'] * 100:.2f}%")

    mean_ceiled = math.ceil(float(mean))
    mean_plus_std = float(mean + std_dev)
    mean_plus_std_ceiled = math.ceil(mean_plus_std)
    prob_at_mean = cdf_at(mean_ceiled)
    prob_at_mean_plus_std = cdf_at(mean_plus_std_ceiled)

    pulls_for_99 = estimate_pulls_for_prob(0.99)
    pulls_for_99_ceiled = math.ceil(float(pulls_for_99))

    st.subheader("指定抽数下的完成概率")
    st.markdown(f"**说明**：第一列为{unit_label}(向上取整)，第二列为完成概率。")
    st.markdown(
        "前两行关注均值与均值+标准差范围内达成目标的概率，第三行关注达成概率首次超过99%时的抽数。"
    )

    prob_table = f"""
<table style="width:100%; border-collapse: collapse;">
    <thead>
        <tr>
            <th style="text-align:left; border-bottom:1px solid #444; padding:6px;">{unit_label}</th>
            <th style="text-align:left; border-bottom:1px solid #444; padding:6px;">完成概率</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="padding:6px;">{mean_ceiled}（均值）</td>
            <td style="padding:6px; font-weight:700;">{float(prob_at_mean) * 100:.6f}%</td>
        </tr>
        <tr>
            <td style="padding:6px;">{mean_plus_std_ceiled}（均值+标准差）</td>
            <td style="padding:6px; font-weight:700;">{float(prob_at_mean_plus_std) * 100:.6f}%</td>
        </tr>
        <tr>
            <td style="padding:6px; font-weight:700;">{pulls_for_99_ceiled}</td>
            <td style="padding:6px; font-weight:700;">99%</td>
        </tr>
    </tbody>
</table>
"""
    st.markdown(prob_table, unsafe_allow_html=True)

    st.subheader(f"输入百分比计算{unit_label}")
    percent_input = st.number_input(
        "完成概率(%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1
    )
    target_prob = percent_input / 100.0
    pulls_for_percent = estimate_pulls_for_prob(target_prob)
    pulls_for_percent_ceiled = math.ceil(float(pulls_for_percent))
    st.write(
        f"完成概率 {percent_input:.2f}% 对应{unit_label}≈{float(pulls_for_percent):.4f}，向上取整{unit_label}={pulls_for_percent_ceiled}"
    )

    st.subheader("区间概率")
    range_col1, range_col2 = st.columns(2)
    range_start = range_col1.number_input(
        f"起始{unit_label}", min_value=0, max_value=max_sim_pulls, value=0, step=1
    )
    range_end = range_col2.number_input(
        f"结束{unit_label}",
        min_value=0,
        max_value=max_sim_pulls,
        value=max_sim_pulls,
        step=1,
    )

    start = min(range_start, range_end)
    end = max(range_start, range_end)
    range_prob = finish_probs[start : end + 1].sum()
    st.write(f"区间 [{start}, {end}] 内完成的概率: {range_prob * 100:.6f}%")

    st.subheader("分布明细")
    detail_df = pd.DataFrame(
        {f"{unit_label}": costs, "离散概率(PMF)": finish_probs, "累计分布(CDF)": cdf}
    )
    st.dataframe(detail_df, width="stretch", hide_index=True)

    st.subheader("概率分布")
    pmf_fig = go.Figure()
    pmf_fig.add_bar(
        x=costs, y=finish_probs, name="离散概率(PMF)", marker_color=THEME["accent"]
    )
    pmf_fig.add_vline(
        x=mean, line_dash="dash", line_color=THEME["warn"], annotation_text="均值(Mean)"
    )
    pmf_fig.add_vrect(
        x0=lower, x1=upper, fillcolor=THEME["accent_alt"], opacity=0.12, line_width=0
    )
    for key in ["p25", "p50", "p75", "p99"]:
        pmf_fig.add_vline(x=result[key], line_dash="dot", line_color=THEME["accent_alt"])

    pmf_fig.update_layout(
        title=f"离散概率(PMF) (target={target_copies})",
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_pmf,
        font=dict(color=THEME["text"]),
        paper_bgcolor=THEME["panel_alt"],
        plot_bgcolor=THEME["panel_alt"],
        xaxis=dict(gridcolor=THEME["grid"]),
        yaxis=dict(gridcolor=THEME["grid"]),
    )

    cdf_fig = go.Figure()
    cdf_fig.add_scatter(
        x=costs,
        y=cdf,
        mode="lines",
        name="累计分布(CDF)",
        line=dict(color=THEME["accent"]),
    )
    cdf_fig.add_vline(x=mean, line_dash="dash", line_color=THEME["warn"])
    cdf_fig.add_vrect(
        x0=lower, x1=upper, fillcolor=THEME["accent_alt"], opacity=0.12, line_width=0
    )
    for key in ["p25", "p50", "p75", "p99"]:
        cdf_fig.add_vline(x=result[key], line_dash="dot", line_color=THEME["accent_alt"])
    cdf_fig.update_layout(
        title=f"累计分布(CDF) (target={target_copies})",
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_cdf,
        font=dict(color=THEME["text"]),
        paper_bgcolor=THEME["panel_alt"],
        plot_bgcolor=THEME["panel_alt"],
        xaxis=dict(gridcolor=THEME["grid"]),
        yaxis=dict(gridcolor=THEME["grid"]),
    )

    chart_col1, chart_col2 = st.columns(2)
    chart_col1.plotly_chart(pmf_fig, width="stretch")
    chart_col2.plotly_chart(cdf_fig, width="stretch")

    st.divider()
    if mode == "干员抽卡":
        st.subheader("数学模型与规则说明")
        st.markdown(
            """
- **模型**：使用离散时间马尔可夫链 DP，状态为 `(6星水位, 已获得UP数)`，逐抽更新概率质量。
- **基础概率**：6星基础概率为 0.8%，66 抽后每抽增加 5%，80 抽必出 6 星；6 星中 UP 占 50%。
- **5星/4星层**：5 星基础 8%，若达到 10 抽未出 5 星则该抽保底至少 5 星。
- **额外规则**：30 抽触发一次“赠送 10 连”（按单抽 UP=0.4% 的二项分布入模）；120 抽首发硬保底（若仍 0 UP 则补 1 个 UP）；每 240 抽额外送 1 个 UP。
- **前提**：本页默认初始水位来自侧边栏输入，结果为精确分布（非蒙特卡洛），显示 PMF/CDF、均值与分位点。
"""
        )
    else:
        st.subheader("数学模型与规则说明")
        st.markdown(
            """
- **模型**：使用离散状态 DP，状态为 `(连续未出6星的10连数, 连续未出UP的10连计数, 已获得UP数)`，按“每次 10 连”更新。
- **基础概率**：单抽 6 星概率 4%，其中 UP 概率 1%（即 6 星内 UP 占 25%）；10 连内 UP 数按二项分布计算。
- **6星保底10连**：若连续 3 次 10 连未出 6 星，则下一次 10 连必出至少 1 个 6 星，并按 25%/75% 拆分为 UP/非UP。
- **UP保底10连**：前 7 次 10 连若持续未出 UP，则触发一次“UP 保底 10 连”；触发后该规则失效。
- **赠送规则**：第 18 次 10 连送 1 个 UP，之后每 16 次触发一次送 UP（对应 18, 34, 50, ...）。
- **前提**：本页以 10 连次数为单位输出精确 PMF/CDF，不模拟箱子价值，仅计算达成目标 UP 数概率分布。
"""
        )
