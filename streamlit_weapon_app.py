import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weapon_draw import calculate_weapon_full_potential

st.set_page_config(page_title="Weapon Draw Simulator", layout="wide")

st.title("武器抽卡达成目标分布")

with st.sidebar:
    st.header("参数")
    target_copies = st.slider("目标UP数量", min_value=1, max_value=6, value=6, step=1)
    max_sim_pulls = st.slider(
        "最大10连次数", min_value=1, max_value=100, value=60, step=1
    )

result = calculate_weapon_full_potential(
    target_copies=target_copies, max_sim_pulls=max_sim_pulls
)

costs = result["costs"]
finish_probs = result["finish_probs"]
cdf = finish_probs.cumsum()

mean = result["mean"]
std_dev = result["std_dev"]


def cdf_at(pulls):
    idx = max(0, min(int(pulls), len(costs) - 1))
    return cdf[idx]


def estimate_pulls_for_prob(target_prob):
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
st.markdown("**说明**：第一列为10连次数(向上取整)，第二列为完成概率。")

prob_table = f"""
<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="text-align:left; border-bottom:1px solid #444; padding:6px;">10连次数</th>
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

st.subheader("输入百分比计算10连次数")
percent_input = st.number_input(
    "完成概率(%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1
)
target_prob = percent_input / 100.0
pulls_for_percent = estimate_pulls_for_prob(target_prob)
pulls_for_percent_ceiled = math.ceil(float(pulls_for_percent))
st.write(
    f"完成概率 {percent_input:.2f}% 对应10连次数≈{float(pulls_for_percent):.4f}，向上取整={pulls_for_percent_ceiled}"
)

st.subheader("区间概率")
range_col1, range_col2 = st.columns(2)
range_start = range_col1.number_input(
    "起始10连次数", min_value=0, max_value=max_sim_pulls, value=0, step=1
)
range_end = range_col2.number_input(
    "结束10连次数", min_value=0, max_value=max_sim_pulls, value=max_sim_pulls, step=1
)

start = min(range_start, range_end)
end = max(range_start, range_end)
range_prob = finish_probs[start : end + 1].sum()
st.write(f"区间 [{start}, {end}] 内完成的概率: {range_prob * 100:.6f}%")

st.subheader("分布明细")
detail_df = pd.DataFrame(
    {"10连次数": costs, "离散概率(PMF)": finish_probs, "累计分布(CDF)": cdf}
)
st.dataframe(detail_df, width="stretch", hide_index=True)

st.subheader("概率分布")
pmf_fig = go.Figure()
pmf_fig.add_bar(x=costs, y=finish_probs, name="离散概率(PMF)")
pmf_fig.add_vline(
    x=mean, line_dash="dash", line_color="#E45756", annotation_text="均值(Mean)"
)
pmf_fig.update_layout(
    title=f"离散概率(PMF) (target={target_copies})",
    xaxis_title="10连次数",
    yaxis_title="恰好在该次完成的概率",
)

cdf_fig = go.Figure()
cdf_fig.add_scatter(x=costs, y=cdf, mode="lines", name="累计分布(CDF)")
cdf_fig.add_vline(x=mean, line_dash="dash", line_color="#E45756")
cdf_fig.update_layout(
    title=f"累计分布(CDF) (target={target_copies})",
    xaxis_title="10连次数",
    yaxis_title="在该次前完成的概率",
)

chart_col1, chart_col2 = st.columns(2)
chart_col1.plotly_chart(pmf_fig, width="stretch")
chart_col2.plotly_chart(cdf_fig, width="stretch")
