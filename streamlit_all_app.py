import streamlit as st

from streamlit_pages.distribution_page import render_distribution_page
from streamlit_pages.joint_page import render_joint_page
from streamlit_theme import apply_theme_css

st.set_page_config(page_title="终末地抽卡计算器", layout="wide")
apply_theme_css()

st.title("抽卡达成目标分布")

with st.sidebar:
    st.header("模式")
    mode = st.selectbox("选择功能", ["干员抽卡", "武器抽卡", "联合目标(干员+武器)"])

if mode in {"干员抽卡", "武器抽卡"}:
    render_distribution_page(mode)
else:
    render_joint_page()
