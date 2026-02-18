import streamlit as st


THEME = {
    "panel": "#0f141b",
    "panel_alt": "#0f141b",
    "text": "#e6eef7",
    "muted": "#9fb0c1",
    "accent": "#3bd6c6",
    "accent_alt": "#7cc6ff",
    "warn": "#f4b259",
    "grid": "#243140",
}


def apply_theme_css() -> None:
    st.markdown(
        f"""
<style>
h1, h2, h3, h4, h5, h6, p, label, span, div, input, textarea, select, button {{
    font-family: "SimSun", "NSimSun", "FangSong", "STFangsong", "Songti SC", serif;
}}

h1, h2, h3 {{
    letter-spacing: 0.5px;
}}

div[data-testid="metric-container"] {{
    background: {THEME["panel"]};
    border: 1px solid {THEME["grid"]};
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 0 0 1px rgba(59, 214, 198, 0.08);
}}

div[data-testid="metric-container"] label {{
    color: {THEME["muted"]};
}}

div[data-testid="metric-container"] p {{
    color: {THEME["text"]};
}}

table {{
    border-collapse: collapse;
}}

table th {{
    background: {THEME["panel_alt"]};
    color: {THEME["muted"]};
}}

table td, table th {{
    border-bottom: 1px solid {THEME["grid"]};
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {THEME["grid"]};
    border-radius: 10px;
    overflow: hidden;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def sync_state_value(target_key: str, source_key: str) -> None:
    st.session_state[target_key] = st.session_state[source_key]
