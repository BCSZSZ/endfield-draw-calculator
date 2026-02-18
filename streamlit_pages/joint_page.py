from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from joint_character_weapon_dp import (
    calculate_joint_character_weapon_probability,
    make_joint_scenario_key,
)

PRECOMPUTED_PATH = Path("precomputed/joint_results.parquet")


@st.cache_data(show_spinner=False)
def load_precomputed_results(path: str = str(PRECOMPUTED_PATH)) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"meta": {}, "results": {}}

    table = pd.read_parquet(file_path)
    if table.empty:
        return {"meta": {}, "results": {}}

    results: dict[str, dict] = {}
    for row in table.to_dict(orient="records"):
        key = row["scenario_key"]
        results[key] = {
            "scenario": {
                "character_pulls": int(row["character_pulls"]),
                "target_up_characters": int(row["target_up_characters"]),
                "target_up_weapons": int(row["target_up_weapons"]),
                "initial_six_pity": int(row["initial_six_pity"]),
                "initial_five_pity": int(row["initial_five_pity"]),
                "keep_legacy_bonus_rules": bool(row["keep_legacy_bonus_rules"]),
                "drop_threshold": float(row["drop_threshold"]),
            },
            "result": {
                "final_probability": float(row["final_probability"]),
                "character_only_probability": float(row["character_only_probability"]),
                "weapon_only_probability": float(row["weapon_only_probability"]),
                "expected_six_star_count": float(row["expected_six_star_count"]),
                "expected_weapon_quota": float(row["expected_weapon_quota"]),
                "expected_weapon_ten_pulls": float(row["expected_weapon_ten_pulls"]),
                "max_weapon_ten_pulls": int(row["max_weapon_ten_pulls"]),
                "state_mass": float(row["state_mass"]),
            },
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "created_at": row["created_at"],
        }

    return {"meta": {}, "results": results}


@st.cache_data(show_spinner=True)
def calculate_joint_cached(
    character_pulls: int,
    target_up_characters: int,
    target_up_weapons: int,
    initial_six_pity: int,
    initial_five_pity: int,
    keep_legacy_bonus_rules: bool,
    drop_threshold: float,
) -> dict:
    result = calculate_joint_character_weapon_probability(
        character_pulls=character_pulls,
        target_up_characters=target_up_characters,
        target_up_weapons=target_up_weapons,
        initial_six_pity=initial_six_pity,
        initial_five_pity=initial_five_pity,
        keep_legacy_bonus_rules=keep_legacy_bonus_rules,
        drop_threshold=drop_threshold,
    )
    return result.as_dict()


def render_joint_page() -> None:
    with st.sidebar:
        character_pulls = st.number_input(
            "角色抽数 X", min_value=1, max_value=720, value=120, step=1
        )
        target_up_characters = st.slider("角色目标 a", min_value=1, max_value=6, value=1)
        target_up_weapons = st.slider("武器目标 b", min_value=1, max_value=6, value=1)
        keep_legacy_bonus_rules = st.checkbox("保留旧额外规则", value=True)
        drop_threshold = st.select_slider(
            "概率裁剪阈值",
            options=[1e-12, 1e-14, 1e-16, 1e-18],
            value=1e-16,
            format_func=lambda x: f"{x:.0e}",
        )

    st.subheader("联合目标（角色 + 武器）")
    st.caption("优先读取预计算结果；未命中时自动使用缓存计算。初始 pity 固定为 0/0。")

    scenario_key = make_joint_scenario_key(
        character_pulls=int(character_pulls),
        target_up_characters=int(target_up_characters),
        target_up_weapons=int(target_up_weapons),
        initial_six_pity=0,
        initial_five_pity=0,
        keep_legacy_bonus_rules=bool(keep_legacy_bonus_rules),
        drop_threshold=float(drop_threshold),
    )

    precomputed = load_precomputed_results()
    precomputed_hit = precomputed.get("results", {}).get(scenario_key)

    if precomputed_hit:
        result = precomputed_hit["result"]
        st.success("命中预计算结果（秒级展示）")
    else:
        st.warning("未命中预计算，正在执行实时计算（参数较大时可能较慢）")
        result = calculate_joint_cached(
            character_pulls=int(character_pulls),
            target_up_characters=int(target_up_characters),
            target_up_weapons=int(target_up_weapons),
            initial_six_pity=0,
            initial_five_pity=0,
            keep_legacy_bonus_rules=bool(keep_legacy_bonus_rules),
            drop_threshold=float(drop_threshold),
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("达成角色目标概率", f"{result['character_only_probability'] * 100:.6f}%")
    col2.metric("达成武器目标概率", f"{result['weapon_only_probability'] * 100:.6f}%")
    col3.metric("综合概率", f"{result['final_probability'] * 100:.6f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("6星期望", f"{result['expected_six_star_count']:.6f}")
    col5.metric("武器配额期望", f"{result['expected_weapon_quota']:.6f}")
    col6.metric("武器十连期望", f"{result['expected_weapon_ten_pulls']:.6f}")

    st.write(f"状态总质量：{result['state_mass']:.12f}")
