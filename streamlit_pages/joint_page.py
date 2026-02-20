from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from joint_character_weapon_dp import (
    calculate_joint_character_weapon_probability,
    make_joint_scenario_key,
)

PRECOMPUTED_DATASET_PATH = Path("precomputed/joint_results_dataset")
LEGACY_PRECOMPUTED_PATH = Path("precomputed/joint_results.parquet")
FIXED_KEEP_LEGACY_BONUS_RULES = True
FIXED_DROP_THRESHOLD = 1e-10


@st.cache_data(show_spinner=False)
def load_precomputed_results(path: str | None = None) -> dict:
    paths: list[Path]
    if path:
        paths = [Path(path)]
    else:
        paths = [LEGACY_PRECOMPUTED_PATH, PRECOMPUTED_DATASET_PATH]

    tables: list[pd.DataFrame] = []
    for file_path in paths:
        if not file_path.exists():
            continue
        table = pd.read_parquet(file_path)
        if not table.empty:
            tables.append(table)

    if not tables:
        return {"meta": {}, "results": {}}

    if len(tables) == 1:
        table = tables[0]
    else:
        table = pd.concat(tables, ignore_index=True)
        table = table.drop_duplicates(subset=["scenario_key"], keep="last")

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


def _get_result_for_pulls(
    pulls: int,
    target_up_characters: int,
    target_up_weapons: int,
    precomputed_results: dict,
) -> tuple[dict, bool]:
    key = make_joint_scenario_key(
        character_pulls=int(pulls),
        target_up_characters=int(target_up_characters),
        target_up_weapons=int(target_up_weapons),
        initial_six_pity=0,
        initial_five_pity=0,
        keep_legacy_bonus_rules=FIXED_KEEP_LEGACY_BONUS_RULES,
        drop_threshold=FIXED_DROP_THRESHOLD,
    )

    hit = precomputed_results.get("results", {}).get(key)
    if hit:
        return hit["result"], True

    result = calculate_joint_cached(
        character_pulls=int(pulls),
        target_up_characters=int(target_up_characters),
        target_up_weapons=int(target_up_weapons),
        initial_six_pity=0,
        initial_five_pity=0,
        keep_legacy_bonus_rules=FIXED_KEEP_LEGACY_BONUS_RULES,
        drop_threshold=FIXED_DROP_THRESHOLD,
    )
    return result, False


def render_joint_page() -> None:
    with st.sidebar:
        character_pulls = st.number_input(
            "角色抽数 X", min_value=1, max_value=720, value=120, step=1
        )
        target_up_characters = st.slider(
            "角色目标 a", min_value=1, max_value=6, value=1
        )
        target_up_weapons = st.slider("武器目标 b", min_value=1, max_value=6, value=1)

    st.subheader("联合目标（角色 + 武器）")

    scenario_key = make_joint_scenario_key(
        character_pulls=int(character_pulls),
        target_up_characters=int(target_up_characters),
        target_up_weapons=int(target_up_weapons),
        initial_six_pity=0,
        initial_five_pity=0,
        keep_legacy_bonus_rules=FIXED_KEEP_LEGACY_BONUS_RULES,
        drop_threshold=FIXED_DROP_THRESHOLD,
    )

    precomputed = load_precomputed_results()
    precomputed_hit = precomputed.get("results", {}).get(scenario_key)

    if precomputed_hit:
        result = precomputed_hit["result"]
    else:
        result = calculate_joint_cached(
            character_pulls=int(character_pulls),
            target_up_characters=int(target_up_characters),
            target_up_weapons=int(target_up_weapons),
            initial_six_pity=0,
            initial_five_pity=0,
            keep_legacy_bonus_rules=FIXED_KEEP_LEGACY_BONUS_RULES,
            drop_threshold=FIXED_DROP_THRESHOLD,
        )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "达成角色目标概率", f"{result['character_only_probability'] * 100:.6f}%"
    )
    col2.metric("达成武器目标概率", f"{result['weapon_only_probability'] * 100:.6f}%")
    col3.metric("综合概率", f"{result['final_probability'] * 100:.6f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("6星期望", f"{result['expected_six_star_count']:.6f}")
    col5.metric("武器配额期望", f"{result['expected_weapon_quota']:.6f}")
    col6.metric("武器十连期望", f"{result['expected_weapon_ten_pulls']:.6f}")

    st.subheader("目标综合概率反算角色抽数")
    target_joint_prob_percent = st.number_input(
        "目标综合概率(%)", min_value=0.0, max_value=99.99, value=50.0, step=0.01
    )
    if st.button("反算最小角色抽数", use_container_width=True):
        target_joint_prob = float(target_joint_prob_percent) / 100.0
        found_pulls: int | None = None

        for pulls in range(1, 721):
            prob_result, from_precomputed = _get_result_for_pulls(
                pulls=pulls,
                target_up_characters=int(target_up_characters),
                target_up_weapons=int(target_up_weapons),
                precomputed_results=precomputed,
            )

            if float(prob_result["final_probability"]) >= target_joint_prob:
                found_pulls = pulls
                break

        if found_pulls is None:
            st.warning("在 1-720 抽范围内未达到该综合概率目标。")
        else:
            st.success(f"最小角色抽数 = {found_pulls}")

    st.divider()
    st.subheader("数学模型与规则说明")
    st.markdown(
        """
- **模型**：角色部分采用稀疏 DP，状态为 `(6星水位, 5星水位, 已获UP角色数, 武器配额)`；武器部分通过独立武器 DP 的 CDF 做映射。
- **角色概率层**：6 星概率规则与干员页一致（0.8% 基础、66 后线性抬升、80 抽小保底为“抽卡出货”）；6 星中 UP 占 50%；5 星基础 8% 且 10 抽内保底。
- **保底与赠送区分**：120 抽首发大保底是“抽卡出货”（若此前 0 UP，则该次按补出 1 个 UP 处理）；240 抽循环是“额外赠送”UP，不是抽卡出货。
- **配额折算**：只有“抽卡出货”会累积武器配额：6 星/5 星/4 星分别 `+100/+10/+1` 单位；240 抽额外赠送 UP 不产生额外配额。最终按 `quota/1980` 折算为武器 10 连次数。
- **联合达成**：对每个终态质量，先判断角色目标是否达成，再乘以“对应武器 10 连次数下达成武器目标”的累计概率，最后加总得到综合概率。
- **额外规则与前提**：当前页面固定初始水位为 0，启用 30/120/240 规则（其中 30 为额外结算、120 为抽卡保底、240 为额外赠送），使用阈值裁剪加速；优先读取预计算结果，未命中时实时精确计算。
"""
    )
