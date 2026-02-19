from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from weapon_draw import calculate_weapon_full_potential


@dataclass
class JointProbabilityResult:
    final_probability: float
    character_only_probability: float
    weapon_only_probability: float
    expected_six_star_count: float
    expected_weapon_quota: float
    expected_weapon_ten_pulls: float
    max_weapon_ten_pulls: int
    state_mass: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "final_probability": self.final_probability,
            "character_only_probability": self.character_only_probability,
            "weapon_only_probability": self.weapon_only_probability,
            "expected_six_star_count": self.expected_six_star_count,
            "expected_weapon_quota": self.expected_weapon_quota,
            "expected_weapon_ten_pulls": self.expected_weapon_ten_pulls,
            "max_weapon_ten_pulls": self.max_weapon_ten_pulls,
            "state_mass": self.state_mass,
        }


def make_joint_scenario_key(
    character_pulls: int,
    target_up_characters: int,
    target_up_weapons: int,
    initial_six_pity: int = 0,
    initial_five_pity: int = 0,
    keep_legacy_bonus_rules: bool = True,
    drop_threshold: float = 1e-10,
) -> str:
    return "|".join(
        [
            str(int(character_pulls)),
            str(int(target_up_characters)),
            str(int(target_up_weapons)),
            str(int(initial_six_pity)),
            str(int(initial_five_pity)),
            str(int(bool(keep_legacy_bonus_rules))),
            f"{float(drop_threshold):.1e}",
        ]
    )


def _six_star_rate(pity6: int) -> float:
    if pity6 < 66:
        return 0.008
    if pity6 < 80:
        return min(1.0, 0.008 + (pity6 - 65) * 0.05)
    return 1.0


def _build_normal_tier_table() -> tuple[
    tuple[tuple[float, float, float, float], ...], ...
]:
    table: list[tuple[float, float, float, float]] = []
    rows: list[tuple[tuple[float, float, float, float], ...]] = []
    for pity6 in range(81):
        table = []
        p6 = _six_star_rate(pity6)
        p6_up = 0.5 * p6
        p6_off = 0.5 * p6
        for pity5 in range(10):
            if pity5 >= 9:
                p5 = 1.0 - p6
                p4 = 0.0
            else:
                p5_base = 0.08
                p5 = min(p5_base, 1.0 - p6)
                p4 = max(0.0, 1.0 - p6 - p5)
            table.append((p6_up, p6_off, p5, p4))
        rows.append(tuple(table))
    return tuple(rows)


_NORMAL_TIER_TABLE = _build_normal_tier_table()


def _normal_tier_probs(pity6: int, pity5: int) -> tuple[float, float, float, float]:
    if 0 <= pity6 <= 80 and 0 <= pity5 <= 9:
        return _NORMAL_TIER_TABLE[pity6][pity5]

    p6 = _six_star_rate(pity6)
    p6_up = 0.5 * p6
    p6_off = 0.5 * p6

    if pity5 >= 9:
        p5 = 1.0 - p6
        p4 = 0.0
        return p6_up, p6_off, p5, p4

    p5_base = 0.08
    p5 = min(p5_base, 1.0 - p6)
    p4 = max(0.0, 1.0 - p6 - p5)
    return p6_up, p6_off, p5, p4


def _weapon_ten_pulls_from_quota_units(quota_units: int) -> int:
    quota = int(quota_units) * 20
    full = quota // 1980
    remainder = quota % 1980
    first_decimal_digit = (remainder * 10) // 1980
    if first_decimal_digit == 9:
        return full + 1
    return full


def _apply_up_bonus_binomial(
    state_map: dict[tuple[int, int, int, int], float], up_cap: int
) -> dict[tuple[int, int, int, int], float]:
    p_bonus = 0.004
    binom = [
        math.comb(10, k) * (p_bonus**k) * ((1.0 - p_bonus) ** (10 - k))
        for k in range(11)
    ]

    new_map: dict[tuple[int, int, int, int], float] = {}
    for (p6, p5, up, quota_units), mass in state_map.items():
        if mass == 0.0:
            continue
        for k, pk in enumerate(binom):
            if pk == 0.0:
                continue
            up2 = min(up_cap, up + k)
            key = (p6, p5, up2, quota_units)
            new_map[key] = new_map.get(key, 0.0) + mass * pk
    return new_map


@lru_cache(maxsize=256)
def _get_weapon_cdf(
    target_up_weapons: int, max_weapon_ten_pulls: int
) -> tuple[float, ...]:
    if target_up_weapons <= 0:
        return (1.0,)

    weapon_result = calculate_weapon_full_potential(
        target_copies=target_up_weapons,
        max_sim_pulls=max_weapon_ten_pulls,
    )
    cdf = weapon_result["finish_probs"].cumsum()
    return tuple(float(x) for x in cdf)


def calculate_joint_character_weapon_probability(
    character_pulls: int,
    target_up_characters: int,
    target_up_weapons: int,
    initial_six_pity: int = 0,
    initial_five_pity: int = 0,
    keep_legacy_bonus_rules: bool = True,
    drop_threshold: float = 1e-10,
) -> JointProbabilityResult:
    character_pulls = max(0, int(character_pulls))
    target_up_characters = max(0, int(target_up_characters))
    target_up_weapons = max(0, int(target_up_weapons))
    initial_six_pity = max(0, min(80, int(initial_six_pity)))
    initial_five_pity = max(0, min(9, int(initial_five_pity)))

    up_cap = max(1, target_up_characters)

    states: dict[tuple[int, int, int, int], float] = {
        (initial_six_pity, initial_five_pity, 0, 0): 1.0
    }
    expected_six_star_count = 0.0

    for draw in range(1, character_pulls + 1):
        next_states: dict[tuple[int, int, int, int], float] = {}

        for (p6, p5, up, quota_units), mass in states.items():
            if mass <= 0.0:
                continue

            p6_up, p6_off, p5_star, p4_star = _normal_tier_probs(p6, p5)
            expected_six_star_count += mass * (p6_up + p6_off)

            if p6_up > 0.0:
                key = (0, 0, min(up_cap, up + 1), quota_units + 100)
                next_states[key] = next_states.get(key, 0.0) + mass * p6_up

            if p6_off > 0.0:
                key = (0, 0, up, quota_units + 100)
                next_states[key] = next_states.get(key, 0.0) + mass * p6_off

            if p5_star > 0.0:
                key = (min(80, p6 + 1), 0, up, quota_units + 10)
                next_states[key] = next_states.get(key, 0.0) + mass * p5_star

            if p4_star > 0.0:
                key = (min(80, p6 + 1), min(9, p5 + 1), up, quota_units + 1)
                next_states[key] = next_states.get(key, 0.0) + mass * p4_star

        if keep_legacy_bonus_rules and draw == 30:
            next_states = _apply_up_bonus_binomial(next_states, up_cap=up_cap)

        if keep_legacy_bonus_rules and draw == 120:
            adjusted: dict[tuple[int, int, int, int], float] = {}
            for (p6, p5, up, quota_units), mass in next_states.items():
                if up == 0:
                    expected_six_star_count += mass
                    key = (0, 0, min(up_cap, 1), quota_units + 100)
                else:
                    key = (p6, p5, up, quota_units)
                adjusted[key] = adjusted.get(key, 0.0) + mass
            next_states = adjusted

        if keep_legacy_bonus_rules and draw % 240 == 0:
            adjusted: dict[tuple[int, int, int, int], float] = {}
            for (p6, p5, up, quota_units), mass in next_states.items():
                key = (p6, p5, min(up_cap, up + 1), quota_units)
                adjusted[key] = adjusted.get(key, 0.0) + mass
            next_states = adjusted

        states = {
            key: value for key, value in next_states.items() if value >= drop_threshold
        }

    total_mass = float(sum(states.values()))

    if target_up_weapons <= 0:
        weapon_cdf = (1.0,)
        max_weapon_ten_pulls = 0
    else:
        max_quota_units = max((q for (_, _, _, q) in states.keys()), default=0)
        max_weapon_ten_pulls = _weapon_ten_pulls_from_quota_units(max_quota_units)
        weapon_cdf = _get_weapon_cdf(
            target_up_weapons=target_up_weapons,
            max_weapon_ten_pulls=max_weapon_ten_pulls,
        )

    final_probability = 0.0
    character_only_probability = 0.0
    weapon_only_probability = 0.0
    expected_weapon_ten_pulls = 0.0
    expected_weapon_quota = 0.0

    for (_, _, up, quota_units), mass in states.items():
        weapon_ten_pulls = _weapon_ten_pulls_from_quota_units(quota_units)
        expected_weapon_ten_pulls += mass * weapon_ten_pulls
        expected_weapon_quota += mass * quota_units * 20

        if target_up_weapons <= 0:
            weapon_only_probability += mass
        else:
            weapon_only_probability += mass * float(weapon_cdf[weapon_ten_pulls])

        char_success = up >= target_up_characters
        if char_success:
            character_only_probability += mass
            if target_up_weapons <= 0:
                final_probability += mass
            else:
                final_probability += mass * float(weapon_cdf[weapon_ten_pulls])

    return JointProbabilityResult(
        final_probability=final_probability,
        character_only_probability=character_only_probability,
        weapon_only_probability=weapon_only_probability,
        expected_six_star_count=expected_six_star_count,
        expected_weapon_quota=expected_weapon_quota,
        expected_weapon_ten_pulls=expected_weapon_ten_pulls,
        max_weapon_ten_pulls=max_weapon_ten_pulls,
        state_mass=total_mass,
    )


if __name__ == "__main__":
    result = calculate_joint_character_weapon_probability(
        character_pulls=120,
        target_up_characters=1,
        target_up_weapons=1,
        initial_six_pity=0,
        initial_five_pity=0,
        keep_legacy_bonus_rules=True,
    )

    expected_quota_div_1980 = result.expected_weapon_quota / 1980.0

    print("=== 联合达成概率 ===")
    print(f"达成角色目标的概率 = {result.character_only_probability * 100:.6f}%")
    print(f"获得的6星个数期望 = {result.expected_six_star_count:.6f}")
    print(f"获得的武器配额期望 = {result.expected_weapon_quota:.6f}")
    print(f"配额期望/1980 = {expected_quota_div_1980:.6f}")
    print(f"期望武器十连次数 = {result.expected_weapon_ten_pulls:.6f}")
    print(f"达成武器目标的概率 = {result.weapon_only_probability * 100:.6f}%")
    print(f"综合概率 = {result.final_probability * 100:.6f}%")
    print(f"状态总质量 = {result.state_mass:.12f}")
