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


def _apply_up_bonus_binomial_layered(
    state_map: dict[tuple[int, int, int], dict[int, float]], up_cap: int
) -> dict[tuple[int, int, int], dict[int, float]]:
    p_bonus = 0.004
    binom = [
        math.comb(10, k) * (p_bonus**k) * ((1.0 - p_bonus) ** (10 - k))
        for k in range(11)
    ]

    new_map: dict[tuple[int, int, int], dict[int, float]] = {}
    for (p6, p5, up), quota_mass_map in state_map.items():
        for quota_units, mass in quota_mass_map.items():
            if mass == 0.0:
                continue
            for k, pk in enumerate(binom):
                if pk == 0.0:
                    continue
                up2 = min(up_cap, up + k)
                key = (p6, p5, up2)
                key_map = new_map.get(key)
                if key_map is None:
                    new_map[key] = {quota_units: mass * pk}
                else:
                    key_map[quota_units] = key_map.get(quota_units, 0.0) + mass * pk
    return new_map


_STATE_UP_BITS = 4
_STATE_P5_BITS = 4
_STATE_UP_MASK = (1 << _STATE_UP_BITS) - 1
_STATE_P5_MASK = (1 << _STATE_P5_BITS) - 1


def _encode_state_id(p6: int, p5: int, up: int) -> int:
    return (int(p6) << (_STATE_P5_BITS + _STATE_UP_BITS)) | (
        int(p5) << _STATE_UP_BITS
    ) | int(up)


def _decode_state_id(state_id: int) -> tuple[int, int, int]:
    up = state_id & _STATE_UP_MASK
    p5 = (state_id >> _STATE_UP_BITS) & _STATE_P5_MASK
    p6 = state_id >> (_STATE_P5_BITS + _STATE_UP_BITS)
    return p6, p5, up


def _apply_up_bonus_binomial_layered_encoded(
    state_map: dict[int, dict[int, float]], up_cap: int
) -> dict[int, dict[int, float]]:
    p_bonus = 0.004
    binom = [
        math.comb(10, k) * (p_bonus**k) * ((1.0 - p_bonus) ** (10 - k))
        for k in range(11)
    ]

    new_map: dict[int, dict[int, float]] = {}
    for state_id, quota_mass_map in state_map.items():
        p6, p5, up = _decode_state_id(state_id)
        for quota_units, mass in quota_mass_map.items():
            if mass == 0.0:
                continue
            for k, pk in enumerate(binom):
                if pk == 0.0:
                    continue
                up2 = min(up_cap, up + k)
                next_id = _encode_state_id(p6, p5, up2)
                key_map = new_map.get(next_id)
                if key_map is None:
                    new_map[next_id] = {quota_units: mass * pk}
                else:
                    key_map[quota_units] = key_map.get(quota_units, 0.0) + mass * pk
    return new_map


def _effective_drop_threshold(base_threshold: float, draw: int) -> float:
    if base_threshold <= 0.0 or draw <= 120:
        return base_threshold
    scale = 1.0 + 0.2 * (draw - 120)
    return min(1e-8, base_threshold * scale)


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

    transition_cache: dict[
        int, tuple[float, float, float, float, int, int, int, int]
    ] = {}
    for p6 in range(81):
        for p5 in range(10):
            p6_up, p6_off, p5_star, p4_star = _normal_tier_probs(p6, p5)
            p6_next = min(80, p6 + 1)
            p5_next = min(9, p5 + 1)
            for up in range(up_cap + 1):
                state_id = _encode_state_id(p6, p5, up)
                transition_cache[state_id] = (
                    p6_up,
                    p6_off,
                    p5_star,
                    p4_star,
                    _encode_state_id(0, 0, min(up_cap, up + 1)),
                    _encode_state_id(0, 0, up),
                    _encode_state_id(p6_next, 0, up),
                    _encode_state_id(p6_next, p5_next, up),
                )

    init_state_id = _encode_state_id(initial_six_pity, initial_five_pity, 0)
    states: dict[int, dict[int, float]] = {init_state_id: {0: 1.0}}
    expected_six_star_count = 0.0

    for draw in range(1, character_pulls + 1):
        next_states: dict[int, dict[int, float]] = {}

        for state_id, quota_mass_map in states.items():
            (
                p6_up,
                p6_off,
                p5_star,
                p4_star,
                next_id_6_up,
                next_id_6_off,
                next_id_5,
                next_id_4,
            ) = transition_cache[state_id]

            for quota_units, mass in quota_mass_map.items():
                if mass <= 0.0:
                    continue

                expected_six_star_count += mass * (p6_up + p6_off)

                if p6_up > 0.0:
                    key_map = next_states.get(next_id_6_up)
                    q2 = quota_units + 100
                    if key_map is None:
                        next_states[next_id_6_up] = {q2: mass * p6_up}
                    else:
                        key_map[q2] = key_map.get(q2, 0.0) + mass * p6_up

                if p6_off > 0.0:
                    key_map = next_states.get(next_id_6_off)
                    q2 = quota_units + 100
                    if key_map is None:
                        next_states[next_id_6_off] = {q2: mass * p6_off}
                    else:
                        key_map[q2] = key_map.get(q2, 0.0) + mass * p6_off

                if p5_star > 0.0:
                    key_map = next_states.get(next_id_5)
                    q2 = quota_units + 10
                    if key_map is None:
                        next_states[next_id_5] = {q2: mass * p5_star}
                    else:
                        key_map[q2] = key_map.get(q2, 0.0) + mass * p5_star

                if p4_star > 0.0:
                    key_map = next_states.get(next_id_4)
                    q2 = quota_units + 1
                    if key_map is None:
                        next_states[next_id_4] = {q2: mass * p4_star}
                    else:
                        key_map[q2] = key_map.get(q2, 0.0) + mass * p4_star

        if keep_legacy_bonus_rules and draw == 30:
            next_states = _apply_up_bonus_binomial_layered_encoded(
                next_states, up_cap=up_cap
            )

        if keep_legacy_bonus_rules and draw == 120:
            adjusted: dict[int, dict[int, float]] = {}
            for state_id, quota_mass_map in next_states.items():
                p6, p5, up = _decode_state_id(state_id)
                for quota_units, mass in quota_mass_map.items():
                    if up == 0:
                        expected_six_star_count += mass
                        next_id = _encode_state_id(0, 0, min(up_cap, 1))
                        q2 = quota_units + 100
                    else:
                        next_id = state_id
                        q2 = quota_units
                    key_map = adjusted.get(next_id)
                    if key_map is None:
                        adjusted[next_id] = {q2: mass}
                    else:
                        key_map[q2] = key_map.get(q2, 0.0) + mass
            next_states = adjusted

        if keep_legacy_bonus_rules and draw % 240 == 0:
            adjusted: dict[int, dict[int, float]] = {}
            for state_id, quota_mass_map in next_states.items():
                p6, p5, up = _decode_state_id(state_id)
                next_id = _encode_state_id(p6, p5, min(up_cap, up + 1))
                key_map = adjusted.get(next_id)
                if key_map is None:
                    adjusted[next_id] = dict(quota_mass_map)
                else:
                    for quota_units, mass in quota_mass_map.items():
                        key_map[quota_units] = key_map.get(quota_units, 0.0) + mass
            next_states = adjusted

        effective_threshold = _effective_drop_threshold(drop_threshold, draw)

        states = {}
        for key, quota_mass_map in next_states.items():
            filtered_map = {
                quota_units: mass
                for quota_units, mass in quota_mass_map.items()
                if mass >= effective_threshold
            }
            if filtered_map:
                states[key] = filtered_map

    total_mass = float(sum(sum(m.values()) for m in states.values()))

    if target_up_weapons <= 0:
        weapon_cdf = (1.0,)
        max_weapon_ten_pulls = 0
    else:
        max_quota_units = 0
        for quota_mass_map in states.values():
            if quota_mass_map:
                local_max = max(quota_mass_map.keys())
                if local_max > max_quota_units:
                    max_quota_units = local_max
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

    for state_id, quota_mass_map in states.items():
        _, _, up = _decode_state_id(state_id)
        char_success = up >= target_up_characters
        for quota_units, mass in quota_mass_map.items():
            weapon_ten_pulls = _weapon_ten_pulls_from_quota_units(quota_units)
            expected_weapon_ten_pulls += mass * weapon_ten_pulls
            expected_weapon_quota += mass * quota_units * 20

            if target_up_weapons <= 0:
                weapon_only_probability += mass
                if char_success:
                    final_probability += mass
            else:
                weapon_success = mass * float(weapon_cdf[weapon_ten_pulls])
                weapon_only_probability += weapon_success
                if char_success:
                    final_probability += weapon_success

            if char_success:
                character_only_probability += mass

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
