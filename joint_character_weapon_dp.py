from __future__ import annotations

from dataclasses import dataclass

from draw_model import MODEL_VERSION, six_star_rate
from joint_model import build_joint_history


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
    character_pulls: int, target_up_characters: int, target_up_weapons: int,
    initial_six_pity: int = 0, initial_five_pity: int = 0,
    keep_legacy_bonus_rules: bool = True, drop_threshold: float = 1e-10,
) -> str:
    return "|".join([
        MODEL_VERSION, str(int(character_pulls)), str(int(target_up_characters)),
        str(int(target_up_weapons)), str(int(initial_six_pity)),
        str(int(initial_five_pity)), str(int(bool(keep_legacy_bonus_rules))),
        f"{float(drop_threshold):.1e}",
    ])


def _six_star_rate(pity6: int) -> float:
    return six_star_rate(pity6)


def _weapon_ten_pulls_from_quota_units(quota_units: int) -> int:
    return int(quota_units) // 99


def calculate_joint_character_weapon_probability(
    character_pulls: int, target_up_characters: int, target_up_weapons: int,
    initial_six_pity: int = 0, initial_five_pity: int = 0,
    keep_legacy_bonus_rules: bool = True, drop_threshold: float = 1e-10,
) -> JointProbabilityResult:
    pulls = max(0, int(character_pulls))
    char_target = max(0, int(target_up_characters))
    weapon_target = max(0, min(6, int(target_up_weapons)))
    history = build_joint_history(
        pulls, copies_cap=max(1, char_target),
        initial_six_pity=max(0, min(79, int(initial_six_pity))),
        initial_five_pity=max(0, min(9, int(initial_five_pity))),
        include_rewards=keep_legacy_bonus_rules, tail_tolerance=drop_threshold,
    )
    six, quota, tens, max_tens, mass = history.metrics[-1]
    probabilities = history.probabilities[-1]
    return JointProbabilityResult(
        final_probability=float(probabilities[char_target, weapon_target]),
        character_only_probability=float(probabilities[char_target, 0]),
        weapon_only_probability=float(probabilities[0, weapon_target]),
        expected_six_star_count=float(six), expected_weapon_quota=float(quota),
        expected_weapon_ten_pulls=float(tens), max_weapon_ten_pulls=int(max_tens),
        state_mass=float(mass),
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
