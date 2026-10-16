from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from draw_model import MODEL_VERSION
from joint_model import build_joint_history
from joint_character_weapon_dp import make_joint_scenario_key


OUTPUT_PATH = Path("precomputed/joint_results_dataset")

DEFAULT_SCENARIOS = [
    {
        "character_pulls": 120,
        "target_up_characters": 1,
        "target_up_weapons": 1,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-10,
    },
    {
        "character_pulls": 720,
        "target_up_characters": 6,
        "target_up_weapons": 6,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-10,
    },
    {
        "character_pulls": 240,
        "target_up_characters": 1,
        "target_up_weapons": 1,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-10,
    },
    {
        "character_pulls": 360,
        "target_up_characters": 2,
        "target_up_weapons": 2,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-10,
    },
]


def build_grid_scenarios(
    min_pulls: int = 1, max_pulls: int = 720, min_char_target: int = 1,
    max_char_target: int = 6, min_weapon_target: int = 1, max_weapon_target: int = 6,
    keep_legacy_bonus_rules: bool = True, drop_threshold: float = 1e-10,
) -> list[dict]:
    return [dict(
        character_pulls=n, target_up_characters=a, target_up_weapons=b,
        initial_six_pity=0, initial_five_pity=0,
        keep_legacy_bonus_rules=keep_legacy_bonus_rules, drop_threshold=drop_threshold,
    ) for n in range(min_pulls, max_pulls + 1)
      for a in range(min_char_target, max_char_target + 1)
      for b in range(min_weapon_target, max_weapon_target + 1)]


def run_precompute(
    scenarios: list[dict] | None = None, output_path: Path = OUTPUT_PATH,
    flush_every: int = 25920,
) -> None:
    scenarios = scenarios or DEFAULT_SCENARIOS
    groups: dict[tuple, list[dict]] = {}
    for scenario in scenarios:
        group = tuple(scenario[key] for key in (
            "initial_six_pity", "initial_five_pity", "keep_legacy_bonus_rules", "drop_threshold",
        ))
        groups.setdefault(group, []).append(scenario)
    output_path.mkdir(parents=True, exist_ok=True)
    for group_index, (settings, group) in enumerate(groups.items()):
        started = time.perf_counter()
        p6, p5, rewards, tolerance = settings
        history = build_joint_history(
            max(s["character_pulls"] for s in group),
            copies_cap=max(1, max(s["target_up_characters"] for s in group)),
            initial_six_pity=p6, initial_five_pity=p5, include_rewards=rewards,
            tail_tolerance=tolerance,
            progress=lambda n, mass: print(f"[draw {n}] state mass={mass:.12f}", flush=True),
        )
        rows = []
        for scenario in group:
            n, a, b = (scenario[k] for k in (
                "character_pulls", "target_up_characters", "target_up_weapons",
            ))
            six, quota, tens, max_tens, mass = history.metrics[n]
            probabilities = history.probabilities[n]
            rows.append(dict(
                scenario_key=make_joint_scenario_key(**scenario), **scenario,
                final_probability=float(probabilities[a, b]),
                character_only_probability=float(probabilities[a, 0]),
                weapon_only_probability=float(probabilities[0, b]),
                expected_six_star_count=float(six), expected_weapon_quota=float(quota),
                expected_weapon_ten_pulls=float(tens), max_weapon_ten_pulls=int(max_tens),
                state_mass=float(mass), elapsed_seconds=time.perf_counter() - started,
                created_at=datetime.now(timezone.utc).isoformat(), generator=Path(__file__).name,
            ))
        for chunk, offset in enumerate(range(0, len(rows), max(1, flush_every))):
            path = output_path / f"v{MODEL_VERSION}-{group_index:03}-{chunk:03}.parquet"
            pq.write_table(pa.Table.from_pylist(rows[offset:offset + max(1, flush_every)]), path)
            print(f"Saved {path}", flush=True)


def run_precompute_grid_incremental(
    min_pulls: int = 1, max_pulls: int = 720, min_char_target: int = 1,
    max_char_target: int = 6, min_weapon_target: int = 1, max_weapon_target: int = 6,
    keep_legacy_bonus_rules: bool = True, drop_threshold: float = 1e-10,
    output_path: Path = OUTPUT_PATH, flush_every: int = 25920,
) -> None:
    run_precompute(build_grid_scenarios(
        min_pulls, max_pulls, min_char_target, max_char_target,
        min_weapon_target, max_weapon_target, keep_legacy_bonus_rules, drop_threshold,
    ), output_path, flush_every)


if __name__ == "__main__":
    run_precompute_grid_incremental()
