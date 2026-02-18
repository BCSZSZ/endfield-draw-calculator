from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from joint_character_weapon_dp import (
    _apply_up_bonus_binomial,
    _get_weapon_cdf,
    _normal_tier_probs,
    _weapon_ten_pulls_from_quota_units,
    calculate_joint_character_weapon_probability,
    make_joint_scenario_key,
)


OUTPUT_PATH = Path("precomputed/joint_results.parquet")

DEFAULT_SCENARIOS = [
    {
        "character_pulls": 120,
        "target_up_characters": 1,
        "target_up_weapons": 1,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-16,
    },
    {
        "character_pulls": 720,
        "target_up_characters": 6,
        "target_up_weapons": 6,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-16,
    },
    {
        "character_pulls": 240,
        "target_up_characters": 1,
        "target_up_weapons": 1,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-16,
    },
    {
        "character_pulls": 360,
        "target_up_characters": 2,
        "target_up_weapons": 2,
        "initial_six_pity": 0,
        "initial_five_pity": 0,
        "keep_legacy_bonus_rules": True,
        "drop_threshold": 1e-16,
    },
]


def _normalize_scenario_fixed_pity(scenario: dict) -> dict:
    normalized = dict(scenario)
    normalized["initial_six_pity"] = 0
    normalized["initial_five_pity"] = 0
    return normalized


def build_grid_scenarios(
    min_pulls: int = 1,
    max_pulls: int = 720,
    min_char_target: int = 1,
    max_char_target: int = 6,
    min_weapon_target: int = 1,
    max_weapon_target: int = 6,
    keep_legacy_bonus_rules: bool = True,
    drop_threshold: float = 1e-16,
) -> list[dict]:
    scenarios: list[dict] = []
    for pulls in range(int(min_pulls), int(max_pulls) + 1):
        for char_target in range(int(min_char_target), int(max_char_target) + 1):
            for weapon_target in range(int(min_weapon_target), int(max_weapon_target) + 1):
                scenarios.append(
                    {
                        "character_pulls": pulls,
                        "target_up_characters": char_target,
                        "target_up_weapons": weapon_target,
                        "initial_six_pity": 0,
                        "initial_five_pity": 0,
                        "keep_legacy_bonus_rules": keep_legacy_bonus_rules,
                        "drop_threshold": drop_threshold,
                    }
                )
    return scenarios


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def _build_row(scenario: dict, key: str, elapsed: float, result_dict: dict) -> dict:
    return {
        "scenario_key": key,
        "character_pulls": int(scenario["character_pulls"]),
        "target_up_characters": int(scenario["target_up_characters"]),
        "target_up_weapons": int(scenario["target_up_weapons"]),
        "initial_six_pity": int(scenario["initial_six_pity"]),
        "initial_five_pity": int(scenario["initial_five_pity"]),
        "keep_legacy_bonus_rules": bool(scenario["keep_legacy_bonus_rules"]),
        "drop_threshold": float(scenario["drop_threshold"]),
        "final_probability": float(result_dict["final_probability"]),
        "character_only_probability": float(result_dict["character_only_probability"]),
        "weapon_only_probability": float(result_dict["weapon_only_probability"]),
        "expected_six_star_count": float(result_dict["expected_six_star_count"]),
        "expected_weapon_quota": float(result_dict["expected_weapon_quota"]),
        "expected_weapon_ten_pulls": float(result_dict["expected_weapon_ten_pulls"]),
        "max_weapon_ten_pulls": int(result_dict["max_weapon_ten_pulls"]),
        "state_mass": float(result_dict["state_mass"]),
        "elapsed_seconds": float(elapsed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "precompute_joint_results.py",
    }


def run_precompute(
    scenarios: list[dict] | None = None,
    output_path: Path = OUTPUT_PATH,
    flush_every: int = 100,
) -> None:
    scenarios = scenarios or DEFAULT_SCENARIOS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = load_existing(output_path)
    known_keys = set(table["scenario_key"].tolist()) if not table.empty else set()
    new_rows: list[dict] = []
    total = len(scenarios)
    processed = 0
    started_at = time.perf_counter()
    try:
        for idx, scenario in enumerate(scenarios, start=1):
            normalized_scenario = _normalize_scenario_fixed_pity(scenario)
            key = make_joint_scenario_key(**normalized_scenario)
            if key in known_keys:
                processed += 1
                elapsed_total = time.perf_counter() - started_at
                avg = elapsed_total / processed
                remain = max(0, total - processed) * avg
                print(f"[{idx}/{total}] [skip] {key} | ETA~{remain/60:.1f} min")
                continue

            print(f"[{idx}/{total}] [run ] {key}")
            start = time.perf_counter()
            result = calculate_joint_character_weapon_probability(**normalized_scenario)
            elapsed = time.perf_counter() - start

            row = _build_row(
                scenario=normalized_scenario,
                key=key,
                elapsed=elapsed,
                result_dict=result.as_dict(),
            )
            new_rows.append(row)
            known_keys.add(key)
            processed += 1
            elapsed_total = time.perf_counter() - started_at
            avg = elapsed_total / max(1, processed)
            remain = max(0, total - processed) * avg
            print(f"[{idx}/{total}] [done] {key} ({elapsed:.2f}s) | ETA~{remain/60:.1f} min")

            if flush_every > 0 and len(new_rows) >= int(flush_every):
                checkpoint = pd.DataFrame(new_rows)
                if table.empty:
                    table = checkpoint
                else:
                    table = pd.concat([table, checkpoint], ignore_index=True)
                table.to_parquet(output_path, index=False)
                print(f"[save] checkpoint rows={len(table)}")
                new_rows = []
    except KeyboardInterrupt:
        print("[interrupt] Caught KeyboardInterrupt. Flushing buffered rows to parquet...")
    finally:
        if new_rows:
            new_table = pd.DataFrame(new_rows)
            if table.empty:
                final_table = new_table
            else:
                final_table = pd.concat([table, new_table], ignore_index=True)
            final_table.to_parquet(output_path, index=False)
            print(f"[save] final rows={len(final_table)}")

    print(f"Saved: {output_path}")


def run_precompute_grid_incremental(
    min_pulls: int = 1,
    max_pulls: int = 720,
    min_char_target: int = 1,
    max_char_target: int = 6,
    min_weapon_target: int = 1,
    max_weapon_target: int = 6,
    keep_legacy_bonus_rules: bool = True,
    drop_threshold: float = 1e-16,
    output_path: Path = OUTPUT_PATH,
    flush_every: int = 300,
) -> None:
    min_pulls = int(min_pulls)
    max_pulls = int(max_pulls)
    min_char_target = int(min_char_target)
    max_char_target = int(max_char_target)
    min_weapon_target = int(min_weapon_target)
    max_weapon_target = int(max_weapon_target)

    if min_pulls < 1:
        min_pulls = 1
    if max_pulls < min_pulls:
        raise ValueError("max_pulls must be >= min_pulls")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = load_existing(output_path)
    known_keys = set(table["scenario_key"].tolist()) if not table.empty else set()

    char_targets = list(range(min_char_target, max_char_target + 1))
    weapon_targets = list(range(min_weapon_target, max_weapon_target + 1))

    up_cap = max_char_target
    states: dict[tuple[int, int, int, int], float] = {(0, 0, 0, 0): 1.0}
    expected_six_star_count = 0.0

    max_quota_units_upper = 100 * max_pulls + (100 if keep_legacy_bonus_rules and max_pulls >= 120 else 0)
    max_weapon_ten_pulls_upper = _weapon_ten_pulls_from_quota_units(max_quota_units_upper)
    weapon_cdfs: dict[int, tuple[float, ...]] = {
        b: _get_weapon_cdf(b, max_weapon_ten_pulls_upper) for b in weapon_targets
    }

    total_rows = (max_pulls - min_pulls + 1) * len(char_targets) * len(weapon_targets)
    written_rows = 0
    skipped_rows = 0
    new_rows: list[dict] = []
    started_at = time.perf_counter()

    for draw in range(1, max_pulls + 1):
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

        states = {k: v for k, v in next_states.items() if v >= drop_threshold}

        if draw < min_pulls:
            continue

        total_mass_by_m: dict[int, float] = {}
        char_mass_by_a_m: dict[int, dict[int, float]] = {a: {} for a in char_targets}
        expected_weapon_quota = 0.0
        state_mass = 0.0

        for (_, _, up, quota_units), mass in states.items():
            if mass <= 0.0:
                continue
            m = _weapon_ten_pulls_from_quota_units(quota_units)
            total_mass_by_m[m] = total_mass_by_m.get(m, 0.0) + mass
            expected_weapon_quota += mass * quota_units * 20
            state_mass += mass

            max_a_hit = min(up, max_char_target)
            if max_a_hit >= min_char_target:
                for a in range(min_char_target, max_a_hit + 1):
                    bucket = char_mass_by_a_m[a]
                    bucket[m] = bucket.get(m, 0.0) + mass

        expected_weapon_ten_pulls = sum(m * mass for m, mass in total_mass_by_m.items())
        character_only_probs = {
            a: sum(char_mass_by_a_m[a].values()) for a in char_targets
        }
        weapon_only_probs = {
            b: sum(mass * weapon_cdfs[b][m] for m, mass in total_mass_by_m.items())
            for b in weapon_targets
        }

        for a in char_targets:
            char_masses = char_mass_by_a_m[a]
            for b in weapon_targets:
                scenario = {
                    "character_pulls": draw,
                    "target_up_characters": a,
                    "target_up_weapons": b,
                    "initial_six_pity": 0,
                    "initial_five_pity": 0,
                    "keep_legacy_bonus_rules": keep_legacy_bonus_rules,
                    "drop_threshold": drop_threshold,
                }
                key = make_joint_scenario_key(**scenario)
                if key in known_keys:
                    skipped_rows += 1
                    continue

                final_probability = sum(
                    mass * weapon_cdfs[b][m] for m, mass in char_masses.items()
                )
                row = {
                    "scenario_key": key,
                    "character_pulls": draw,
                    "target_up_characters": a,
                    "target_up_weapons": b,
                    "initial_six_pity": 0,
                    "initial_five_pity": 0,
                    "keep_legacy_bonus_rules": bool(keep_legacy_bonus_rules),
                    "drop_threshold": float(drop_threshold),
                    "final_probability": float(final_probability),
                    "character_only_probability": float(character_only_probs[a]),
                    "weapon_only_probability": float(weapon_only_probs[b]),
                    "expected_six_star_count": float(expected_six_star_count),
                    "expected_weapon_quota": float(expected_weapon_quota),
                    "expected_weapon_ten_pulls": float(expected_weapon_ten_pulls),
                    "max_weapon_ten_pulls": int(max(total_mass_by_m) if total_mass_by_m else 0),
                    "state_mass": float(state_mass),
                    "elapsed_seconds": float("nan"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "generator": "precompute_joint_results.py::incremental_grid",
                }
                new_rows.append(row)
                known_keys.add(key)
                written_rows += 1

        elapsed_total = time.perf_counter() - started_at
        processed_rows = written_rows + skipped_rows
        avg = elapsed_total / max(1, processed_rows)
        remain = max(0, total_rows - processed_rows) * avg
        print(
            f"[draw {draw}/{max_pulls}] states={len(states)} written={written_rows} "
            f"skipped={skipped_rows} ETA~{remain/60:.1f} min"
        )

        if flush_every > 0 and len(new_rows) >= int(flush_every):
            checkpoint = pd.DataFrame(new_rows)
            if table.empty:
                table = checkpoint
            else:
                table = pd.concat([table, checkpoint], ignore_index=True)
            table.to_parquet(output_path, index=False)
            print(f"[save] checkpoint rows={len(table)}")
            new_rows = []

    if new_rows:
        tail = pd.DataFrame(new_rows)
        if table.empty:
            table = tail
        else:
            table = pd.concat([table, tail], ignore_index=True)
        table.to_parquet(output_path, index=False)
        print(f"[save] final rows={len(table)}")

    print(f"Saved: {output_path}")
    print(f"Summary: written={written_rows}, skipped={skipped_rows}, total={total_rows}")


if __name__ == "__main__":
    run_precompute()
