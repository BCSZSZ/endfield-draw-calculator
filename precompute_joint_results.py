from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from joint_character_weapon_dp import (
    _apply_up_bonus_binomial_layered_encoded,
    _decode_state_id,
    _effective_drop_threshold,
    _encode_state_id,
    _get_weapon_cdf,
    _normal_tier_probs,
    _weapon_ten_pulls_from_quota_units,
    calculate_joint_character_weapon_probability,
    make_joint_scenario_key,
)

LEGACY_OUTPUT_PATH = Path("precomputed/joint_results.parquet")
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
    drop_threshold: float = 1e-10,
) -> list[dict]:
    scenarios: list[dict] = []
    for pulls in range(int(min_pulls), int(max_pulls) + 1):
        for char_target in range(int(min_char_target), int(max_char_target) + 1):
            for weapon_target in range(
                int(min_weapon_target), int(max_weapon_target) + 1
            ):
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


def _load_existing_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue

        if path.is_dir():
            dataset = ds.dataset(path, format="parquet")
            table = dataset.to_table(columns=["scenario_key"])
        else:
            table = pq.read_table(path, columns=["scenario_key"])

        if table.num_rows:
            keys.update(table.column("scenario_key").to_pylist())
    return keys


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_part_index(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0

    indices: list[int] = []
    for part in path.glob("part-*.parquet"):
        name = part.stem
        pieces = name.split("-")
        if len(pieces) == 2 and pieces[1].isdigit():
            indices.append(int(pieces[1]))
    return max(indices, default=-1) + 1


def _write_rows_dataset(
    output_path: Path,
    rows: list[dict],
    part_index: int,
    schema: pa.Schema | None,
) -> pa.Schema:
    if not rows:
        return schema or pa.schema([])

    output_path = _ensure_output_dir(output_path)
    part_path = output_path / f"part-{part_index:06d}.parquet"
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, part_path)
    return table.schema


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


def _find_first_missing_draw(
    known_keys: set[str],
    min_pulls: int,
    max_pulls: int,
    char_targets: list[int],
    weapon_targets: list[int],
    keep_legacy_bonus_rules: bool,
    drop_threshold: float,
) -> int:
    for draw in range(min_pulls, max_pulls + 1):
        draw_complete = True
        for a in char_targets:
            for b in weapon_targets:
                key = make_joint_scenario_key(
                    character_pulls=draw,
                    target_up_characters=a,
                    target_up_weapons=b,
                    initial_six_pity=0,
                    initial_five_pity=0,
                    keep_legacy_bonus_rules=keep_legacy_bonus_rules,
                    drop_threshold=drop_threshold,
                )
                if key not in known_keys:
                    draw_complete = False
                    break
            if not draw_complete:
                break

        if not draw_complete:
            return draw

    return max_pulls + 1


def run_precompute(
    scenarios: list[dict] | None = None,
    output_path: Path = OUTPUT_PATH,
    flush_every: int = 100,
) -> None:
    scenarios = scenarios or DEFAULT_SCENARIOS
    key_paths = [output_path]
    if LEGACY_OUTPUT_PATH != output_path:
        key_paths.append(LEGACY_OUTPUT_PATH)
    known_keys = _load_existing_keys(key_paths)
    print(f"[resume] loaded existing scenario keys: {len(known_keys)}")
    print(f"[resume] loaded existing scenario keys: {len(known_keys)}")

    schema: pa.Schema | None = None
    part_index = _next_part_index(output_path)
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
                print(f"[{idx}/{total}] [skip] {key} | ETA~{remain / 60:.1f} min")
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
            print(
                f"[{idx}/{total}] [done] {key} ({elapsed:.2f}s) | ETA~{remain / 60:.1f} min"
            )

            if flush_every > 0 and len(new_rows) >= int(flush_every):
                schema = _write_rows_dataset(
                    output_path=output_path,
                    rows=new_rows,
                    part_index=part_index,
                    schema=schema,
                )
                part_index += 1
                print(f"[save] checkpoint rows={len(new_rows)}")
                new_rows = []
    except KeyboardInterrupt:
        print(
            "[interrupt] Caught KeyboardInterrupt. Flushing buffered rows to parquet..."
        )
    finally:
        if new_rows:
            _write_rows_dataset(
                output_path=output_path,
                rows=new_rows,
                part_index=part_index,
                schema=schema,
            )
            print(f"[save] final rows={len(new_rows)}")

    print(f"Saved: {output_path}")


def run_precompute_grid_incremental(
    min_pulls: int = 1,
    max_pulls: int = 720,
    min_char_target: int = 1,
    max_char_target: int = 6,
    min_weapon_target: int = 1,
    max_weapon_target: int = 6,
    keep_legacy_bonus_rules: bool = True,
    drop_threshold: float = 1e-10,
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

    key_paths = [output_path]
    if LEGACY_OUTPUT_PATH != output_path:
        key_paths.append(LEGACY_OUTPUT_PATH)
    known_keys = _load_existing_keys(key_paths)

    schema: pa.Schema | None = None
    part_index = _next_part_index(output_path)

    char_targets = list(range(min_char_target, max_char_target + 1))
    weapon_targets = list(range(min_weapon_target, max_weapon_target + 1))
    effective_min_pulls = max(
        min_pulls,
        _find_first_missing_draw(
            known_keys=known_keys,
            min_pulls=min_pulls,
            max_pulls=max_pulls,
            char_targets=char_targets,
            weapon_targets=weapon_targets,
            keep_legacy_bonus_rules=keep_legacy_bonus_rules,
            drop_threshold=drop_threshold,
        ),
    )
    if effective_min_pulls > max_pulls:
        print(
            f"[resume] all scenarios already exist for draws {min_pulls}-{max_pulls}; nothing to do."
        )
        print(f"Saved: {output_path}")
        print("Summary: written=0, skipped=0, total=0")
        return

    print(
        f"[resume] first missing draw: {effective_min_pulls} (requested range {min_pulls}-{max_pulls})"
    )

    up_cap = max_char_target
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

    states: dict[int, dict[int, float]] = {_encode_state_id(0, 0, 0): {0: 1.0}}
    expected_six_star_count = 0.0

    max_quota_units_upper = 100 * max_pulls + (
        100 if keep_legacy_bonus_rules and max_pulls >= 120 else 0
    )
    max_weapon_ten_pulls_upper = _weapon_ten_pulls_from_quota_units(
        max_quota_units_upper
    )
    m_values = np.arange(max_weapon_ten_pulls_upper + 1, dtype=int)
    weapon_cdfs: dict[int, np.ndarray] = {}
    for b in weapon_targets:
        cdf = np.asarray(_get_weapon_cdf(b, max_weapon_ten_pulls_upper), dtype=float)
        if cdf.size < m_values.size:
            cdf = np.pad(cdf, (0, m_values.size - cdf.size), mode="edge")
        weapon_cdfs[b] = cdf

    total_rows = (
        (max_pulls - effective_min_pulls + 1)
        * len(char_targets)
        * len(weapon_targets)
    )
    written_rows = 0
    skipped_rows = 0
    new_rows: list[dict] = []
    started_at = time.perf_counter()
    quota_to_m: dict[int, int] = {}

    # 预分配numpy数组用于复用
    total_mass_by_m = np.zeros(m_values.size, dtype=float)
    char_mass_by_a_m = np.zeros((len(char_targets), m_values.size), dtype=float)

    try:
        for draw in range(1, max_pulls + 1):
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
                        q2 = quota_units + 100
                        key_map = next_states.get(next_id_6_up)
                        if key_map is None:
                            next_states[next_id_6_up] = {q2: mass * p6_up}
                        else:
                            key_map[q2] = key_map.get(q2, 0.0) + mass * p6_up

                    if p6_off > 0.0:
                        q2 = quota_units + 100
                        key_map = next_states.get(next_id_6_off)
                        if key_map is None:
                            next_states[next_id_6_off] = {q2: mass * p6_off}
                        else:
                            key_map[q2] = key_map.get(q2, 0.0) + mass * p6_off

                    if p5_star > 0.0:
                        q2 = quota_units + 10
                        key_map = next_states.get(next_id_5)
                        if key_map is None:
                            next_states[next_id_5] = {q2: mass * p5_star}
                        else:
                            key_map[q2] = key_map.get(q2, 0.0) + mass * p5_star

                    if p4_star > 0.0:
                        q2 = quota_units + 1
                        key_map = next_states.get(next_id_4)
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

            effective_threshold = _effective_drop_threshold(drop_threshold, draw) / (
                up_cap + 1
            )
            states = {}
            for key, quota_mass_map in next_states.items():
                filtered_map = {
                    quota_units: mass
                    for quota_units, mass in quota_mass_map.items()
                    if mass >= effective_threshold
                }
                if filtered_map:
                    states[key] = filtered_map

            if draw < effective_min_pulls:
                continue

            # 复用预分配的数组
            total_mass_by_m.fill(0.0)
            char_mass_by_a_m.fill(0.0)
            expected_weapon_quota = 0.0
            state_mass = 0.0

            for state_id, quota_mass_map in states.items():
                _, _, up = _decode_state_id(state_id)
                max_a_hit = min(up, max_char_target)

                for quota_units, mass in quota_mass_map.items():
                    if mass <= 0.0:
                        continue

                    m = quota_to_m.get(quota_units)
                    if m is None:
                        m = _weapon_ten_pulls_from_quota_units(quota_units)
                        quota_to_m[quota_units] = m

                    if m > max_weapon_ten_pulls_upper:
                        continue

                    total_mass_by_m[m] += mass
                    expected_weapon_quota += mass * quota_units * 20
                    state_mass += mass

                    if max_a_hit >= min_char_target:
                        char_mass_by_a_m[: max_a_hit - min_char_target + 1, m] += mass

            expected_weapon_ten_pulls = float(np.dot(m_values, total_mass_by_m))
            character_only_probs = char_mass_by_a_m.sum(axis=1)

            weapon_only_probs = {}
            weapon_cdf_slices = {}
            for b in weapon_targets:
                weapon_only_probs[b] = float(np.dot(total_mass_by_m, weapon_cdfs[b]))
                weapon_cdf_slices[b] = weapon_cdfs[b]

            nonzero_m = np.where(total_mass_by_m > 0)[0]
            max_weapon_ten_pulls = int(nonzero_m[-1]) if nonzero_m.size > 0 else 0

            for a in char_targets:
                char_masses = char_mass_by_a_m[a - min_char_target]
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

                    final_probability = float(np.dot(char_masses, weapon_cdf_slices[b]))
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
                        "character_only_probability": float(
                            character_only_probs[a - min_char_target]
                        ),
                        "weapon_only_probability": float(weapon_only_probs[b]),
                        "expected_six_star_count": float(expected_six_star_count),
                        "expected_weapon_quota": float(expected_weapon_quota),
                        "expected_weapon_ten_pulls": float(expected_weapon_ten_pulls),
                        "max_weapon_ten_pulls": max_weapon_ten_pulls,
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
                f"[draw {draw}/{max_pulls}] states={sum(len(m) for m in states.values())} written={written_rows} "
                f"skipped={skipped_rows} ETA~{remain / 60:.1f} min"
            )

            if flush_every > 0 and len(new_rows) >= int(flush_every):
                schema = _write_rows_dataset(
                    output_path=output_path,
                    rows=new_rows,
                    part_index=part_index,
                    schema=schema,
                )
                part_index += 1
                print(f"[save] checkpoint rows={len(new_rows)}")
                new_rows = []
    except KeyboardInterrupt:
        print(
            "[interrupt] Caught KeyboardInterrupt. Flushing buffered rows to parquet..."
        )
    finally:
        if new_rows:
            _write_rows_dataset(
                output_path=output_path,
                rows=new_rows,
                part_index=part_index,
                schema=schema,
            )
            print(f"[save] final rows={len(new_rows)}")

    print(f"Saved: {output_path}")
    print(
        f"Summary: written={written_rows}, skipped={skipped_rows}, total={total_rows}"
    )


if __name__ == "__main__":
    run_precompute_grid_incremental()
