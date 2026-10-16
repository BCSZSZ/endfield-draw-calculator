"""One forward pass for every character/weapon target, with a bounded discarded tail."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from draw_model import six_star_rate
from weapon_draw import calculate_weapon_full_potential


@dataclass
class JointHistory:
    probabilities: np.ndarray
    metrics: np.ndarray


def _add_up(destination: np.ndarray, source: np.ndarray) -> None:
    destination[..., 1:, :] += source[..., :-1, :]
    destination[..., -1, :] += source[..., -1, :]


def _free_ten(state: np.ndarray) -> np.ndarray:
    """Bonus draws earn quota; the all-4-star ten-pull is corrected to include a 5-star."""
    original = state
    for _ in range(10):
        result = np.zeros((*state.shape[:-1], state.shape[-1] + 11))
        result[..., :state.shape[-1]] += state * 0.912
        result[..., 1:state.shape[-1] + 1] += state * 0.08
        result[..., 11:] += state * 0.004
        _add_up(result[..., 11:], state * 0.004)
        state = result
    all_four = original * 0.912**10
    state[..., :original.shape[-1]] -= all_four
    state[..., 1:original.shape[-1] + 1] += all_four
    return state


def build_joint_history(
    max_pulls: int = 720, copies_cap: int = 6, initial_six_pity: int = 0,
    initial_five_pity: int = 0, include_rewards: bool = True,
    tail_tolerance: float = 1e-14, progress: Callable[[int, float], None] | None = None,
) -> JointHistory:
    # quota_units = (ordinary pulls + bonus pulls) + 9 * quota_index.
    # A 4/5/6-star adds 0/1/11 to quota_index; one unit is 20 Arsenal Tickets.
    state = np.zeros((80, 10, 2, copies_cap + 1, 1))
    state[initial_six_pity, initial_five_pity, 0, 0, 0] = 1.0
    probabilities = np.zeros((max_pulls + 1, copies_cap + 1, 7))
    probabilities[0, 0, 0] = 1.0
    # six-star expectation, quota expectation, weapon ten-pull expectation, max ten-pulls, mass.
    metrics = np.zeros((max_pulls + 1, 5))
    metrics[0, 4] = 1.0
    six = np.array([six_star_rate(p) for p in range(80)])
    five = np.minimum(0.08, 1 - six)[:, None] * np.ones((1, 10))
    five[:, 9] = 1 - six
    four = 1 - six[:, None] - five
    weapon_cdfs = np.ones((101, 7))
    for target in range(1, 7):
        weapon_cdfs[:, target] = calculate_weapon_full_potential(target, 100)['finish_probs'].cumsum()
    expected_six, bonus_count = 0.0, 0
    for draw in range(1, max_pulls + 1):
        width = state.shape[-1]
        result = np.zeros((*state.shape[:-1], width + 11))
        seen_index = state.shape[2] - 1
        if include_rewards and draw == 120:
            forced = state[:, :, 0].sum(axis=(0, 1))
            _add_up(result[0, 0, seen_index, :, 11:], forced)
            expected_six += float(forced.sum())
            state[:, :, 0] = 0.0
        six_mass = (state * (six / 2)[:, None, None, None, None]).sum(axis=(0, 1))
        expected_six += float(six_mass.sum()) * 2
        result[0, 0, :, :, 11:] += six_mass
        _add_up(result[0, 0, seen_index, :, 11:], six_mass.sum(axis=0))
        result[1:, 0, :, :, 1:width + 1] += (
            state[:-1] * five[:-1, :, None, None, None]
        ).sum(axis=1)
        result[1:, 1:, :, :, :width] += state[:-1, :-1] * four[:-1, :-1, None, None, None]
        state = result
        if include_rewards and draw == 30:
            expected_six += float(state.sum()) * 0.08
            state = _free_ten(state)
            bonus_count += 10
        if include_rewards and draw % 240 == 0:
            gift = np.zeros_like(state)
            _add_up(gift, state)
            state = gift
        if include_rewards and draw == 120:
            state = state[:, :, 1:]
        marginal = state.sum(axis=(0, 1, 2, 3))
        if tail_tolerance > 0:
            tail = marginal[::-1].cumsum()[::-1]
            retained = np.flatnonzero(tail > tail_tolerance)
            if len(retained):
                state = state[..., :retained[-1] + 1]
                marginal = marginal[:retained[-1] + 1]
        quota_units = draw + bonus_count + 9 * np.arange(state.shape[-1])
        ten_pulls = quota_units // 99
        copy_quota = state.sum(axis=(0, 1, 2))
        copy_tails = copy_quota[::-1].cumsum(axis=0)[::-1]
        probabilities[draw] = copy_tails @ weapon_cdfs[np.minimum(100, ten_pulls)]
        metrics[draw] = (
            expected_six, float(quota_units @ marginal) * 20,
            float(ten_pulls @ marginal), int(ten_pulls[-1]), float(marginal.sum()),
        )
        if progress and (draw % 60 == 0 or draw == max_pulls):
            progress(draw, metrics[draw, 4])
    return JointHistory(probabilities, metrics)
