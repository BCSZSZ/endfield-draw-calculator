"""Character probabilities. Pity counts previous ordinary misses, not the next pull."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

MODEL_VERSION = "2.0.0"
FREE_PULL_MILESTONES = {"chartered": (30,), "rerun": (30, 60, 90)}


def six_star_rate(pity: int, max_pity: int = 80) -> float:
    if pity >= max_pity - 1:
        return 1.0
    return min(1.0, 0.008 + max(0, pity - 64) * 0.05)


def add_free_pulls(state: np.ndarray) -> np.ndarray:
    """Independent base-rate ten-pull; neither pity nor ordinary-UP flag changes."""
    cap = state.shape[-1] - 1
    result = np.zeros_like(state)
    for copies in range(cap + 1):
        for bonus in range(11):
            probability = math.comb(10, bonus) * 0.004**bonus * 0.996 ** (10 - bonus)
            result[:, :, min(cap, copies + bonus)] += state[:, :, copies] * probability
    return result


def character_step(state: np.ndarray, draw: int, bonus: bool = False) -> np.ndarray:
    """State axes: six-star misses, ordinary UP seen, copies (last is >= cap)."""
    cap = state.shape[-1] - 1
    rate = np.array([six_star_rate(p, state.shape[0]) for p in range(state.shape[0])])
    active = state[:, :, :cap].copy()
    result = np.zeros_like(state)
    result[0, 1, cap] = state[:, :, cap].sum()
    if draw == 120:
        result[0, 1, 1:] += active[:, 0, :].sum(axis=0)
        active[:, 0, :] = 0.0
    result[0, 1, 1:] += (active * (rate / 2)[:, None, None]).sum(axis=(0, 1))
    result[0, :, :cap] += (active * (rate / 2)[:, None, None]).sum(axis=0)
    result[1:, :, :cap] += active[:-1] * (1 - rate[:-1])[:, None, None]
    if bonus:
        result = add_free_pulls(result)
    if draw % 240 == 0:
        gift = np.zeros_like(result)
        gift[:, :, 1:] = result[:, :, :cap]
        gift[:, :, cap] += result[:, :, cap]
        result = gift
    return result


def character_states(
    max_pulls: int, initial_pity: int = 0, banner: str = "chartered",
    copies_cap: int = 6, max_pity: int = 80,
) -> Iterator[np.ndarray]:
    if banner not in FREE_PULL_MILESTONES:
        raise ValueError("Unknown banner")
    if not 0 <= initial_pity < max_pity or max_pulls < 0 or copies_cap < 1:
        raise ValueError("Invalid character scenario")
    state = np.zeros((max_pity, 2, copies_cap + 1))
    state[initial_pity, 0, 0] = 1.0
    yield state
    for draw in range(1, max_pulls + 1):
        state = character_step(state, draw, draw in FREE_PULL_MILESTONES[banner])
        yield state


def character_distributions(
    max_pulls: int, initial_pity: int = 0, banner: str = "chartered", copies_cap: int = 6,
) -> np.ndarray:
    return np.array([
        state.sum(axis=(0, 1))
        for state in character_states(max_pulls, initial_pity, banner, copies_cap)
    ])


def distribution_statistics(cdf: np.ndarray) -> dict:
    pmf = np.diff(cdf, prepend=0.0)
    pmf = np.maximum(pmf, 0.0)
    costs = np.arange(len(cdf))
    mass = float(cdf[-1])
    mean = float(costs @ pmf) / (mass or 1.0)
    variance = max(0.0, float(costs**2 @ pmf) / (mass or 1.0) - mean**2)
    std = math.sqrt(variance)
    lower, upper = max(0, math.ceil(mean - std)), min(len(cdf) - 1, math.floor(mean + std))
    quantiles = {}
    for percent in (25, 50, 75, 90, 99):
        index = int(np.searchsorted(cdf, percent / 100))
        quantiles[f"p{percent}"] = index if index < len(cdf) else None
    return {
        "costs": costs, "finish_probs": pmf, "mean": mean, "variance": variance,
        "std_dev": std, "check_sum": float(cdf[-1]), "one_std_range": (lower, upper),
        "within_one_std": float(pmf[lower:upper + 1].sum()) / (mass or 1.0), **quantiles,
    }
