from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from draw_model import character_states, six_star_rate


@dataclass
class CharacterMarkovResult:
    history: np.ndarray
    finish_pmf: np.ndarray
    cdf: np.ndarray


@dataclass
class CharacterTrajectory:
    pity_path: np.ndarray
    copies_path: np.ndarray


def _get_rates(pity_counter: int, max_pity: int) -> tuple[float, float, float]:
    base = six_star_rate(pity_counter, max_pity)
    return base * 0.5, base * 0.5, 1.0 - base


def _bonus_binom_probs() -> np.ndarray:
    p_bonus = 0.004
    probs = np.zeros(11, dtype=float)
    for k in range(11):
        probs[k] = math.comb(10, k) * (p_bonus**k) * ((1 - p_bonus) ** (10 - k))
    return probs


def build_character_markov_history(
    target_copies: int = 6,
    max_pity: int = 80,
    max_steps: int = 400,
    initial_pity: int = 0,
) -> CharacterMarkovResult:
    history = np.zeros((max_steps + 1, max_pity + 1, target_copies + 1))
    for draw, state in enumerate(character_states(
        max_steps, initial_pity, copies_cap=target_copies, max_pity=max_pity,
    )):
        history[draw, :max_pity] = state.sum(axis=1)
    cdf = history[:, :, target_copies].sum(axis=1)
    return CharacterMarkovResult(
        history=history, finish_pmf=np.maximum(0.0, np.diff(cdf, prepend=0.0)), cdf=cdf,
    )


def sample_character_trajectory(
    target_copies: int = 6,
    max_pity: int = 80,
    max_steps: int = 400,
    initial_pity: int = 0,
    seed: int | None = None,
) -> CharacterTrajectory:
    target_copies = max(1, int(target_copies))
    max_pity = max(1, int(max_pity))
    max_steps = max(1, int(max_steps))
    pity = max(0, min(int(initial_pity), max_pity - 1))
    copies = 0

    rng = np.random.default_rng(seed)
    binom_probs = _bonus_binom_probs()

    pity_path = np.zeros(max_steps + 1, dtype=int)
    copies_path = np.zeros(max_steps + 1, dtype=int)
    pity_path[0] = pity
    copies_path[0] = copies

    graduated = False
    ordinary_up_seen = False

    for cost in range(1, max_steps + 1):
        if not graduated:
            p_up, p_other, _ = _get_rates(pity, max_pity)
            if cost == 120 and not ordinary_up_seen:
                p_up, p_other = 1.0, 0.0
            roll = rng.random()

            if roll < p_up:
                pity = 0
                copies += 1
                ordinary_up_seen = True
            elif roll < p_up + p_other:
                pity = 0
            else:
                pity = min(pity + 1, max_pity)

            if cost == 30 and copies < target_copies:
                bonus_k = int(rng.choice(np.arange(11), p=binom_probs))
                copies += bonus_k

            if cost % 240 == 0 and copies < target_copies:
                copies += 1

            if copies >= target_copies:
                copies = target_copies
                graduated = True

        pity_path[cost] = pity
        copies_path[cost] = copies

    return CharacterTrajectory(pity_path=pity_path, copies_path=copies_path)
