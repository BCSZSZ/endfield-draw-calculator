from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
    if pity_counter < 66:
        base = 0.008
    elif pity_counter < max_pity:
        base = 0.008 + (pity_counter - 65) * 0.05
    else:
        base = 1.0

    base = min(1.0, base)
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
    target_copies = max(1, int(target_copies))
    max_pity = max(1, int(max_pity))
    max_steps = max(1, int(max_steps))
    init_pity = max(0, min(int(initial_pity), max_pity))

    dp = np.zeros((max_pity + 1, target_copies), dtype=float)
    dp[init_pity, 0] = 1.0

    finish_pmf = np.zeros(max_steps + 1, dtype=float)
    history = np.zeros((max_steps + 1, max_pity + 1, target_copies + 1), dtype=float)
    history[0, :, :target_copies] = dp

    binom_probs = _bonus_binom_probs()

    for cost in range(1, max_steps + 1):
        new_dp = np.zeros_like(dp)

        for pity in range(max_pity + 1):
            p_up, p_other, p_fail = _get_rates(pity, max_pity)
            for copies in range(target_copies):
                prob = dp[pity, copies]
                if prob == 0.0:
                    continue

                if copies + 1 >= target_copies:
                    finish_pmf[cost] += prob * p_up
                else:
                    new_dp[0, copies + 1] += prob * p_up

                new_dp[0, copies] += prob * p_other
                next_pity = min(pity + 1, max_pity)
                new_dp[next_pity, copies] += prob * p_fail

        if cost == 30:
            temp = np.zeros_like(new_dp)
            for pity in range(max_pity + 1):
                for copies in range(target_copies):
                    val = new_dp[pity, copies]
                    if val == 0.0:
                        continue
                    for k, prob_k in enumerate(binom_probs):
                        final_c = copies + k
                        if final_c >= target_copies:
                            finish_pmf[cost] += val * prob_k
                        else:
                            temp[pity, final_c] += val * prob_k
            new_dp = temp

        if cost == 120:
            shift_mass = float(np.sum(new_dp[:, 0]))
            new_dp[:, 0] = 0.0
            if target_copies == 1:
                finish_pmf[cost] += shift_mass
            elif shift_mass > 0:
                new_dp[0, 1] += shift_mass

        if cost % 240 == 0:
            temp = np.zeros_like(new_dp)
            for copies in range(target_copies):
                col_mass = float(np.sum(new_dp[:, copies]))
                if copies + 1 >= target_copies:
                    finish_pmf[cost] += col_mass
                else:
                    temp[:, copies + 1] += new_dp[:, copies]
            new_dp = temp

        dp = new_dp
        history[cost, :, :target_copies] = dp
        history[cost, 0, target_copies] = float(np.sum(finish_pmf[: cost + 1]))

        if np.sum(dp) < 1e-12:
            if cost < max_steps:
                history[cost + 1 :, :, :] = history[cost, :, :]
            break

    cdf = np.cumsum(finish_pmf)
    return CharacterMarkovResult(history=history, finish_pmf=finish_pmf, cdf=cdf)


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
    pity = max(0, min(int(initial_pity), max_pity))
    copies = 0

    rng = np.random.default_rng(seed)
    binom_probs = _bonus_binom_probs()

    pity_path = np.zeros(max_steps + 1, dtype=int)
    copies_path = np.zeros(max_steps + 1, dtype=int)
    pity_path[0] = pity
    copies_path[0] = copies

    graduated = False

    for cost in range(1, max_steps + 1):
        if not graduated:
            p_up, p_other, _ = _get_rates(pity, max_pity)
            roll = rng.random()

            if roll < p_up:
                pity = 0
                copies += 1
            elif roll < p_up + p_other:
                pity = 0
            else:
                pity = min(pity + 1, max_pity)

            if cost == 30 and copies < target_copies:
                bonus_k = int(rng.choice(np.arange(11), p=binom_probs))
                copies += bonus_k

            if cost == 120 and copies == 0:
                copies = 1
                pity = 0

            if cost % 240 == 0 and copies < target_copies:
                copies += 1

            if copies >= target_copies:
                copies = target_copies
                graduated = True

        pity_path[cost] = pity
        copies_path[cost] = copies

    return CharacterTrajectory(pity_path=pity_path, copies_path=copies_path)
