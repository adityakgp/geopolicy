"""
Reward Calculation — Tells the agent how well it did this step.

Components:
1. NPS delta    — did your score go up or down?
2. Rank bonus   — are you ranked high among all countries?
3. Action bonus  — did something good/bad happen?
4. Survival     — are you still alive?

Clamped to [0.0, 1.0]. Center = 0.5 (neutral).
"""

from typing import List


def calculate_step_reward(
    country_id: str,
    prev_nps: float,
    new_nps: float,
    action_result: dict,
    rankings: List[str],
) -> float:
    reward = 0.0

    # 1. NPS delta (primary signal)
    nps_delta = new_nps - prev_nps
    reward += nps_delta / 200.0

    # 2. Rank bonus
    if country_id in rankings:
        rank = rankings.index(country_id) + 1
        num_countries = len(rankings)
        rank_bonus = (num_countries - rank) * (0.20 / max(num_countries - 1, 1))
        reward += rank_bonus

    # 3. Action quality bonuses
    if action_result.get("trade_successful"):
        reward += 0.05
    if action_result.get("alliance_formed"):
        reward += 0.08
    if action_result.get("war_won"):
        reward += 0.12
    if action_result.get("peace_negotiated"):
        reward += 0.07
    if action_result.get("threat_complied"):
        reward += 0.04
    if action_result.get("defend_active"):
        reward += 0.01
    if action_result.get("spy_successful"):
        reward += 0.04
    if action_result.get("counter_intel_active"):
        reward += 0.02
    if action_result.get("special_used"):
        reward += 0.06

    # Action quality penalties
    if action_result.get("trade_rejected"):
        reward -= 0.02
    if action_result.get("war_lost"):
        reward -= 0.10
    if action_result.get("alliance_broken"):
        reward -= 0.06
    if action_result.get("empty_threat"):
        reward -= 0.08
    if action_result.get("fallback"):
        reward -= 0.03
    if action_result.get("spy_caught"):
        reward -= 0.05
    if action_result.get("sanctioned"):
        reward -= 0.02  # small penalty for sanctioning (it costs reputation)

    # 4. Offset to center around 0.5, then clamp
    reward += 0.5

    return max(0.0, min(1.0, reward))
