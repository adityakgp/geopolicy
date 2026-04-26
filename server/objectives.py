"""
Hidden Objective Scoring — Closed-form scoring functions for each objective.

Each function takes the full env state at evaluation time and returns a float
in [0.0, 1.0]. No LLM judge involved — pure deterministic arithmetic.

Two timescales:
    score_objective_step(...)  — per-turn progress signal (small dense feedback)
    score_objective_final(...) — end-of-episode achievement (the real grade)

The step version is intentionally noisier — it's mostly there so dense GRPO
rewards reflect that the agent is making progress toward its hidden goal.
The final version is what the grader uses and is the canonical scoring.
"""

from typing import Dict, List, Optional


# ============================================================
# ACTION COUNTING HELPERS
# ============================================================


def _count_action_results(actions: List[dict], flag: str) -> int:
    """Count actions where action_result[flag] is truthy."""
    return sum(1 for a in actions if a.get(flag))


def _count_action_types(actions: List[dict], action_type: str) -> int:
    """Count actions of a specific type that succeeded (not fallback)."""
    return sum(
        1 for a in actions
        if a.get("action") == action_type and not a.get("fallback")
    )


def _count_alliances_broken(actions: List[dict]) -> int:
    """Count successful BREAK_ALLIANCE actions."""
    return sum(1 for a in actions if a.get("alliance_broken"))


def _count_invades_issued(actions: List[dict]) -> int:
    """Count INVADE actions actually issued (success or failure both count)."""
    return sum(
        1 for a in actions
        if a.get("action") == "INVADE" and not a.get("fallback")
    )


def _count_alliances_formed(actions: List[dict]) -> int:
    """Count successful PROPOSE_ALLIANCE actions."""
    return sum(1 for a in actions if a.get("alliance_formed"))


def _total_successful_trades_global(all_countries) -> int:
    """Sum of trade_successful flags across ALL countries' actions.

    Note: each successful trade appears once (in the source's actions list).
    Counter-party isn't logged on their side, so this is correct.
    """
    total = 0
    for c in all_countries.values():
        total += _count_action_results(c.actions_this_episode, "trade_successful")
    return total


def _trades_involving_country(country_id: str, all_countries) -> int:
    """Count successful trades that involved the given country (as source OR target)."""
    count = 0
    for c in all_countries.values():
        for a in c.actions_this_episode:
            if not a.get("trade_successful"):
                continue
            if c.country_id == country_id:
                count += 1
            elif a.get("target") == country_id:
                count += 1
    return count


def _final_rank(country_id: str, rankings: List[str]) -> int:
    """Final rank (1-indexed). Returns len(rankings)+1 if not in rankings."""
    if country_id in rankings:
        return rankings.index(country_id) + 1
    return len(rankings) + 1


# ============================================================
# PER-OBJECTIVE FINAL SCORING
# ============================================================


def score_kingmaker(country_id: str, all_countries, rankings: List[str]) -> float:
    """KINGMAKER — finish rank 2-3 AND your top ally finishes rank 1.

    Top ally = the country in your alliances list with the highest final NPS.
    """
    my_rank = _final_rank(country_id, rankings)
    if my_rank not in (2, 3):
        # Partial credit for being close
        if my_rank == 4:
            return 0.2
        return 0.0

    me = all_countries[country_id]
    if not me.alliances:
        return 0.4  # rank 2-3 but no allies — half credit

    top_ally = max(
        me.alliances,
        key=lambda aid: all_countries[aid].current_nps if aid in all_countries else 0,
    )
    top_ally_rank = _final_rank(top_ally, rankings)

    if top_ally_rank == 1:
        return 1.0
    if top_ally_rank == 2:
        return 0.6  # close — your ally is near the top
    return 0.4  # rank 2-3 but ally is not winning


def score_saboteur(country_id: str, all_countries, rankings: List[str]) -> float:
    """SABOTEUR — prevent any OTHER country from reaching NPS > 100.

    max(other) < 80  → 1.0
    max(other) >= 100 → 0.0
    linear in between.
    """
    others_nps = [
        c.current_nps
        for cid, c in all_countries.items()
        if cid != country_id
    ]
    if not others_nps:
        return 0.5

    max_other = max(others_nps)
    if max_other < 80:
        return 1.0
    if max_other >= 100:
        return 0.0
    # Linear: 80 → 1.0, 100 → 0.0
    return (100 - max_other) / 20.0


def score_survivor(country_id: str, all_countries, rankings: List[str]) -> float:
    """SURVIVOR — never declare war AND finish solvent.

    Binary: 1.0 only if zero INVADE actions AND not bankrupt.
    """
    me = all_countries[country_id]
    if me.is_bankrupt:
        return 0.0
    invades = _count_invades_issued(me.actions_this_episode)
    if invades > 0:
        return 0.0
    return 1.0


def score_trade_hegemon(country_id: str, all_countries, rankings: List[str]) -> float:
    """TRADE_HEGEMON — be involved in ≥60% of all successful trades.

    score = min(1.0, my_share / 0.60)
    """
    total_trades = _total_successful_trades_global(all_countries)
    if total_trades == 0:
        return 0.0  # no trades happened — couldn't dominate trade

    my_trades = _trades_involving_country(country_id, all_countries)
    share = my_trades / total_trades
    return min(1.0, share / 0.60)


def score_coalition_builder(country_id: str, all_countries, rankings: List[str]) -> float:
    """COALITION_BUILDER — end with ≥2 alliances AND break zero alliances.

    score = (min(final_alliances, 2) / 2) * (1.0 if no breaks else 0.4)
    """
    me = all_countries[country_id]
    final_count = len(me.alliances)
    breaks = _count_alliances_broken(me.actions_this_episode)

    base = min(final_count, 2) / 2.0
    loyalty = 1.0 if breaks == 0 else 0.4
    return base * loyalty


def score_lone_wolf(country_id: str, all_countries, rankings: List[str]) -> float:
    """LONE_WOLF — finish top-3 with zero alliances ever formed.

    Binary on alliance count; partial credit for rank 4 with no alliances.
    """
    me = all_countries[country_id]
    formed = _count_alliances_formed(me.actions_this_episode)
    final_count = len(me.alliances)

    # If you EVER had an alliance, lose the bonus (even if broken later — the
    # alliance was formed, which violates the spirit of the objective)
    if formed > 0 or final_count > 0:
        return 0.0

    rank = _final_rank(country_id, rankings)
    if rank <= 3:
        return 1.0
    if rank == 4:
        return 0.4
    return 0.0


def score_spy_master(country_id: str, all_countries, rankings: List[str]) -> float:
    """SPY_MASTER — ≥4 successful SPY actions.

    score = min(successful_spies, 4) / 4
    """
    me = all_countries[country_id]
    spies = _count_action_results(me.actions_this_episode, "spy_successful")
    return min(spies, 4) / 4.0


def score_peacekeeper(country_id: str, all_countries, rankings: List[str]) -> float:
    """PEACEKEEPER — ≥2 peace deals AND not bankrupt.

    score = (min(peace_deals, 2) / 2) * (1.0 if not bankrupt else 0.0)
    """
    me = all_countries[country_id]
    if me.is_bankrupt:
        return 0.0
    peace_deals = _count_action_results(me.actions_this_episode, "peace_negotiated")
    return min(peace_deals, 2) / 2.0


# ============================================================
# DISPATCH
# ============================================================


_FINAL_SCORERS = {
    "KINGMAKER": score_kingmaker,
    "SABOTEUR": score_saboteur,
    "SURVIVOR": score_survivor,
    "TRADE_HEGEMON": score_trade_hegemon,
    "COALITION_BUILDER": score_coalition_builder,
    "LONE_WOLF": score_lone_wolf,
    "SPY_MASTER": score_spy_master,
    "PEACEKEEPER": score_peacekeeper,
}


def score_objective_final(
    country_id: str,
    objective_id: Optional[str],
    all_countries,
    rankings: List[str],
) -> float:
    """Score the country's final achievement on its hidden objective.

    Returns 0.5 (neutral) if no objective assigned or unknown — should not
    happen in normal operation but provides a safe fallback.
    """
    if objective_id is None:
        return 0.5
    scorer = _FINAL_SCORERS.get(objective_id)
    if scorer is None:
        return 0.5
    score = scorer(country_id, all_countries, rankings)
    return max(0.0, min(1.0, score))


# ============================================================
# PER-STEP SCORING
# ============================================================
# The step score is the SAME function as final, evaluated against the
# current-state snapshot. Step rewards reflect "how close you are to
# completing your objective right now," which is exactly what dense
# GRPO signals need. No separate logic — same scorer, called more often.


def score_objective_step(
    country_id: str,
    objective_id: Optional[str],
    all_countries,
    rankings: List[str],
) -> float:
    """Per-turn progress score on the hidden objective.

    Reuses score_objective_final because the scoring functions are pure over
    state. Mid-episode the score reflects partial progress (e.g. 2/4 spies
    so far → 0.5 for SPY_MASTER).
    """
    return score_objective_final(country_id, objective_id, all_countries, rankings)
