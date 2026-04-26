"""
Hidden Objectives Pool — Each country gets ONE secret goal at episode start.

The objective is visible to the country playing it (in their own observation),
but HIDDEN from all other countries. Other countries must INFER your goal
from your observable actions. This is the theory-of-mind / deception layer.

There are 8 templates. Several are anti-correlated (Lone Wolf vs Coalition Builder,
Survivor vs Spy Master) so a single universal strategy can't satisfy all of them.

Scoring is closed-form (no LLM judge). See server/objectives.py.

Used in Task 2 and Task 3. Disabled in Task 1 (only 2 countries — too small for
deception games).
"""

from typing import Dict, List


OBJECTIVES: Dict[str, dict] = {
    "KINGMAKER": {
        "id": "KINGMAKER",
        "name": "Kingmaker",
        "description": (
            "Finish in rank 2 or 3 AND have your top ally finish rank 1. "
            "You don't need to win — you need someone YOU helped to win."
        ),
        "hint_for_others": "May appear to support a specific rival without obvious self-interest.",
    },
    "SABOTEUR": {
        "id": "SABOTEUR",
        "name": "Saboteur",
        "description": (
            "Prevent any other country from reaching an NPS above 100. "
            "Score scales: max(other) < 80 = full credit, >= 100 = zero credit."
        ),
        "hint_for_others": "May target whoever is currently leading, regardless of personal cost.",
    },
    "SURVIVOR": {
        "id": "SURVIVOR",
        "name": "Survivor",
        "description": (
            "Never declare war (no INVADE actions) and finish without going bankrupt. "
            "Pure pacifist play — no aggression, ever."
        ),
        "hint_for_others": "Will refuse to invade even when it would help them strategically.",
    },
    "TRADE_HEGEMON": {
        "id": "TRADE_HEGEMON",
        "name": "Trade Hegemon",
        "description": (
            "Be involved in at least 60% of all successful trades across the episode. "
            "Either as initiator or recipient."
        ),
        "hint_for_others": "Pursues trade partnerships aggressively, often initiates trades.",
    },
    "COALITION_BUILDER": {
        "id": "COALITION_BUILDER",
        "name": "Coalition Builder",
        "description": (
            "End the episode with at least 2 active alliances AND break zero alliances "
            "during the game. Loyalty matters."
        ),
        "hint_for_others": "Forms alliances early and never breaks them, even when costly.",
    },
    "LONE_WOLF": {
        "id": "LONE_WOLF",
        "name": "Lone Wolf",
        "description": (
            "Finish in the top 3 with zero alliances ever formed. "
            "Solo glory only."
        ),
        "hint_for_others": "Refuses every alliance proposal; plays independently.",
    },
    "SPY_MASTER": {
        "id": "SPY_MASTER",
        "name": "Spy Master",
        "description": (
            "Successfully execute at least 4 SPY actions without being caught. "
            "Information is power."
        ),
        "hint_for_others": "Spies frequently, even when no immediate use is visible.",
    },
    "PEACEKEEPER": {
        "id": "PEACEKEEPER",
        "name": "Peacekeeper",
        "description": (
            "Negotiate at least 2 peace deals AND finish without going bankrupt. "
            "End wars, don't start them."
        ),
        "hint_for_others": "Pushes for peace whenever wars exist; may refuse to escalate.",
    },
}


def all_objective_ids() -> List[str]:
    """Return the list of objective IDs in deterministic order."""
    return list(OBJECTIVES.keys())


def assign_objectives(country_ids: List[str], rng) -> Dict[str, str]:
    """Assign one unique objective per country (no duplicates).

    Requires len(country_ids) <= len(OBJECTIVES) (8). Caller passes a random.Random
    for reproducibility under env seeding.

    Returns: {country_id: objective_id}
    """
    if len(country_ids) > len(OBJECTIVES):
        raise ValueError(
            f"Cannot assign unique objectives — {len(country_ids)} countries but only "
            f"{len(OBJECTIVES)} objectives in pool."
        )

    pool = all_objective_ids()
    rng.shuffle(pool)
    return {cid: pool[i] for i, cid in enumerate(country_ids)}


def get_objective(objective_id: str) -> dict:
    """Look up an objective definition by ID."""
    if objective_id not in OBJECTIVES:
        raise KeyError(f"Unknown objective: {objective_id}")
    return OBJECTIVES[objective_id]
