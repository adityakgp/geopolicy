"""
Task Graders — Score each country's performance 0.0 to 1.0.

Each task has different criteria for scoring:

Task 1 (Easy): Bilateral — just NPS ratio vs opponent
Task 2 (Medium): Coalition — rank + alliance formation + event survival
Task 3 (Hard): Full Sim — rank + NPS ratio + alliances + stability

IMPORTANT: All scores MUST be in [0.0, 1.0]. Judges auto-check this.
A score outside this range = disqualification.
"""

from typing import Dict


def grade_task1(country_id: str, results: dict) -> float:
    """
    Task 1 — Bilateral Survival (Easy).

    Only 2 countries. Score based on NPS ratio vs opponent.

    1.0  = dominating win (NPS ratio >= 1.3)
    0.8  = clear win
    0.6  = narrow win
    0.4  = close loss
    0.2  = clear loss
    0.1  = heavy loss
    0.0  = bankrupt
    """
    if results["bankrupt"].get(country_id, False):
        return 0.0

    my_nps = results["final_nps"][country_id]
    opponent_id = [c for c in results["final_nps"] if c != country_id][0]
    opponent_nps = results["final_nps"][opponent_id]

    nps_ratio = my_nps / max(opponent_nps, 1.0)

    if nps_ratio >= 1.3:
        return 1.0
    elif nps_ratio >= 1.1:
        return 0.8
    elif nps_ratio >= 1.0:
        return 0.6
    elif nps_ratio >= 0.9:
        return 0.4
    elif nps_ratio >= 0.7:
        return 0.2
    else:
        return 0.1


def grade_task2(country_id: str, results: dict) -> float:
    """
    Task 2 — Coalition Wars (Medium).

    5 countries. Score based on:
    - Rank (60%): where you finish in NPS rankings
    - Alliance bonus (30%): did you form alliances?
    - Event survival (10%): did you survive global events without collapse?
    """
    if results["bankrupt"].get(country_id, False):
        return 0.0

    rankings = results["final_rankings"]
    rank = rankings.index(country_id) + 1 if country_id in rankings else 5

    # Rank score (60%)
    rank_score = {1: 0.60, 2: 0.50, 3: 0.35, 4: 0.20, 5: 0.10}.get(rank, 0.10)

    # Alliance bonus (30%) — based on alliances formed during episode
    alliances_count = results["alliances_ever_formed"].get(country_id, 0)
    alliance_bonus = min(alliances_count * 0.15, 0.30)

    # Stability bonus (10%) — survived without collapse
    collapsed = results["ever_collapsed"].get(country_id, False)
    stability_bonus = 0.0 if collapsed else 0.10

    total = rank_score + alliance_bonus + stability_bonus
    return min(1.0, round(total, 3))


def grade_task3(country_id: str, results: dict) -> float:
    """
    Task 3 — Full Geopolitical Simulation (Hard).

    5 countries, everything active. Score based on:
    - NPS ratio vs best (40%): how close to the top NPS?
    - Rank (30%): final ranking position
    - Alliance (20%): maintaining alliances at episode end
    - Stability (10%): final stability score
    """
    if results["bankrupt"].get(country_id, False):
        return 0.0

    collapsed = results["ever_collapsed"].get(country_id, False)
    if collapsed:
        rankings = results["final_rankings"]
        rank = rankings.index(country_id) + 1 if country_id in rankings else 5
        return max(0.1, round(0.3 - (rank * 0.05), 3))

    rankings = results["final_rankings"]
    rank = rankings.index(country_id) + 1 if country_id in rankings else 5

    # NPS component (40%)
    my_nps = results["final_nps"][country_id]
    max_nps = max(results["final_nps"].values())
    nps_score = (my_nps / max(max_nps, 1.0)) * 0.40

    # Rank component (30%)
    rank_score = {1: 0.30, 2: 0.22, 3: 0.15, 4: 0.08, 5: 0.03}.get(rank, 0.03)

    # Alliance component (20%)
    final_alliances = results["final_alliances"].get(country_id, 0)
    alliance_score = min(final_alliances * 0.10, 0.20)

    # Stability component (10%)
    final_stability = results["final_stability"].get(country_id, 0)
    stability_score = (final_stability / 100.0) * 0.10

    total = nps_score + rank_score + alliance_score + stability_score
    return min(1.0, round(total, 3))


def grade(country_id: str, task_id: str, results: dict) -> float:
    """Route to the correct grader based on task."""
    if task_id == "task1":
        return grade_task1(country_id, results)
    elif task_id == "task2":
        return grade_task2(country_id, results)
    elif task_id == "task3":
        return grade_task3(country_id, results)
    else:
        return 0.0
