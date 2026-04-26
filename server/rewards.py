"""
Reward Calculation — env v2 thin shim over TaskRubric.

The real per-step reward logic now lives in server/rubrics.py +
server/scoring.py. This module is kept as a compatibility shim for any
external code that imports `calculate_step_reward` directly.

Returns the blended total in [0, 1]. For the per-rubric breakdown, use
TaskRubric.step_reward(...) directly.
"""

from typing import List

from server.scoring import TaskRubric


def calculate_step_reward(
    country_id: str,
    prev_nps: float,
    new_nps: float,
    action_result: dict,
    rankings: List[str],
    all_countries: dict = None,
    task_id: str = "task3",
) -> float:
    """Backward-compatible step reward.

    The old signature took prev_nps and new_nps but didn't need the full
    countries dict; the new rubric-based computation does. Callers that
    don't have `all_countries` available will get a degenerate score of 0.5.

    For new code, call env.task_rubric.step_reward(...) directly to get
    both the total and the per-rubric components.
    """
    if all_countries is None:
        return 0.5  # safe fallback when called without full state

    rubric = TaskRubric(task_id)
    result = rubric.step_reward(country_id, all_countries, action_result, rankings)
    return result["total"]
