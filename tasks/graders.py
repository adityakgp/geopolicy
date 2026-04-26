"""
Task Graders — env v2 thin shim over TaskRubric.

In env v1 each task had its own hand-coded grader function. In env v2 the
final grade flows through the same composable TaskRubric used for per-step
rewards (see server/rubrics.py + server/scoring.py). This eliminates the
step-reward / grader misalignment that made v1 reward-hackable.

These wrappers exist for backward compatibility. The canonical interface is
env.grade_country() which returns the blended total, or
env.grade_country_detailed() which returns per-rubric components for plotting.

IMPORTANT: All scores are in [0.0, 1.0].
"""

from typing import List

from server.scoring import TaskRubric


def _grade_via_rubric(country_id: str, task_id: str, results: dict) -> float:
    """Shared backbone that reconstructs TaskRubric inputs from a results dict.

    The results dict format predates TaskRubric and uses precomputed summaries.
    For full fidelity the env should call env.grade_country() directly which
    has access to the live country state. This shim handles the case where
    only `results` is available.
    """
    # If `results` carries the live countries dict, use it directly (env v2 path)
    all_countries = results.get("countries")
    rankings = results.get("final_rankings", [])

    if all_countries is None:
        # Legacy path: no live state available, return a degenerate score
        # so old callers don't crash. New code should use env.grade_country().
        if results.get("bankrupt", {}).get(country_id, False):
            return 0.0
        return 0.5

    rubric = TaskRubric(task_id)
    out = rubric.final_grade(country_id, all_countries, rankings)
    return round(out["total"], 4)


def grade_task1(country_id: str, results: dict) -> float:
    """Task 1 (Bilateral) — delegates to TaskRubric."""
    return _grade_via_rubric(country_id, "task1", results)


def grade_task2(country_id: str, results: dict) -> float:
    """Task 2 (Coalition Wars) — delegates to TaskRubric."""
    return _grade_via_rubric(country_id, "task2", results)


def grade_task3(country_id: str, results: dict) -> float:
    """Task 3 (Full Simulation) — delegates to TaskRubric."""
    return _grade_via_rubric(country_id, "task3", results)


def grade(country_id: str, task_id: str, results: dict) -> float:
    """Route to the correct grader based on task."""
    return _grade_via_rubric(country_id, task_id, results)
