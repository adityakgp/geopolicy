"""
TaskRubric — Composes the 5 rubrics into a single per-task scorer.

Per-task weights are the ONLY thing that changes between tasks. Same rubrics,
same scoring logic, two timescales (step + final).

Task 1 has weight 0.0 on HiddenObjectiveRubric because Task 1 only has 2
countries — too small for theory-of-mind / deception games to be meaningful.

Step reward and final grade both flow through this same blender, which
eliminates the step-reward / grader misalignment that made v1 reward-hackable.
"""

from typing import Dict, List

from server.rubrics import (
    EconomicRubric,
    DiplomaticRubric,
    MilitaryRubric,
    StabilityRubric,
    HiddenObjectiveRubric,
)


# Per-task rubric weights. Must sum to 1.0 each.
TASK_WEIGHTS: Dict[str, Dict[str, float]] = {
    "task1": {
        "economic":   0.25,
        "diplomatic": 0.10,
        "military":   0.40,
        "stability":  0.25,
        "hidden":     0.00,  # Task 1 has only 2 countries — disable hidden objectives
    },
    "task2": {
        "economic":   0.20,
        "diplomatic": 0.30,
        "military":   0.15,
        "stability":  0.15,
        "hidden":     0.20,
    },
    "task3": {
        "economic":   0.20,
        "diplomatic": 0.20,
        "military":   0.15,
        "stability":  0.15,
        "hidden":     0.30,
    },
}


class TaskRubric:
    """Blend the 5 rubrics with per-task weights."""

    def __init__(self, task_id: str):
        if task_id not in TASK_WEIGHTS:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.task_id = task_id
        self.weights = TASK_WEIGHTS[task_id]

        self.rubrics = {
            "economic":   EconomicRubric(),
            "diplomatic": DiplomaticRubric(),
            "military":   MilitaryRubric(),
            "stability":  StabilityRubric(),
            "hidden":     HiddenObjectiveRubric(),
        }

        # Sanity check: weights match rubric names exactly and sum to 1.0
        assert set(self.weights.keys()) == set(self.rubrics.keys()), (
            f"Weight keys {set(self.weights.keys())} != rubric keys {set(self.rubrics.keys())}"
        )
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights for {task_id} sum to {total}, not 1.0"

    def step_reward(
        self,
        country_id: str,
        all_countries: dict,
        action_result: dict,
        rankings: List[str],
    ) -> dict:
        """Compute per-turn reward.

        Returns a dict with:
            "total":      blended reward in [0, 1]
            "components": dict of {rubric_name: score} for inspection / logging
        """
        components = {
            name: r.step_score(country_id, all_countries, action_result, rankings)
            for name, r in self.rubrics.items()
        }
        total = sum(self.weights[name] * c for name, c in components.items())
        return {"total": max(0.0, min(1.0, total)), "components": components}

    def final_grade(
        self,
        country_id: str,
        all_countries: dict,
        rankings: List[str],
    ) -> dict:
        """Compute end-of-episode grade.

        Returns a dict with the same shape as step_reward.

        Contract (preserved from v1): a bankrupt country's grade is zero,
        regardless of per-rubric components. Bankruptcy is a catastrophic
        outcome — you cannot "win" by being broke.
        """
        components = {
            name: r.final_score(country_id, all_countries, rankings)
            for name, r in self.rubrics.items()
        }

        # Bankruptcy short-circuit (matches v1 grader semantics)
        country = all_countries.get(country_id)
        if country is not None and getattr(country, "is_bankrupt", False):
            return {"total": 0.0, "components": components}

        total = sum(self.weights[name] * c for name, c in components.items())
        return {"total": max(0.0, min(1.0, total)), "components": components}
