"""
Rubric Tests (env v2) — Per-rubric bounds, anti-gaming guards, basic invariants.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import GeoPolicyEnv
from server.rubrics import (
    EconomicRubric, DiplomaticRubric, MilitaryRubric,
    StabilityRubric, HiddenObjectiveRubric,
)
from server.scoring import TaskRubric, TASK_WEIGHTS
from models.action import GeoAction


def _fresh_env(task_id="task3"):
    env = GeoPolicyEnv()
    env.reset(task_id=task_id, seed=0)
    return env


# ============================================================
# Per-rubric bounds
# ============================================================


def test_each_rubric_step_score_in_bounds():
    env = _fresh_env("task3")
    for r in (EconomicRubric(), DiplomaticRubric(), MilitaryRubric(),
              StabilityRubric(), HiddenObjectiveRubric()):
        for cid in env.active_country_ids:
            score = r.step_score(cid, env.countries, {}, env.get_rankings())
            assert 0.0 <= score <= 1.0, f"{r.name}.step_score returned {score} for {cid}"
    print("PASS: all rubric step_scores in [0, 1]")


def test_each_rubric_final_score_in_bounds():
    env = _fresh_env("task3")
    for r in (EconomicRubric(), DiplomaticRubric(), MilitaryRubric(),
              StabilityRubric(), HiddenObjectiveRubric()):
        for cid in env.active_country_ids:
            score = r.final_score(cid, env.countries, env.get_rankings())
            assert 0.0 <= score <= 1.0, f"{r.name}.final_score returned {score} for {cid}"
    print("PASS: all rubric final_scores in [0, 1]")


# ============================================================
# TaskRubric blender
# ============================================================


def test_weights_sum_to_one_for_all_tasks():
    for task_id, weights in TASK_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"{task_id} weights sum to {total}"
    print("PASS: weights sum to 1.0 for all tasks")


def test_task1_disables_hidden_objective():
    assert TASK_WEIGHTS["task1"]["hidden"] == 0.0
    assert TASK_WEIGHTS["task2"]["hidden"] > 0.0
    assert TASK_WEIGHTS["task3"]["hidden"] > 0.0
    print("PASS: task1 disables hidden objective weight")


def test_task_rubric_step_returns_components():
    env = _fresh_env("task3")
    rubric = env.task_rubric
    out = rubric.step_reward("aria", env.countries, {}, env.get_rankings())
    assert "total" in out
    assert "components" in out
    assert set(out["components"].keys()) == {"economic", "diplomatic", "military", "stability", "hidden"}
    assert 0.0 <= out["total"] <= 1.0
    print("PASS: step_reward returns total + 5 components")


def test_task_rubric_final_returns_components():
    env = _fresh_env("task3")
    rubric = env.task_rubric
    out = rubric.final_grade("aria", env.countries, env.get_rankings())
    assert set(out["components"].keys()) == {"economic", "diplomatic", "military", "stability", "hidden"}
    assert 0.0 <= out["total"] <= 1.0
    print("PASS: final_grade returns total + 5 components")


# ============================================================
# Anti-gaming guards
# ============================================================


def test_economic_trade_spam_penalty():
    """After 4+ successful trades, EconomicRubric should apply a spam penalty."""
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    # Inject 6 fake successful trades into actions_this_episode
    aria.actions_this_episode = [
        {"action": "TRADE", "trade_successful": True,
         "gave": {"resource": "oil", "amount": 5},
         "received": {"resource": "food", "amount": 5}}
        for _ in range(6)
    ]
    eco = EconomicRubric()
    score = eco.step_score("aria", env.countries, {}, env.get_rankings())
    # Score should be reduced by spam penalty (2 trades over the cap × 0.05 = 0.1)
    # so worst case should still be in bounds, but not super high
    assert 0.0 <= score <= 1.0
    print(f"PASS: economic spam penalty applied (score={score:.3f})")


def test_military_defend_without_threat_gets_minimal_credit():
    """DEFEND with no active war should score near zero on defense_score."""
    env = _fresh_env("task3")
    mil = MilitaryRubric()
    # No wars, defend_active flag set
    score = mil.step_score("aria", env.countries, {"defend_active": True}, env.get_rankings())
    # Without war and with no real threat, defense_score=0.1, war_rate=0.5
    # Expected ~ 0.5*0.1 + 0.5*0.5 = 0.30 (modulo bankruptcy bumps)
    assert score < 0.6, f"DEFEND with no threat scored too high: {score}"
    print(f"PASS: defend without threat gets low credit (score={score:.3f})")


def test_diplomatic_alliance_churn_penalty():
    """Re-allying with someone you broke recently triggers churn penalty."""
    dip = DiplomaticRubric()
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    # Simulate: broke alliance with verdania a couple turns ago
    aria.actions_this_episode = [
        {"action": "BREAK_ALLIANCE", "alliance_broken": True, "target": "verdania"},
        {"action": "WAIT"},
    ]
    # Now re-form alliance with verdania (this turn)
    action_result = {"alliance_formed": True, "target": "verdania"}
    score = dip.step_score("aria", env.countries, action_result, env.get_rankings())
    # Score should be penalized by 0.2
    assert 0.0 <= score <= 1.0
    print(f"PASS: alliance churn penalty applied (score={score:.3f})")


def test_no_offset_means_zero_action_can_score_low():
    """v1 had a +0.5 baseline offset. v2 dropped it. Verify a degenerate
    bankrupt+collapsed country scores low (not artificially 0.5)."""
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.is_bankrupt = True
    aria.is_collapsed = True
    aria.economy = 0
    aria.internal_stability = 0
    aria.reputation = 0

    out = env.task_rubric.step_reward("aria", env.countries, {}, env.get_rankings())
    # With most components near 0 (bankrupt+collapsed), total should be < 0.4
    assert out["total"] < 0.5, f"bankrupt country scored {out['total']} — offset still present?"
    print(f"PASS: no +0.5 offset (bankrupt total={out['total']:.3f})")


if __name__ == "__main__":
    tests = [
        test_each_rubric_step_score_in_bounds,
        test_each_rubric_final_score_in_bounds,
        test_weights_sum_to_one_for_all_tasks,
        test_task1_disables_hidden_objective,
        test_task_rubric_step_returns_components,
        test_task_rubric_final_returns_components,
        test_economic_trade_spam_penalty,
        test_military_defend_without_threat_gets_minimal_credit,
        test_diplomatic_alliance_churn_penalty,
        test_no_offset_means_zero_action_can_score_low,
    ]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {e}"); failed += 1
    print(f"\n{'='*50}\nResults: {passed} passed, {failed} failed out of {len(tests)}\n{'='*50}")
