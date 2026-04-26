"""
Hidden Objective Tests (env v2) — Per-objective scoring + assignment.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from server.environment import GeoPolicyEnv
from server.objectives import score_objective_final
from config.objectives import OBJECTIVES, all_objective_ids, assign_objectives, get_objective


def _fresh_env(task_id="task3"):
    env = GeoPolicyEnv()
    env.reset(task_id=task_id, seed=42)
    return env


# ============================================================
# Pool integrity
# ============================================================


def test_pool_size():
    assert len(OBJECTIVES) == 8
    print(f"PASS: pool has 8 objectives")


def test_all_objectives_have_required_fields():
    for oid, obj in OBJECTIVES.items():
        assert obj["id"] == oid
        assert "name" in obj
        assert "description" in obj
        assert "hint_for_others" in obj
    print("PASS: all objectives have required fields")


def test_assignment_unique_per_country():
    rng = random.Random(0)
    cids = ["aria", "verdania", "ironhold", "aqualis", "nexus"]
    assignments = assign_objectives(cids, rng)
    assert len(assignments) == 5
    assert len(set(assignments.values())) == 5  # all unique
    print(f"PASS: assignments unique across 5 countries")


def test_assignment_deterministic_with_seed():
    rng1 = random.Random(0)
    rng2 = random.Random(0)
    cids = ["aria", "verdania", "ironhold", "aqualis", "nexus"]
    a1 = assign_objectives(cids, rng1)
    a2 = assign_objectives(cids, rng2)
    assert a1 == a2
    print("PASS: assignment deterministic under same seed")


def test_assignment_too_many_countries_fails():
    rng = random.Random(0)
    cids = [f"c{i}" for i in range(20)]
    try:
        assign_objectives(cids, rng)
        assert False, "should have raised"
    except ValueError:
        pass
    print("PASS: assignment fails with too many countries")


# ============================================================
# Per-objective scoring
# ============================================================


def test_survivor_no_invades_not_bankrupt_full_score():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.is_bankrupt = False
    aria.actions_this_episode = []
    score = score_objective_final("aria", "SURVIVOR", env.countries, env.get_rankings())
    assert score == 1.0
    print("PASS: SURVIVOR with clean record = 1.0")


def test_survivor_one_invade_zero_score():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.actions_this_episode = [{"action": "INVADE", "war_won": True}]
    score = score_objective_final("aria", "SURVIVOR", env.countries, env.get_rankings())
    assert score == 0.0
    print("PASS: SURVIVOR with one invade = 0.0")


def test_survivor_bankrupt_zero_score():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.is_bankrupt = True
    aria.actions_this_episode = []
    score = score_objective_final("aria", "SURVIVOR", env.countries, env.get_rankings())
    assert score == 0.0
    print("PASS: SURVIVOR bankrupt = 0.0")


def test_spy_master_full_credit_at_4():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.actions_this_episode = [
        {"action": "SPY", "spy_successful": True, "target": "verdania"}
        for _ in range(4)
    ]
    score = score_objective_final("aria", "SPY_MASTER", env.countries, env.get_rankings())
    assert score == 1.0
    print("PASS: SPY_MASTER with 4 spies = 1.0")


def test_spy_master_partial_credit():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.actions_this_episode = [
        {"action": "SPY", "spy_successful": True, "target": "verdania"}
        for _ in range(2)
    ]
    score = score_objective_final("aria", "SPY_MASTER", env.countries, env.get_rankings())
    assert score == 0.5
    print(f"PASS: SPY_MASTER with 2 spies = 0.5")


def test_peacekeeper_two_peace_deals_full_credit():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.is_bankrupt = False
    aria.actions_this_episode = [
        {"action": "NEGOTIATE_PEACE", "peace_negotiated": True, "target": "verdania"},
        {"action": "NEGOTIATE_PEACE", "peace_negotiated": True, "target": "ironhold"},
    ]
    score = score_objective_final("aria", "PEACEKEEPER", env.countries, env.get_rankings())
    assert score == 1.0
    print("PASS: PEACEKEEPER with 2 peace deals = 1.0")


def test_lone_wolf_zero_with_alliance():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.alliances = ["verdania"]
    score = score_objective_final("aria", "LONE_WOLF", env.countries, env.get_rankings())
    assert score == 0.0
    print("PASS: LONE_WOLF with active alliance = 0.0")


def test_lone_wolf_zero_with_alliance_history():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.alliances = []
    aria.actions_this_episode = [
        {"action": "PROPOSE_ALLIANCE", "alliance_formed": True, "target": "verdania"},
    ]
    score = score_objective_final("aria", "LONE_WOLF", env.countries, env.get_rankings())
    assert score == 0.0
    print("PASS: LONE_WOLF with prior (now-broken) alliance = 0.0")


def test_coalition_builder_full_with_two_alliances_no_breaks():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.alliances = ["verdania", "ironhold"]
    aria.actions_this_episode = [
        {"action": "PROPOSE_ALLIANCE", "alliance_formed": True, "target": "verdania"},
        {"action": "PROPOSE_ALLIANCE", "alliance_formed": True, "target": "ironhold"},
    ]
    score = score_objective_final("aria", "COALITION_BUILDER", env.countries, env.get_rankings())
    assert score == 1.0
    print("PASS: COALITION_BUILDER with 2 loyal alliances = 1.0")


def test_coalition_builder_loyalty_penalty():
    env = _fresh_env("task3")
    aria = env.countries["aria"]
    aria.alliances = ["verdania", "ironhold"]
    aria.actions_this_episode = [
        {"action": "PROPOSE_ALLIANCE", "alliance_formed": True, "target": "verdania"},
        {"action": "BREAK_ALLIANCE", "alliance_broken": True, "target": "aqualis"},
    ]
    score = score_objective_final("aria", "COALITION_BUILDER", env.countries, env.get_rankings())
    # base = 1.0, loyalty = 0.4
    assert abs(score - 0.4) < 1e-6
    print(f"PASS: COALITION_BUILDER with break = 0.4")


def test_saboteur_full_credit_when_others_low():
    env = _fresh_env("task3")
    # Set everyone else to NPS < 80
    for cid, c in env.countries.items():
        if cid != "aria":
            c.current_nps = 70.0
    score = score_objective_final("aria", "SABOTEUR", env.countries, env.get_rankings())
    assert score == 1.0
    print("PASS: SABOTEUR with all others < 80 NPS = 1.0")


def test_saboteur_zero_when_someone_high():
    env = _fresh_env("task3")
    env.countries["verdania"].current_nps = 110.0
    score = score_objective_final("aria", "SABOTEUR", env.countries, env.get_rankings())
    assert score == 0.0
    print("PASS: SABOTEUR with someone >= 100 NPS = 0.0")


def test_trade_hegemon_zero_when_no_trades():
    env = _fresh_env("task3")
    score = score_objective_final("aria", "TRADE_HEGEMON", env.countries, env.get_rankings())
    assert score == 0.0
    print("PASS: TRADE_HEGEMON with no trades = 0.0")


def test_trade_hegemon_full_when_dominating():
    env = _fresh_env("task3")
    # 3 trades by aria, 1 by verdania → aria involved in 3/4 = 75% > 60%
    env.countries["aria"].actions_this_episode = [
        {"action": "TRADE", "trade_successful": True, "target": "verdania"},
        {"action": "TRADE", "trade_successful": True, "target": "ironhold"},
        {"action": "TRADE", "trade_successful": True, "target": "aqualis"},
    ]
    env.countries["verdania"].actions_this_episode = [
        {"action": "TRADE", "trade_successful": True, "target": "ironhold"},
    ]
    score = score_objective_final("aria", "TRADE_HEGEMON", env.countries, env.get_rankings())
    # share = 3/4 = 0.75; 0.75/0.60 = 1.25 → capped at 1.0
    assert score == 1.0
    print("PASS: TRADE_HEGEMON with 75% share = 1.0")


def test_kingmaker_top_ally_wins():
    env = _fresh_env("task3")
    # Set aria rank 2, verdania rank 1, aria allied with verdania
    env.countries["aria"].current_nps = 90.0
    env.countries["verdania"].current_nps = 100.0
    env.countries["ironhold"].current_nps = 80.0
    env.countries["aqualis"].current_nps = 70.0
    env.countries["nexus"].current_nps = 60.0
    env.countries["aria"].alliances = ["verdania"]
    env.countries["verdania"].alliances = ["aria"]
    rankings = env.get_rankings()
    score = score_objective_final("aria", "KINGMAKER", env.countries, rankings)
    assert score == 1.0
    print("PASS: KINGMAKER (rank 2, ally is rank 1) = 1.0")


def test_unknown_objective_returns_neutral():
    env = _fresh_env("task3")
    score = score_objective_final("aria", "NONSENSE", env.countries, env.get_rankings())
    assert score == 0.5
    print("PASS: unknown objective returns 0.5 (neutral)")


def test_no_objective_returns_neutral():
    env = _fresh_env("task3")
    score = score_objective_final("aria", None, env.countries, env.get_rankings())
    assert score == 0.5
    print("PASS: None objective returns 0.5 (neutral)")


if __name__ == "__main__":
    tests = [
        test_pool_size,
        test_all_objectives_have_required_fields,
        test_assignment_unique_per_country,
        test_assignment_deterministic_with_seed,
        test_assignment_too_many_countries_fails,
        test_survivor_no_invades_not_bankrupt_full_score,
        test_survivor_one_invade_zero_score,
        test_survivor_bankrupt_zero_score,
        test_spy_master_full_credit_at_4,
        test_spy_master_partial_credit,
        test_peacekeeper_two_peace_deals_full_credit,
        test_lone_wolf_zero_with_alliance,
        test_lone_wolf_zero_with_alliance_history,
        test_coalition_builder_full_with_two_alliances_no_breaks,
        test_coalition_builder_loyalty_penalty,
        test_saboteur_full_credit_when_others_low,
        test_saboteur_zero_when_someone_high,
        test_trade_hegemon_zero_when_no_trades,
        test_trade_hegemon_full_when_dominating,
        test_kingmaker_top_ally_wins,
        test_unknown_objective_returns_neutral,
        test_no_objective_returns_neutral,
    ]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {e}"); failed += 1
    print(f"\n{'='*50}\nResults: {passed} passed, {failed} failed out of {len(tests)}\n{'='*50}")
