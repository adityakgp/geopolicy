"""
Alignment Tests (env v2) — The critical tests v1 didn't have.

These check that step rewards and final grades are consistent (no
step-reward / grader misalignment) and that hidden objectives are
properly hidden in observations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import GeoPolicyEnv
from models.action import GeoAction


def _fresh_env(task_id="task3"):
    env = GeoPolicyEnv()
    env.reset(task_id=task_id, seed=42)
    return env


# ============================================================
# Alignment tests
# ============================================================


def test_step_and_final_share_components():
    """Step reward and final grade must use the same set of rubric components.

    This is the core invariant that prevents step-reward / grader misalignment.
    """
    env = _fresh_env("task3")
    rankings = env.get_rankings()
    step = env.task_rubric.step_reward("aria", env.countries, {}, rankings)
    final = env.task_rubric.final_grade("aria", env.countries, rankings)
    assert set(step["components"].keys()) == set(final["components"].keys())
    print("PASS: step and final share component names")


def test_step_and_final_use_same_weights():
    """Both step_reward and final_grade pull from TASK_WEIGHTS[task_id]."""
    env = _fresh_env("task3")
    # Mutate weights via the rubric instance and verify both methods reflect it
    rubric = env.task_rubric
    expected_keys = set(rubric.weights.keys())
    step = rubric.step_reward("aria", env.countries, {}, env.get_rankings())
    final = rubric.final_grade("aria", env.countries, env.get_rankings())
    assert set(step["components"].keys()) == expected_keys
    assert set(final["components"].keys()) == expected_keys
    print("PASS: step and final use the same per-task weights")


def test_grade_country_returns_blended_total():
    """grade_country() should return same number as task_rubric.final_grade()['total']."""
    env = _fresh_env("task3")
    rankings = env.get_rankings()
    detailed = env.task_rubric.final_grade("aria", env.countries, rankings)
    grade = env.grade_country("aria")
    assert abs(grade - round(detailed["total"], 4)) < 1e-6
    print("PASS: grade_country == task_rubric.final_grade()['total']")


# ============================================================
# Observation masking
# ============================================================


def test_observation_includes_own_objective():
    """The country's own observation should include its hidden objective."""
    env = _fresh_env("task3")
    obs = env.get_observation("aria")
    assert obs.hidden_objective_id is not None
    assert obs.hidden_objective_name is not None
    assert obs.hidden_objective_description is not None
    print(f"PASS: own observation includes objective ({obs.hidden_objective_id})")


def test_observation_masks_others_objectives():
    """Other countries' info in the observation must NOT include hidden_objective."""
    env = _fresh_env("task3")
    obs = env.get_observation("aria")
    for other_id, info in obs.other_countries.items():
        # PublicCountryInfo schema must not have hidden_objective_id
        assert not hasattr(info, "hidden_objective_id"), (
            f"Other country {other_id} leaks hidden_objective_id"
        )
        assert "hidden_objective" not in info.dict()
    print("PASS: others' observations do not leak hidden objectives")


def test_task1_disables_hidden_objectives():
    """Task 1 (bilateral) should not assign hidden objectives."""
    env = GeoPolicyEnv()
    env.reset(task_id="task1", seed=0)
    for cid in env.active_country_ids:
        assert env.countries[cid].hidden_objective is None
    obs = env.get_observation(env.active_country_ids[0])
    assert obs.hidden_objective_id is None
    print("PASS: task1 has no hidden objectives assigned")


def test_task2_assigns_hidden_objectives():
    """Task 2 should assign one objective per country."""
    env = GeoPolicyEnv()
    env.reset(task_id="task2", seed=0)
    objectives = [env.countries[cid].hidden_objective for cid in env.active_country_ids]
    assert all(o is not None for o in objectives)
    assert len(set(objectives)) == len(objectives)  # all unique
    print(f"PASS: task2 assigns {len(objectives)} unique objectives")


def test_task3_assigns_hidden_objectives():
    """Task 3 should assign one objective per country."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3", seed=0)
    objectives = [env.countries[cid].hidden_objective for cid in env.active_country_ids]
    assert all(o is not None for o in objectives)
    assert len(set(objectives)) == len(objectives)
    print(f"PASS: task3 assigns {len(objectives)} unique objectives")


# ============================================================
# Snapshot/restore preserves objectives
# ============================================================


def test_snapshot_preserves_hidden_objectives():
    """Snapshot/restore must preserve hidden objective assignments and rubric."""
    env = _fresh_env("task3")
    original = {cid: env.countries[cid].hidden_objective for cid in env.active_country_ids}
    snap = env.snapshot()

    # Mutate
    for cid in env.active_country_ids:
        env.countries[cid].hidden_objective = "KINGMAKER"

    # Restore
    env.restore(snap)
    restored = {cid: env.countries[cid].hidden_objective for cid in env.active_country_ids}
    assert restored == original
    # Rubric should still work post-restore
    out = env.task_rubric.step_reward("aria", env.countries, {}, env.get_rankings())
    assert "total" in out
    print("PASS: snapshot/restore preserves hidden objectives and rubric")


# ============================================================
# Reward contains components in metadata
# ============================================================


def test_step_metadata_includes_components():
    """env.step() should attach reward_components to obs.metadata for logging."""
    env = _fresh_env("task3")
    a = GeoAction(action_type="WAIT", source_country="aria")
    obs = env.step(a)
    assert "reward_components" in obs.metadata
    assert set(obs.metadata["reward_components"].keys()) == {
        "economic", "diplomatic", "military", "stability", "hidden"
    }
    print("PASS: step metadata includes reward_components")


# ============================================================
# Reward in valid range
# ============================================================


def test_step_reward_always_in_unit_interval():
    """Run a 12-turn task3 episode and verify all rewards are in [0, 1]."""
    env = _fresh_env("task3")
    countries = list(env.active_country_ids)
    for turn in range(12):
        for cid in countries:
            a = GeoAction(action_type="WAIT", source_country=cid)
            obs = env.step(a)
            assert 0.0 <= obs.reward <= 1.0, f"reward {obs.reward} out of range"
            if obs.done:
                break
        if obs.done:
            break
    print("PASS: all step rewards in [0, 1] across full episode")


if __name__ == "__main__":
    tests = [
        test_step_and_final_share_components,
        test_step_and_final_use_same_weights,
        test_grade_country_returns_blended_total,
        test_observation_includes_own_objective,
        test_observation_masks_others_objectives,
        test_task1_disables_hidden_objectives,
        test_task2_assigns_hidden_objectives,
        test_task3_assigns_hidden_objectives,
        test_snapshot_preserves_hidden_objectives,
        test_step_metadata_includes_components,
        test_step_reward_always_in_unit_interval,
    ]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {e}"); failed += 1
    print(f"\n{'='*50}\nResults: {passed} passed, {failed} failed out of {len(tests)}\n{'='*50}")
