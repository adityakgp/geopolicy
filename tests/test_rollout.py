"""
Test play_one_rollout — the generic rollout wrapper.

What we verify:
1. Stub WAIT policy → 12-turn task3 episode, valid grade
2. Stub DEVELOP policy → resources actually grow (proves actions take effect)
3. Stub that raises exceptions → falls back to WAIT cleanly
4. History contains all the keys downstream code (Part 4 mining, Part 11 plots) needs
5. Snapshot/restore + rollout work together (GRPO use case)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rollout import play_one_rollout
from server.environment import GeoPolicyEnv


def test_stub_wait():
    """All-WAIT stub policy → episode runs to end, grade is valid."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")

    def stub_wait(obs, cid):
        return {"action_type": "WAIT"}

    history, reward = play_one_rollout(stub_wait, env, country_id="ironhold")

    assert len(history) == 12, f"task3 should be 12 turns, got {len(history)}"
    assert 0.0 <= reward <= 1.0, f"reward out of range: {reward}"
    assert all(
        turn["actions"]["ironhold"]["action_type"] == "WAIT" for turn in history
    ), "ironhold should have WAITed every turn"
    print(f"✅ stub WAIT: 12 turns, ironhold reward={reward:.3f}")


def test_stub_develop_changes_state():
    """DEVELOP oil should make aria's oil go up (proves actions take effect)."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    initial_oil = env.countries["aria"].oil

    def stub_develop(obs, cid):
        return {"action_type": "DEVELOP", "resource": "oil"}

    history, reward = play_one_rollout(stub_develop, env, country_id="aria")
    final_oil = env.countries["aria"].oil

    assert final_oil > initial_oil, \
        f"DEVELOP oil should grow oil; started {initial_oil}, ended {final_oil}"
    assert 0.0 <= reward <= 1.0
    print(f"✅ stub DEVELOP: aria oil {initial_oil:.0f} → {final_oil:.0f}, reward={reward:.3f}")


def test_stub_that_raises_falls_back_to_wait():
    """If generate_fn raises, country should still get a WAIT and episode completes."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")

    def stub_broken(obs, cid):
        raise ValueError("LLM API timeout simulation")

    history, reward = play_one_rollout(stub_broken, env, country_id="ironhold")

    assert len(history) == 12
    assert 0.0 <= reward <= 1.0
    assert all(
        turn["actions"]["ironhold"]["action_type"] == "WAIT" for turn in history
    ), "exceptions should fall back to WAIT"
    print(f"✅ exceptions handled: 12 turns, fallback to WAIT, reward={reward:.3f}")


def test_history_has_required_keys():
    """history entries must contain everything Part 4 (SFT mining) and Part 11 (plots) need."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")

    def stub(obs, cid):
        return {"action_type": "WAIT"}

    history, _ = play_one_rollout(stub, env, "ironhold")

    required_keys = {"turn", "obs", "actions", "rewards", "rankings", "results"}
    for i, turn in enumerate(history):
        missing = required_keys - turn.keys()
        assert not missing, f"turn {i} missing keys: {missing}"
        # obs/actions/rewards/results should each cover all 5 countries
        assert set(turn["obs"].keys()) == set(env.active_country_ids)
        assert set(turn["actions"].keys()) == set(env.active_country_ids)
        # rankings should be all 5 countries
        assert len(turn["rankings"]) == len(env.active_country_ids)

    # An obs must look like a real GeoObservation (have at least these)
    obs_required = {"country_id", "turn", "oil", "current_nps", "other_countries"}
    sample_obs = history[0]["obs"]["ironhold"]
    missing_obs = obs_required - sample_obs.keys()
    assert not missing_obs, f"obs missing fields: {missing_obs}"
    print(f"✅ history has all required keys (obs, actions, rewards, rankings, results)")


def test_rollout_with_snapshot_restore():
    """8 rollouts from same snapshot start identically — the GRPO use case."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    snap = env.snapshot()

    def stub_wait(obs, cid):
        return {"action_type": "WAIT"}

    rewards = []
    first_turn_obs = []
    for rollout_id in range(3):
        env.restore(snap)
        history, reward = play_one_rollout(stub_wait, env, "ironhold")
        rewards.append(reward)
        first_turn_obs.append(history[0]["obs"]["ironhold"]["oil"])

    # All 3 rollouts started from same snapshot + deterministic policy → same reward
    assert rewards[0] == rewards[1] == rewards[2], \
        f"deterministic rollouts should match: {rewards}"
    # Turn-1 observation should be identical across all 3 rollouts
    assert first_turn_obs[0] == first_turn_obs[1] == first_turn_obs[2], \
        f"turn-1 obs differs: {first_turn_obs}"
    print(f"✅ snapshot+rollout: 3 deterministic rollouts gave identical reward={rewards[0]:.3f}")


def test_country_id_validation():
    """Asking for a country not in the episode should error early, not silently."""
    env = GeoPolicyEnv()
    env.reset(task_id="task1")  # task1 only has aria + verdania

    def stub(obs, cid):
        return {"action_type": "WAIT"}

    try:
        play_one_rollout(stub, env, country_id="ironhold")
        raise AssertionError("should have raised — ironhold not in task1")
    except ValueError as e:
        assert "ironhold" in str(e)
        print(f"✅ validates country_id is in active_country_ids")


if __name__ == "__main__":
    test_stub_wait()
    test_stub_develop_changes_state()
    test_stub_that_raises_falls_back_to_wait()
    test_history_has_required_keys()
    test_rollout_with_snapshot_restore()
    test_country_id_validation()
    print("\n🎉 All rollout tests passed.")
