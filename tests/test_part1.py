"""
Part 1 Tests — Skeleton works (updated for openenv refactor).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import GeoPolicyEnv
from models.action import GeoAction
from models.observation import GeoObservation
from models.state import GeoState


def test_reset_returns_observation():
    env = GeoPolicyEnv()
    obs = env.reset(task_id="task1")
    assert isinstance(obs, GeoObservation)
    assert obs.turn == 0
    assert obs.done is False
    assert obs.task_id == "task1"
    print("PASS: reset returns valid observation")


def test_reset_sets_episode_id():
    env = GeoPolicyEnv()
    env.reset()
    id1 = env.state.episode_id
    env.reset()
    id2 = env.state.episode_id
    assert id1 != id2
    assert len(id1) > 0
    print("PASS: reset creates unique episode IDs")


def test_step_returns_observation():
    env = GeoPolicyEnv()
    env.reset(task_id="task1")
    action = GeoAction(action_type="WAIT", source_country="aria")
    result = env.step(action)
    assert isinstance(result, GeoObservation)
    assert result.done is not None
    assert result.reward is not None
    assert 0.0 <= result.reward <= 1.0
    print("PASS: step returns valid GeoObservation with done+reward")


def test_step_advances_turn():
    env = GeoPolicyEnv()
    env.reset(task_id="task1")
    action = GeoAction(action_type="WAIT", source_country="aria")
    env.step(action)
    assert env.state.current_turn == 1
    print("PASS: step advances turn")


def test_episode_ends_after_max_turns():
    env = GeoPolicyEnv()
    env.reset(task_id="task1")
    action = GeoAction(action_type="WAIT", source_country="aria")
    for i in range(8):
        result = env.step(action)
    assert result.done is True
    print("PASS: episode ends after max turns")


def test_state_returns_metadata():
    env = GeoPolicyEnv()
    env.reset(task_id="task2")
    state = env.state
    assert isinstance(state, GeoState)
    assert state.task_id == "task2"
    assert state.max_turns == 10
    assert state.done is False
    print("PASS: state returns valid metadata")


def test_task_configs():
    env = GeoPolicyEnv()
    env.reset(task_id="task1"); assert env.state.max_turns == 8
    env.reset(task_id="task2"); assert env.state.max_turns == 10
    env.reset(task_id="task3"); assert env.state.max_turns == 12
    print("PASS: all task configs correct")


def test_action_model():
    a1 = GeoAction(action_type="WAIT")
    assert a1.action_type == "WAIT"
    a2 = GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                    resource="oil", amount=20.0, counter_resource="food", counter_amount=30.0)
    assert a2.target_country == "verdania"
    print("PASS: action model works")


if __name__ == "__main__":
    tests = [test_reset_returns_observation, test_reset_sets_episode_id,
             test_step_returns_observation, test_step_advances_turn,
             test_episode_ends_after_max_turns, test_state_returns_metadata,
             test_task_configs, test_action_model]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {e}"); failed += 1
    print(f"\n{'='*40}\nResults: {passed} passed, {failed} failed out of {len(tests)}\n{'='*40}")
