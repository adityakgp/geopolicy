"""Part 3 Tests — Actions change the world (openenv refactor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.environment import GeoPolicyEnv
from models.action import GeoAction

def test_wait_recovers_resources():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    oil_before = env.countries["aria"].oil
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert env.countries["aria"].oil == oil_before + 3 + 2  # WAIT +3, natural +2
    print(f"PASS: WAIT recovers resources ({oil_before} → {env.countries['aria'].oil})")

def test_develop_increases_resource():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    oil_b = env.countries["aria"].oil; eco_b = env.countries["aria"].economy
    env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"))
    assert env.countries["aria"].oil == oil_b + 15 + 2
    assert env.countries["aria"].economy == eco_b - 20 + 2
    print("PASS: DEVELOP works")

def test_develop_diminishing_returns():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r1 = env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"))
    r2 = env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"))
    assert r1.metadata["action_result"]["gain"] == 15
    assert r2.metadata["action_result"]["gain"] == 8
    print("PASS: diminishing returns work")

def test_develop_invalid_resource():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r = env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="gold"))
    assert r.metadata["action_result"].get("fallback") is True
    print("PASS: invalid DEVELOP falls back to WAIT")

def test_trade_successful():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    a_oil = env.countries["aria"].oil; a_food = env.countries["aria"].food
    r = env.step(GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                            resource="oil", amount=20.0, counter_resource="food", counter_amount=20.0))
    assert r.metadata["action_result"]["success"] is True
    assert env.countries["aria"].oil == a_oil - 20 + 2
    assert env.countries["aria"].food == a_food + 20 + 2
    print("PASS: trade works")

def test_trade_creates_agreement():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.step(GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                        resource="oil", amount=15, counter_resource="food", counter_amount=15))
    assert "verdania" in env.countries["aria"].trade_agreements
    print("PASS: trade creates trade agreements")

def test_trade_rejected_unfair():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r = env.step(GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                            resource="oil", amount=50, counter_resource="food", counter_amount=1))
    assert r.metadata["action_result"]["success"] is False
    print("PASS: unfair trade rejected")

def test_trade_rejected_insufficient():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r = env.step(GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                            resource="food", amount=20, counter_resource="oil", counter_amount=40))
    assert r.metadata["action_result"]["success"] is False
    print("PASS: trade rejected when target lacks resources")

def test_trade_with_self_rejected():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r = env.step(GeoAction(action_type="TRADE", source_country="aria", target_country="aria",
                            resource="oil", amount=10, counter_resource="food", counter_amount=10))
    assert r.metadata["action_result"].get("fallback") is True
    print("PASS: self-trade rejected")

def test_reward_in_valid_range():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    for action in [GeoAction(action_type="WAIT", source_country="aria"),
                   GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"),
                   GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                             resource="oil", amount=10, counter_resource="food", counter_amount=10)]:
        r = env.step(action)
        assert 0.0 <= r.reward <= 1.0, f"Reward {r.reward} out of range"
    print("PASS: all rewards in [0.0, 1.0]")

def test_reward_differs_by_action():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r1 = env.step(GeoAction(action_type="WAIT", source_country="aria"))
    r2 = env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"))
    r3 = env.step(GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                             resource="oil", amount=15, counter_resource="food", counter_amount=15))
    rewards = [r1.reward, r2.reward, r3.reward]
    assert len(set(rewards)) >= 2, f"Rewards should vary: {rewards}"
    print(f"PASS: rewards differ: {rewards}")

def test_natural_recovery_all_countries():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    vb = env.countries["verdania"].oil
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert env.countries["verdania"].oil == vb + 2
    print("PASS: natural recovery applies to all countries")

def test_episode_ends_at_max_turns():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    for _ in range(8):
        r = env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert r.done is True
    print("PASS: episode ends at max turns")

def test_nps_changes_after_actions():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    initial = env.countries["aria"].current_nps
    env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"))
    assert env.countries["aria"].current_nps != initial
    print("PASS: NPS changes after action")

def test_bankrupt_country_cant_act():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.countries["aria"].economy = 0; env.countries["aria"].is_bankrupt = True
    r = env.step(GeoAction(action_type="DEVELOP", source_country="aria", resource="oil"))
    assert r.metadata["action_result"].get("fallback") is True
    print("PASS: bankrupt country falls back to WAIT")

if __name__ == "__main__":
    tests = [test_wait_recovers_resources, test_develop_increases_resource, test_develop_diminishing_returns,
             test_develop_invalid_resource, test_trade_successful, test_trade_creates_agreement,
             test_trade_rejected_unfair, test_trade_rejected_insufficient, test_trade_with_self_rejected,
             test_reward_in_valid_range, test_reward_differs_by_action, test_natural_recovery_all_countries,
             test_episode_ends_at_max_turns, test_nps_changes_after_actions, test_bankrupt_country_cant_act]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {type(e).__name__}: {e}"); f += 1
    print(f"\n{'='*40}\nResults: {p} passed, {f} failed out of {len(tests)}\n{'='*40}")
