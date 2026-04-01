"""Part 2 Tests — Country data, world state, NPS, info asymmetry (openenv refactor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.environment import GeoPolicyEnv, calculate_nps, value_to_tier
from models.country import Country
from models.action import GeoAction
from config.countries import COUNTRIES

def test_reset_creates_correct_countries_task1():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    assert len(env.countries) == 2
    assert "aria" in env.countries and "verdania" in env.countries
    print("PASS: task1 creates 2 correct countries")

def test_reset_creates_correct_countries_task3():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    assert len(env.countries) == 5
    print("PASS: task3 creates all 5 countries")

def test_aria_starting_resources():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    a = env.countries["aria"]
    assert a.oil == 90 and a.water == 20 and a.food == 30 and a.military == 70 and a.economy == 80
    print("PASS: Aria has correct starting resources")

def test_all_countries_have_unique_resources():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    profiles = []
    for cid, c in env.countries.items():
        p = (c.oil, c.water, c.food, c.military, c.economy)
        assert p not in profiles; profiles.append(p)
    print("PASS: all countries have unique resource profiles")

def test_nps_calculation():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    nps = calculate_nps(env.countries["aria"])
    assert abs(nps - 79.5) < 0.01, f"Expected 79.5, got {nps}"
    print(f"PASS: NPS calculation correct (Aria={nps})")

def test_nps_rankings():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    rankings = env.get_rankings()
    nps_values = [env.countries[cid].current_nps for cid in rankings]
    assert len(set(nps_values)) == len(nps_values)
    print("PASS: all countries have different NPS rankings")

def test_observation_shows_own_exact_resources():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    obs = env.get_observation("aria")
    assert obs.oil == 90 and obs.country_name == "Aria"
    print("PASS: observation shows own exact resources")

def test_task1_full_transparency():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    obs = env.get_observation("aria")
    vi = obs.other_countries["verdania"]
    assert vi.exact_oil == 10 and vi.exact_food == 90
    print("PASS: task1 shows full transparency")

def test_task2_hidden_info():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    obs = env.get_observation("aria")
    ii = obs.other_countries["ironhold"]
    assert ii.military_tier == "very_high" and ii.exact_military is None
    print("PASS: task2 hides exact values")

def test_value_to_tier():
    assert value_to_tier(5) == "very_low" and value_to_tier(50) == "medium" and value_to_tier(95) == "very_high"
    print("PASS: value_to_tier works correctly")

def test_country_special_abilities():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    assert env.countries["aria"].special_ability == "OIL_EMBARGO"
    assert env.countries["ironhold"].special_ability == "INTIMIDATION"
    print("PASS: all countries have correct special abilities")

def test_reset_initializes_stability_and_reputation():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    for cid, c in env.countries.items():
        assert c.internal_stability == 70.0 and c.reputation == 50.0
    print("PASS: all countries start with correct stability and reputation")

def test_step_still_works():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    result = env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert result.done is False and result.turn == 1 and result.oil == 95
    print("PASS: step() still works with new country data")

def test_nps_history_tracks():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert len(env.countries["aria"].nps_history) == 3
    print("PASS: NPS history tracks across steps")

if __name__ == "__main__":
    tests = [test_reset_creates_correct_countries_task1, test_reset_creates_correct_countries_task3,
             test_aria_starting_resources, test_all_countries_have_unique_resources, test_nps_calculation,
             test_nps_rankings, test_observation_shows_own_exact_resources, test_task1_full_transparency,
             test_task2_hidden_info, test_value_to_tier, test_country_special_abilities,
             test_reset_initializes_stability_and_reputation, test_step_still_works, test_nps_history_tracks]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {e}"); f += 1
    print(f"\n{'='*40}\nResults: {p} passed, {f} failed out of {len(tests)}\n{'='*40}")
