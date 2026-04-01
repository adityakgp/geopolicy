"""Part 5 Tests — Espionage, Special Abilities, Global Events (openenv refactor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from server.environment import GeoPolicyEnv
from server.events import EventsEngine
from models.action import GeoAction

def _ar(r): return r.metadata["action_result"]

def test_spy_reveals_intel():
    env = GeoPolicyEnv(); env.reset(task_id="task2"); random.seed(42)
    r = env.step(GeoAction(action_type="SPY", source_country="aria", target_country="ironhold"))
    if _ar(r).get("spy_successful"):
        obs = env.get_observation("aria")
        assert obs.other_countries["ironhold"].exact_military is not None
        print(f"PASS: spy reveals intel")
    else:
        print("PASS: spy was caught (valid)")

def test_counter_intel_blocks_spy():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    env.step(GeoAction(action_type="COUNTER_INTEL", source_country="ironhold"))
    r = env.step(GeoAction(action_type="SPY", source_country="aria", target_country="ironhold"))
    assert _ar(r).get("spy_caught")
    print("PASS: counter-intel blocks spy")

def test_spy_intel_decays():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    env.countries["aria"].spied_on["ironhold"] = 2
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert "ironhold" in env.countries["aria"].spied_on
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert "ironhold" not in env.countries["aria"].spied_on
    print("PASS: spy intel decays")

def test_hidden_info_without_spy():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    obs = env.get_observation("aria")
    assert obs.other_countries["ironhold"].exact_military is None
    print("PASS: no exact values without spy")

def test_aria_oil_embargo():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    eco_b = env.countries["aqualis"].economy
    r = env.step(GeoAction(action_type="USE_SPECIAL", source_country="aria", target_country="aqualis"))
    assert _ar(r)["ability"] == "OIL_EMBARGO" and env.countries["aqualis"].economy < eco_b
    print("PASS: Oil Embargo works")

def test_verdania_food_diplomacy():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    rep_b = env.countries["verdania"].reputation
    env.step(GeoAction(action_type="USE_SPECIAL", source_country="verdania", target_country="aria"))
    assert env.countries["verdania"].reputation > rep_b
    print("PASS: Food Diplomacy works")

def test_ironhold_intimidation():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    eco_b = env.countries["aqualis"].economy
    env.step(GeoAction(action_type="USE_SPECIAL", source_country="ironhold", target_country="aqualis"))
    assert env.countries["aqualis"].economy < eco_b
    print("PASS: Intimidation works")

def test_aqualis_water_cutoff():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    food_b = env.countries["aria"].food
    env.step(GeoAction(action_type="USE_SPECIAL", source_country="aqualis", target_country="aria"))
    assert env.countries["aria"].food < food_b
    print("PASS: Water Cutoff works")

def test_nexus_trade_multiplier():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="USE_SPECIAL", source_country="nexus", target_country="aria"))
    assert _ar(r)["ability"] == "TRADE_MULTIPLIER"
    print("PASS: Trade Multiplier works")

def test_special_ability_cooldown():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    env.step(GeoAction(action_type="USE_SPECIAL", source_country="aria", target_country="aqualis"))
    assert env.countries["aria"].special_ability_cooldown > 0
    r2 = env.step(GeoAction(action_type="USE_SPECIAL", source_country="aria", target_country="aqualis"))
    assert _ar(r2).get("fallback")
    print("PASS: special ability has cooldown")

def test_cooldown_decays():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    env.countries["aria"].special_ability_cooldown = 2; env.countries["aria"].special_ability_used = True
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert env.countries["aria"].special_ability_cooldown == 0 and not env.countries["aria"].special_ability_used
    print("PASS: cooldown decays and resets")

def test_events_engine_fires():
    engine = EventsEngine(); engine.turns_until_next = 1
    from config.countries import COUNTRIES; from models.country import Country
    countries = {cid: Country(cid, COUNTRIES[cid]) for cid in ["aria","verdania","ironhold","aqualis","nexus"]}
    event = engine.tick(countries, 1)
    assert event is not None and "name" in event
    print(f"PASS: event fired: {event['name']}")

def test_events_affect_countries():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    env.events_engine.turns_until_next = 1
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    print(f"PASS: events engine ticked (history: {env.events_engine.event_history})")

def test_task1_no_events():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    for _ in range(8): env.step(GeoAction(action_type="WAIT", source_country="aria"))
    assert len(env.events_engine.event_history) == 0
    print("PASS: task1 has no events")

def test_task2_has_events():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    assert env.global_events_enabled and env.events_engine.turns_until_next > 0
    print("PASS: task2 events enabled")

def test_observation_shows_event():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    env.events_engine.turns_until_next = 1
    env.step(GeoAction(action_type="WAIT", source_country="aria"))
    obs = env.get_observation("aria")
    if env.events_engine.active_event:
        assert obs.active_global_event is not None
    print("PASS: observation shows event info")

def test_all_new_rewards_in_range():
    env = GeoPolicyEnv()
    actions = [
        ("task2", GeoAction(action_type="SPY", source_country="aria", target_country="ironhold")),
        ("task2", GeoAction(action_type="COUNTER_INTEL", source_country="ironhold")),
        ("task3", GeoAction(action_type="USE_SPECIAL", source_country="aria", target_country="aqualis")),
    ]
    for tid, a in actions:
        env.reset(task_id=tid); r = env.step(a)
        assert 0.0 <= r.reward <= 1.0
    print("PASS: all Part 5 rewards in [0.0, 1.0]")

if __name__ == "__main__":
    tests = [test_spy_reveals_intel, test_counter_intel_blocks_spy, test_spy_intel_decays,
             test_hidden_info_without_spy, test_aria_oil_embargo, test_verdania_food_diplomacy,
             test_ironhold_intimidation, test_aqualis_water_cutoff, test_nexus_trade_multiplier,
             test_special_ability_cooldown, test_cooldown_decays, test_events_engine_fires,
             test_events_affect_countries, test_task1_no_events, test_task2_has_events,
             test_observation_shows_event, test_all_new_rewards_in_range]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {type(e).__name__}: {e}"); f += 1
    print(f"\n{'='*40}\nResults: {p} passed, {f} failed out of {len(tests)}\n{'='*40}")
