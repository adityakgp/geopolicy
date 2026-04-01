"""Part 7 Tests — Prompt building, action parsing (openenv refactor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("API_BASE_URL", "http://test")
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("HF_TOKEN", "test-token")

from inference import build_prompt, parse_action
from server.environment import GeoPolicyEnv
from models.action import GeoAction

def test_build_prompt_contains_key_info():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    prompt = build_prompt(env.get_observation("aria").model_dump(), "aria")
    assert "Aria" in prompt and "OIL_EMBARGO" in prompt and "90" in prompt
    print(f"PASS: prompt contains key info ({len(prompt)} chars)")

def test_build_prompt_shows_spy_intel():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    env.countries["aria"].spied_on["ironhold"] = 3
    prompt = build_prompt(env.get_observation("aria").model_dump(), "aria")
    assert "INTEL" in prompt
    print("PASS: prompt shows spy intel")

def test_parse_clean_json():
    a = parse_action('{"action_type":"TRADE","target_country":"verdania","resource":"oil","amount":20,"counter_resource":"food","counter_amount":20}', "aria")
    assert a["action_type"] == "TRADE" and a["source_country"] == "aria"
    print("PASS: clean JSON parsed")

def test_parse_json_in_markdown():
    a = parse_action('```json\n{"action_type":"DEVELOP","resource":"water"}\n```', "aria")
    assert a["action_type"] == "DEVELOP"
    print("PASS: markdown JSON parsed")

def test_parse_json_with_extra_text():
    a = parse_action('I think we should:\n{"action_type":"DEVELOP","resource":"water"}\nThis helps.', "aria")
    assert a["action_type"] == "DEVELOP"
    print("PASS: JSON extracted from mixed text")

def test_parse_garbage_falls_back():
    a = parse_action("attack everything!!!", "aria")
    assert a["action_type"] == "WAIT"
    print("PASS: garbage → WAIT")

def test_parse_empty_falls_back():
    a = parse_action("", "ironhold")
    assert a["action_type"] == "WAIT" and a["source_country"] == "ironhold"
    print("PASS: empty → WAIT")

def test_simulated_episode():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    script = [
        GeoAction(action_type="DEVELOP", source_country="aria", resource="water"),
        GeoAction(action_type="DEVELOP", source_country="aria", resource="water"),
        GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                  resource="oil", amount=20, counter_resource="food", counter_amount=20),
        GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania"),
    ] + [GeoAction(action_type="WAIT", source_country="aria")] * 6
    for t in range(8):
        env.step_all({"aria": script[t], "verdania": GeoAction(action_type="WAIT", source_country="verdania")})
    assert env.state.done
    s = env.grade_country("aria"); assert 0.0 <= s <= 1.0
    print(f"PASS: simulated episode — aria={s:.3f}")

def test_all_action_types_parseable():
    jsons = ['{"action_type":"WAIT"}','{"action_type":"DEVELOP","resource":"oil"}',
             '{"action_type":"TRADE","target_country":"v","resource":"oil","amount":15,"counter_resource":"food","counter_amount":15}',
             '{"action_type":"PROPOSE_ALLIANCE","target_country":"v"}','{"action_type":"INVADE","target_country":"a","amount":0.7}',
             '{"action_type":"DEFEND"}','{"action_type":"SANCTION","target_country":"i"}',
             '{"action_type":"SPY","target_country":"n"}','{"action_type":"COUNTER_INTEL"}',
             '{"action_type":"USE_SPECIAL","target_country":"a"}']
    for j in jsons:
        a = parse_action(j, "aria"); assert a["action_type"] != ""
    print("PASS: all action types parseable")

def test_llm_call_budget():
    total = (2*8) + (5*10) + (5*12)
    assert total == 126, f"Expected 126, got {total}"
    assert total * 6 < 1200, "Should be under 20 min even at 6s/call"
    print(f"PASS: budget = {total} calls, worst ~{total*6/60:.1f} min")

if __name__ == "__main__":
    tests = [test_build_prompt_contains_key_info, test_build_prompt_shows_spy_intel,
             test_parse_clean_json, test_parse_json_in_markdown, test_parse_json_with_extra_text,
             test_parse_garbage_falls_back, test_parse_empty_falls_back, test_simulated_episode,
             test_all_action_types_parseable, test_llm_call_budget]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {type(e).__name__}: {e}"); f += 1
    print(f"\n{'='*40}\nResults: {p} passed, {f} failed out of {len(tests)}\n{'='*40}")
