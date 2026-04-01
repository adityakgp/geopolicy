"""Part 4 Tests — Diplomacy + Military (openenv refactor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.environment import GeoPolicyEnv
from models.action import GeoAction

def _ar(result): return result.metadata["action_result"]

def test_propose_alliance_accepted():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    r = env.step(GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania"))
    assert _ar(r)["success"] and _ar(r).get("alliance_formed")
    assert "verdania" in env.countries["aria"].alliances
    print("PASS: alliance formed successfully")

def test_alliance_rejected_low_reputation():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.countries["aria"].reputation = 10
    r = env.step(GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania"))
    assert _ar(r).get("fallback") or _ar(r).get("alliance_rejected")
    print("PASS: alliance rejected when reputation too low")

def test_alliance_rejected_at_war():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.countries["aria"].at_war_with.append("verdania")
    env.countries["verdania"].at_war_with.append("aria")
    r = env.step(GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania"))
    assert not _ar(r).get("alliance_formed")
    print("PASS: can't ally with wartime enemy")

def test_duplicate_alliance_rejected():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.countries["aria"].alliances.append("verdania"); env.countries["verdania"].alliances.append("aria")
    r = env.step(GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania"))
    assert _ar(r)["success"] is False
    print("PASS: duplicate alliance rejected")

def test_break_alliance():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.countries["aria"].alliances.append("verdania"); env.countries["verdania"].alliances.append("aria")
    r = env.step(GeoAction(action_type="BREAK_ALLIANCE", source_country="aria", target_country="verdania"))
    assert _ar(r).get("alliance_broken") and "verdania" not in env.countries["aria"].alliances
    print("PASS: alliance broken")

def test_invade_strong_beats_weak():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.8))
    assert _ar(r).get("war_won")
    print(f"PASS: invasion won (atk={_ar(r)['attack_power']} vs def={_ar(r)['defense_power']})")

def test_invade_weak_loses_to_strong():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="INVADE", source_country="aqualis", target_country="ironhold", amount=0.8))
    assert _ar(r).get("war_lost")
    print("PASS: invasion lost")

def test_alliance_boosts_defense():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    env.countries["aqualis"].alliances = ["verdania", "nexus"]
    r = env.step(GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.4))
    assert _ar(r).get("war_lost"), f"Should lose: atk={_ar(r)['attack_power']}, def={_ar(r)['defense_power']}"
    print("PASS: alliances boost defense")

def test_invade_cancels_trade():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    env.countries["ironhold"].trade_agreements.append("aqualis")
    env.countries["aqualis"].trade_agreements.append("ironhold")
    env.step(GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.8))
    assert "aqualis" not in env.countries["ironhold"].trade_agreements
    print("PASS: invasion cancels trade")

def test_cannot_invade_ally():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    env.countries["ironhold"].alliances.append("aqualis")
    r = env.step(GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.8))
    assert _ar(r).get("fallback")
    print("PASS: can't invade ally")

def test_defend_boosts_defense():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="DEFEND", source_country="aqualis"))
    assert _ar(r)["success"] and _ar(r).get("defend_active")
    print("PASS: DEFEND action succeeded")

def test_sanction_damages_target():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    eco_b = env.countries["aqualis"].economy
    r = env.step(GeoAction(action_type="SANCTION", source_country="aria", target_country="aqualis"))
    assert _ar(r).get("sanctioned") and env.countries["aqualis"].economy < eco_b
    print("PASS: sanction works")

def test_threaten_strong_extracts_tribute():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="THREATEN", source_country="ironhold", target_country="aqualis"))
    assert _ar(r).get("threat_complied")
    print("PASS: threat complied")

def test_threaten_weak_backfires():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="THREATEN", source_country="aqualis", target_country="ironhold"))
    assert _ar(r).get("empty_threat")
    print("PASS: empty threat backfires")

def test_negotiate_peace():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    env.countries["aria"].at_war_with.append("ironhold"); env.countries["ironhold"].at_war_with.append("aria")
    r = env.step(GeoAction(action_type="NEGOTIATE_PEACE", source_country="aria", target_country="ironhold"))
    assert _ar(r).get("peace_negotiated")
    print("PASS: peace negotiated")

def test_negotiate_peace_not_at_war():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    r = env.step(GeoAction(action_type="NEGOTIATE_PEACE", source_country="aria", target_country="ironhold"))
    assert _ar(r).get("fallback")
    print("PASS: peace rejected when not at war")

def test_invasion_reward_differs():
    e1 = GeoPolicyEnv(); e1.reset(task_id="task3")
    r1 = e1.step(GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.8))
    e2 = GeoPolicyEnv(); e2.reset(task_id="task3")
    r2 = e2.step(GeoAction(action_type="INVADE", source_country="aqualis", target_country="ironhold", amount=0.8))
    assert r1.reward > r2.reward
    print(f"PASS: invasion rewards differ (winner={r1.reward}, loser={r2.reward})")

def test_all_rewards_in_range():
    env = GeoPolicyEnv()
    actions = [
        ("task3", GeoAction(action_type="WAIT", source_country="aria")),
        ("task3", GeoAction(action_type="DEFEND", source_country="ironhold")),
        ("task3", GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania")),
        ("task3", GeoAction(action_type="SANCTION", source_country="nexus", target_country="aqualis")),
        ("task3", GeoAction(action_type="THREATEN", source_country="ironhold", target_country="aqualis")),
        ("task3", GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.5)),
    ]
    for tid, a in actions:
        env.reset(task_id=tid)
        r = env.step(a)
        assert 0.0 <= r.reward <= 1.0
    print("PASS: all action rewards in [0.0, 1.0]")

if __name__ == "__main__":
    tests = [test_propose_alliance_accepted, test_alliance_rejected_low_reputation,
             test_alliance_rejected_at_war, test_duplicate_alliance_rejected, test_break_alliance,
             test_invade_strong_beats_weak, test_invade_weak_loses_to_strong, test_alliance_boosts_defense,
             test_invade_cancels_trade, test_cannot_invade_ally, test_defend_boosts_defense,
             test_sanction_damages_target, test_threaten_strong_extracts_tribute, test_threaten_weak_backfires,
             test_negotiate_peace, test_negotiate_peace_not_at_war, test_invasion_reward_differs, test_all_rewards_in_range]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {e}"); f += 1
    print(f"\n{'='*40}\nResults: {p} passed, {f} failed out of {len(tests)}\n{'='*40}")
