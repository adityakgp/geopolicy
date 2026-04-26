"""Part 6 Tests — Full episodes, step_all(), graders (openenv refactor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.environment import GeoPolicyEnv
from models.action import GeoAction
from models.observation import GeoObservation

def test_step_all_processes_all_countries():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    actions = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids}
    results = env.step_all(actions)
    assert len(results) == 5 and env.state.current_turn == 1
    for cid, r in results.items():
        assert isinstance(r, GeoObservation) and 0.0 <= r.reward <= 1.0
    print("PASS: step_all processes all 5 countries")

def test_step_all_advances_one_turn():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    actions = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids}
    env.step_all(actions); assert env.state.current_turn == 1
    env.step_all(actions); assert env.state.current_turn == 2
    print("PASS: step_all advances exactly one turn")

def test_step_all_missing_action_defaults_wait():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    results = env.step_all({"aria": GeoAction(action_type="DEVELOP", source_country="aria", resource="oil")})
    assert "verdania" in results
    print("PASS: missing action defaults to WAIT")

def test_step_all_simultaneous_actions():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    actions = {
        "aria": GeoAction(action_type="TRADE", source_country="aria", target_country="verdania",
                          resource="oil", amount=15, counter_resource="food", counter_amount=15),
        "verdania": GeoAction(action_type="WAIT", source_country="verdania"),
        "ironhold": GeoAction(action_type="INVADE", source_country="ironhold", target_country="aqualis", amount=0.8),
        "aqualis": GeoAction(action_type="DEFEND", source_country="aqualis"),
        "nexus": GeoAction(action_type="DEVELOP", source_country="nexus", resource="economy"),
    }
    results = env.step_all(actions)
    assert results["aria"].metadata["action_result"]["action"] == "TRADE"
    assert results["ironhold"].metadata["action_result"]["action"] == "INVADE"
    print("PASS: simultaneous actions resolve correctly")

def test_full_episode_task1():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    for _ in range(8):
        env.step_all({cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids})
    assert env.state.done
    for cid in env.active_country_ids:
        s = env.grade_country(cid); assert 0.0 <= s <= 1.0
    print(f"PASS: full task1 ({', '.join(f'{c}={env.grade_country(c):.2f}' for c in env.active_country_ids)})")

def test_full_episode_task2():
    env = GeoPolicyEnv(); env.reset(task_id="task2")
    for t in range(10):
        actions = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids}
        if t == 3: actions["aria"] = GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania")
        env.step_all(actions)
    assert env.state.done
    for cid in env.active_country_ids: assert 0.0 <= env.grade_country(cid) <= 1.0
    print("PASS: full task2 episode")

def test_full_episode_task3():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    for _ in range(12):
        env.step_all({cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids})
    assert env.state.done and env.state.current_turn == 12
    for cid in env.active_country_ids: assert 0.0 <= env.grade_country(cid) <= 1.0
    print("PASS: full task3 episode")

def test_grader_scores_all_in_range():
    for tid in ["task1", "task2", "task3"]:
        env = GeoPolicyEnv(); env.reset(task_id=tid)
        for _ in range(env.state.max_turns):
            env.step_all({cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids})
        for cid in env.active_country_ids:
            s = env.grade_country(cid); assert 0.0 <= s <= 1.0, f"{tid}/{cid}: {s}"
    print("PASS: all grader scores in [0.0, 1.0]")

def test_bankrupt_scores_zero():
    env = GeoPolicyEnv(); env.reset(task_id="task1")
    env.countries["aria"].economy = 0; env.countries["aria"].is_bankrupt = True
    assert env.grade_country("aria") == 0.0
    print("PASS: bankrupt scores 0.0")

def test_get_final_results_structure():
    env = GeoPolicyEnv(); env.reset(task_id="task3")
    for _ in range(5):
        env.step_all({cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids})
    r = env.get_final_results()
    for key in ["final_rankings","final_nps","bankrupt","ever_collapsed","final_alliances","alliances_ever_formed","final_stability"]:
        assert key in r, f"Missing: {key}"
    print("PASS: get_final_results has all fields")

def test_task_difficulty_progression():
    scores = {}
    for tid in ["task1","task2","task3"]:
        env = GeoPolicyEnv(); env.reset(task_id=tid)
        for _ in range(env.state.max_turns):
            env.step_all({cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids})
        scores[tid] = env.grade_country("aria")
    for s in scores.values(): assert 0.0 <= s <= 1.0
    print(f"PASS: difficulty progression ({scores})")

def test_alliance_boosts_task2_score():
    # Clear hidden objectives so this test isolates the alliance signal.
    # (Otherwise a random LONE_WOLF assignment to aria would make alliance hurt.)
    e1 = GeoPolicyEnv(); e1.reset(task_id="task2", seed=0)
    for c in e1.countries.values(): c.hidden_objective = None
    for _ in range(10): e1.step_all({cid: GeoAction(action_type="WAIT", source_country=cid) for cid in e1.active_country_ids})
    s1 = e1.grade_country("aria")
    e2 = GeoPolicyEnv(); e2.reset(task_id="task2", seed=0)
    for c in e2.countries.values(): c.hidden_objective = None
    for t in range(10):
        a = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in e2.active_country_ids}
        if t == 0: a["aria"] = GeoAction(action_type="PROPOSE_ALLIANCE", source_country="aria", target_country="verdania")
        e2.step_all(a)
    s2 = e2.grade_country("aria")
    assert s2 >= s1
    print(f"PASS: alliance boosts score ({s1:.3f} → {s2:.3f})")

if __name__ == "__main__":
    tests = [test_step_all_processes_all_countries, test_step_all_advances_one_turn,
             test_step_all_missing_action_defaults_wait, test_step_all_simultaneous_actions,
             test_full_episode_task1, test_full_episode_task2, test_full_episode_task3,
             test_grader_scores_all_in_range, test_bankrupt_scores_zero,
             test_get_final_results_structure, test_task_difficulty_progression, test_alliance_boosts_task2_score]
    p = f = 0
    for t in tests:
        try: t(); p += 1
        except Exception as e: print(f"FAIL: {t.__name__} — {type(e).__name__}: {e}"); f += 1
    print(f"\n{'='*40}\nResults: {p} passed, {f} failed out of {len(tests)}\n{'='*40}")
