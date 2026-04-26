"""
Env v2 smoke test — task3, all 5 countries played by gpt-4o-mini.

Verifies:
- Hidden objectives are assigned and visible to the focal country only
- Per-rubric reward components are surfaced in step metadata
- Final grade includes per-rubric breakdown
- Full 12-turn episode runs without crashes
- gpt-4o-mini handles the v2 prompt (including hidden objective) cleanly

Logs everything to a timestamped subdirectory of test_runs/.

Usage:
    OPENAI_API_KEY=sk-... python test_runs/run_task3_smoke.py
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional

# Make geopolicy/ root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from server.environment import GeoPolicyEnv
from models.action import GeoAction
from inference import parse_action, build_prompt as _v1_build_prompt


# ============================================================
# Setup
# ============================================================

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("OPENAI_API_KEY not set")

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=API_KEY)

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"env_v2_task3_{TIMESTAMP}")
os.makedirs(LOG_DIR, exist_ok=True)


def log_path(name: str) -> str:
    return os.path.join(LOG_DIR, name)


# ============================================================
# Prompt with hidden objective (env v2 addition)
# ============================================================

SYSTEM_PROMPT = """You are a strategic advisor for a country in a geopolitical simulation.
You receive your country's state and must choose ONE action for this turn.

Some episodes give your country a SECRET OBJECTIVE in addition to the public goal.
Other countries do NOT know your secret objective. Pursue it without revealing it openly.

RESPOND WITH ONLY A VALID JSON OBJECT. No explanation, no markdown, no extra text.

AVAILABLE ACTIONS:
- WAIT: Do nothing. Recover +3 all resources, +5 stability. Cost: 0
- DEVELOP: Grow a resource. Fields: resource (oil/water/food/military/economy). Cost: 20 economy
- TRADE: Exchange resources. Fields: target_country, resource, amount, counter_resource, counter_amount. Cost: 5 economy
- PROPOSE_ALLIANCE: Form defense pact. Fields: target_country. Cost: 0
- BREAK_ALLIANCE: Exit alliance. Fields: target_country. Cost: 0
- INVADE: Military attack. Fields: target_country, amount (0.1-1.0 force level). Cost: 30 economy
- DEFEND: Boost defense this turn. Cost: 0
- SANCTION: Economic pressure. Fields: target_country. Cost: 10 economy
- THREATEN: Demand tribute. Fields: target_country. Cost: 5 economy
- NEGOTIATE_PEACE: End a war. Fields: target_country. Cost: 10 economy
- SPY: Reveal rival's exact resources. Fields: target_country. Cost: 25 economy
- COUNTER_INTEL: Block spy attempts for 3 turns. Cost: 15 economy
- USE_SPECIAL: Use your unique ability. Fields: target_country. Cost: 20 economy

JSON FORMAT:
{"action_type": "ACTION_NAME", "target_country": "country_id_or_null", "resource": "resource_or_null", "amount": number_or_null, "counter_resource": "resource_or_null", "counter_amount": number_or_null}"""


def build_prompt_v2(obs_dict: dict, country_id: str) -> str:
    """Build a prompt including the hidden objective for this country.

    Uses the existing v1 prompt builder, then prepends the secret objective.
    """
    base = _v1_build_prompt(obs_dict, country_id)
    obj_id = obs_dict.get("hidden_objective_id")
    if not obj_id:
        return base

    obj_name = obs_dict.get("hidden_objective_name") or obj_id
    obj_desc = obs_dict.get("hidden_objective_description") or ""

    secret_block = (
        f"\n--- YOUR SECRET OBJECTIVE ---\n"
        f"{obj_name}: {obj_desc}\n"
        f"This is HIDDEN from other countries. Pursue it strategically.\n"
        f"-----------------------------\n"
    )
    # Insert the secret block right after the OBJECTIVE: line
    return base.replace(
        f"OBJECTIVE: {obs_dict['task_objective']}",
        f"OBJECTIVE: {obs_dict['task_objective']}{secret_block}",
    )


def obs_to_dict(obs) -> dict:
    """Pydantic obs → plain dict for prompt building."""
    d = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
    # other_countries are PublicCountryInfo pydantic objects — they're dumped
    # as nested dicts by model_dump, but the v1 prompt builder expects dicts.
    return d


def call_llm(obs_dict: dict, country_id: str, log_handle) -> dict:
    """Ask gpt-4o-mini for an action for this country."""
    prompt = build_prompt_v2(obs_dict, country_id)
    log_handle.write(f"\n=== PROMPT FOR {country_id} (turn {obs_dict.get('turn')}) ===\n")
    log_handle.write(prompt + "\n")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=150,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        log_handle.write(f"\n=== RESPONSE FROM {country_id} ===\n{raw}\n")
        action = parse_action(raw, country_id)
        return action
    except Exception as e:
        log_handle.write(f"\n=== ERROR for {country_id} ===\n{e}\n")
        return {"action_type": "WAIT", "source_country": country_id}


# ============================================================
# Run the episode
# ============================================================


def main():
    print(f"[smoke] log dir: {LOG_DIR}")
    print(f"[smoke] starting task3 episode with {MODEL}")

    env = GeoPolicyEnv()
    env.reset(task_id="task3", seed=42)

    # Open log files
    prompts_log = open(log_path("prompts_responses.txt"), "w")
    turns_jsonl = open(log_path("turns.jsonl"), "w")

    # Record assignments
    assignments = {cid: env.countries[cid].hidden_objective for cid in env.active_country_ids}
    print(f"[smoke] hidden objective assignments:")
    for cid, oid in assignments.items():
        print(f"          {cid:10s} → {oid}")

    summary = {
        "model": MODEL,
        "task": "task3",
        "seed": 42,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hidden_objectives": assignments,
        "turns": [],
        "final_grades": {},
        "wall_clock_seconds": None,
    }
    t_start = time.time()

    # Run 12 turns (max_turns for task3)
    for turn_idx in range(12):
        # Build per-country actions in parallel-ish (sequential calls but state is snapshot at turn start)
        actions: Dict[str, GeoAction] = {}
        for cid in env.active_country_ids:
            obs = env.get_observation(cid)
            obs_dict = obs_to_dict(obs)
            action_dict = call_llm(obs_dict, cid, prompts_log)
            try:
                actions[cid] = GeoAction(**action_dict)
            except Exception as e:
                prompts_log.write(f"[!] failed to build GeoAction for {cid}: {e}; falling back to WAIT\n")
                actions[cid] = GeoAction(action_type="WAIT", source_country=cid)

        # Step all countries simultaneously
        results = env.step_all(actions)

        # Log turn
        turn_record = {
            "turn": env._current_turn,
            "actions": {cid: actions[cid].action_type + (
                f"->{actions[cid].target_country}" if actions[cid].target_country else ""
            ) for cid in env.active_country_ids},
            "rewards": {cid: results[cid].reward for cid in env.active_country_ids},
            "reward_components": {
                cid: results[cid].metadata["reward_components"]
                for cid in env.active_country_ids
            },
            "rankings": env.get_rankings(),
            "nps": {cid: env.countries[cid].current_nps for cid in env.active_country_ids},
            "alliances": {cid: list(env.countries[cid].alliances) for cid in env.active_country_ids},
            "at_war": {cid: list(env.countries[cid].at_war_with) for cid in env.active_country_ids},
        }
        summary["turns"].append(turn_record)
        turns_jsonl.write(json.dumps(turn_record) + "\n")

        print(f"[smoke] turn {env._current_turn}/12  rewards: " + " ".join(
            f"{cid}={results[cid].reward:.2f}" for cid in env.active_country_ids
        ))

        if any(obs.done for obs in results.values()):
            print(f"[smoke] episode terminated at turn {env._current_turn}")
            break

    # Final grades with per-rubric breakdown
    print(f"\n[smoke] === final grades ===")
    for cid in env.active_country_ids:
        detailed = env.grade_country_detailed(cid)
        summary["final_grades"][cid] = {
            "objective": assignments[cid],
            "total": detailed["total"],
            "components": detailed["components"],
        }
        print(f"  {cid:10s} obj={assignments[cid]:18s} grade={detailed['total']:.3f}  "
              f"comp={ {k: round(v, 2) for k, v in detailed['components'].items()} }")

    summary["wall_clock_seconds"] = round(time.time() - t_start, 2)
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Write summary
    with open(log_path("summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    prompts_log.close()
    turns_jsonl.close()

    print(f"\n[smoke] DONE in {summary['wall_clock_seconds']}s")
    print(f"[smoke] logs in: {LOG_DIR}")


if __name__ == "__main__":
    main()
