"""
inference.py — GeoPolicy LLM Agent

Runs LLM-powered country agents through all 3 tasks and produces scores.

MANDATORY environment variables:
    API_BASE_URL  - LLM API endpoint
    MODEL_NAME    - Model identifier
    HF_TOKEN      - API key

STDOUT FORMAT (required by evaluator):
    [START] task=<task_id> env=geopolicy model=<model_name>
    [STEP]  step=<n> action=<actions_summary> reward=<avg_reward> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<avg_score> rewards=<r1,r2,...,rn>

Constraints:
    - Must complete in < 20 minutes
    - Must run on 2 vCPU, 8 GB RAM
    - Must use OpenAI client library
    - Must produce scores for all 3 tasks
"""

import os
import json
import re
import time
from typing import List, Optional

from openai import OpenAI

# ---- Required environment variables (competition spec) ----
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

BENCHMARK = "geopolicy"

# ---- Logging control ----
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_log_file = None


def log(msg, indent=0):
    """Write to current task log file only (not stdout — stdout is for structured logs)."""
    prefix = "  " * indent
    line = f"{prefix}{msg}"
    if _log_file:
        _log_file.write(line + "\n")


# ---- Structured stdout logging (required by evaluator) ----

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---- System prompt (shared by all country agents) ----

SYSTEM_PROMPT = """You are a strategic advisor for a country in a geopolitical simulation.
You receive your country's state and must choose ONE action for this turn.

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

STRATEGY TIPS:
- Trade to fix weaknesses (low resources). Both sides benefit.
- Alliances give +30% defense and +10 NPS each.
- Only invade if your military x force > target military x (1 + 0.3 x their allies).
- Don't threaten stronger countries -- it backfires.
- DEVELOP when economy is high and you have a clear weakness.
- WAIT when destabilized or broke. It's safe recovery.
- Your special ability is powerful but has 5-turn cooldown.

JSON FORMAT:
{"action_type": "ACTION_NAME", "target_country": "country_id_or_null", "resource": "resource_or_null", "amount": number_or_null, "counter_resource": "resource_or_null", "counter_amount": number_or_null}"""


def build_prompt(obs: dict, country_id: str) -> str:
    """Build a concise prompt from the observation."""
    lines = [
        f"COUNTRY: {obs['country_name']} ({obs['country_id']}) -- {obs['archetype']}",
        f"TURN: {obs['turn']}/{obs['max_turns']}",
        f"OBJECTIVE: {obs['task_objective']}",
        "",
        "YOUR RESOURCES:",
        f"  Oil:{obs['oil']:.0f} Water:{obs['water']:.0f} Food:{obs['food']:.0f} "
        f"Military:{obs['military']:.0f} Economy:{obs['economy']:.0f}",
        f"  Stability:{obs['internal_stability']:.0f} Reputation:{obs['reputation']:.0f} NPS:{obs['current_nps']:.1f}",
        "",
        f"ALLIANCES: {obs['alliances'] or 'None'}",
        f"TRADE AGREEMENTS: {obs['trade_agreements'] or 'None'}",
        f"AT WAR WITH: {obs['at_war_with'] or 'None'}",
        f"EMBARGOES ON YOU: {obs['embargoes_received'] or 'None'}",
        "",
        f"SPECIAL ABILITY: {obs['special_ability']} -- {obs['special_ability_description']}",
    ]
    if not obs['special_ability_used'] and obs['special_ability_cooldown'] == 0:
        lines.append("  Available: YES")
    else:
        cd = obs['special_ability_cooldown']
        lines.append(f"  Available: No (cooldown: {cd} turns)")

    lines.append("\nOTHER COUNTRIES:")
    for rid, info in obs.get("other_countries", {}).items():
        if isinstance(info, dict):
            rival_line = f"  {info.get('country_name', rid)} ({rid}): "
            if info.get("exact_military") is not None:
                rival_line += (
                    f"oil={info['exact_oil']:.0f} water={info['exact_water']:.0f} "
                    f"food={info['exact_food']:.0f} mil={info['exact_military']:.0f} "
                    f"eco={info['exact_economy']:.0f} [INTEL]"
                )
            else:
                rival_line += (
                    f"oil={info.get('oil_tier','?')} water={info.get('water_tier','?')} "
                    f"food={info.get('food_tier','?')} mil={info.get('military_tier','?')} "
                    f"eco={info.get('economy_tier','?')}"
                )
            if info.get("at_war_with"):
                rival_line += f" WAR:{info['at_war_with']}"
            if info.get("known_alliances"):
                rival_line += f" ALLIES:{info['known_alliances']}"
            lines.append(rival_line)

    if obs.get("active_global_event"):
        lines.append(f"\nACTIVE EVENT: {obs['active_global_event']}")

    eco = obs['economy']
    affordable = []
    if eco >= 30: affordable.append("INVADE(30)")
    if eco >= 25: affordable.append("SPY(25)")
    if eco >= 20: affordable.append("DEVELOP(20)/SPECIAL(20)")
    if eco >= 15: affordable.append("COUNTER_INTEL(15)")
    if eco >= 10: affordable.append("SANCTION(10)/PEACE(10)")
    if eco >= 5: affordable.append("TRADE(5)/THREATEN(5)")
    affordable.append("WAIT/DEFEND/ALLY(free)")
    lines.append(f"\nYOU CAN AFFORD: {', '.join(affordable)}")

    lines.append("\nChoose your action. Respond with ONLY a JSON object.")
    return "\n".join(lines)


def _clean_action(action: dict, country_id: str) -> dict:
    """Clean up LLM-returned action dict before passing to GeoAction."""
    action["source_country"] = country_id
    for key in ["amount", "counter_amount"]:
        val = action.get(key)
        if isinstance(val, str):
            if val.lower() in ("null", "none", ""):
                action[key] = None
            else:
                try:
                    action[key] = float(val)
                except ValueError:
                    action[key] = None
    for key in ["target_country", "resource", "counter_resource"]:
        val = action.get(key)
        if isinstance(val, str) and val.lower() in ("null", "none", ""):
            action[key] = None
    return action


def parse_action(text: str, country_id: str) -> dict:
    """Parse LLM response into an action dict."""
    text = text.strip()

    # Strategy 1: direct JSON parse
    try:
        action = json.loads(text)
        return _clean_action(action, country_id)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            action = json.loads(json_match.group(1))
            return _clean_action(action, country_id)
        except json.JSONDecodeError:
            pass

    # Strategy 3: find any {...}
    brace_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if brace_match:
        try:
            action = json.loads(brace_match.group(0))
            return _clean_action(action, country_id)
        except json.JSONDecodeError:
            pass

    # Strategy 4: fallback
    return {"action_type": "WAIT", "source_country": country_id}


def get_agent_action(obs: dict, country_id: str) -> dict:
    """Call LLM to decide an action for one country."""
    prompt = build_prompt(obs, country_id)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=150,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw_text = response.choices[0].message.content.strip()
        action = parse_action(raw_text, country_id)
        log(f"[{country_id}] LLM -> {action.get('action_type', '?')}", indent=2)
        return action
    except Exception as e:
        log(f"[{country_id}] LLM ERROR: {e}", indent=2)
        return {"action_type": "WAIT", "source_country": country_id}


def format_resources(country) -> str:
    """One-line resource summary for a country."""
    return (f"O:{country.oil:.0f} W:{country.water:.0f} F:{country.food:.0f} "
            f"M:{country.military:.0f} E:{country.economy:.0f} "
            f"S:{country.internal_stability:.0f} R:{country.reputation:.0f}")


def run_task(task_id: str, env) -> dict:
    """Run one complete task and return scores for all countries."""
    global _log_file
    from models.action import GeoAction

    # Open log file
    log_path = os.path.join(LOG_DIR, f"{task_id}_{_RUN_TIMESTAMP}.log")
    _log_file = open(log_path, "w")
    _log_file.write(f"{'='*60}\n  TASK: {task_id.upper()}\n{'='*60}\n")

    env.reset(task_id=task_id)

    # ---- [START] ----
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    log(f"Countries: {env.active_country_ids}")
    log(f"Max turns: {env.state.max_turns}")

    # Log starting state to file
    for cid in env.active_country_ids:
        c = env.countries[cid]
        log(f"  {c.name:10s} ({cid:10s}): {format_resources(c)}  NPS:{c.current_nps:.1f}")

    step_num = 0
    all_rewards: List[float] = []
    error_msg = None

    try:
        while not env.state.done:
            step_num += 1

            # Get action for each country from LLM
            actions = {}
            for cid in env.active_country_ids:
                obs = env.get_observation(cid)
                action_dict = get_agent_action(obs.model_dump(), cid)
                actions[cid] = GeoAction(**action_dict)

            # Execute all actions
            results = env.step_all(actions)

            # Compute average reward across all countries this step
            step_rewards = []
            action_parts = []
            for cid in env.active_country_ids:
                r = results[cid]
                step_rewards.append(r.reward)
                atype = r.metadata.get("action_result", {}).get("action", actions[cid].action_type)
                action_parts.append(f"{cid}:{atype}")

            avg_reward = sum(step_rewards) / len(step_rewards)
            all_rewards.append(avg_reward)
            action_summary = "|".join(action_parts)

            # ---- [STEP] ----
            log_step(
                step=step_num,
                action=action_summary,
                reward=avg_reward,
                done=env.state.done,
                error=None,
            )

            # Detailed log to file
            for cid in env.active_country_ids:
                r = results[cid]
                ar = r.metadata.get("action_result", {})
                outcome = _describe_outcome(ar)
                log(f"  {cid:10s}: {outcome}  | reward:{r.reward:.3f}", indent=1)

            rankings = env.get_rankings()
            log(f"  Rankings: {' > '.join(rankings)}", indent=1)
            for cid in rankings:
                c = env.countries[cid]
                log(f"    {c.name:10s} NPS:{c.current_nps:6.1f} | {format_resources(c)}", indent=1)

    except Exception as e:
        error_msg = str(e)
        log(f"ERROR: {error_msg}")

    # Final scores via graders
    scores = {}
    final_rankings = env.get_rankings()
    for rank, cid in enumerate(final_rankings, 1):
        score = env.grade_country(cid)
        scores[cid] = score
        c = env.countries[cid]
        log(f"  #{rank} {c.name:10s}: score={score:.3f}  NPS={c.current_nps:.1f}")

    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    success = avg_score > 0.1 and error_msg is None

    # ---- [END] ----
    log_end(
        success=success,
        steps=step_num,
        score=avg_score,
        rewards=all_rewards,
    )

    # Close log file
    _log_file.write(f"\nFinal scores: {scores}\n")
    _log_file.close()
    _log_file = None

    return scores


def _describe_outcome(ar: dict) -> str:
    """Convert action result dict to a human-readable string for log files."""
    if ar.get("trade_successful"):
        gave = ar.get("gave", {})
        got = ar.get("received", {})
        return f"Traded {gave.get('amount',0):.0f} {gave.get('resource','')} for {got.get('amount',0):.0f} {got.get('resource','')}"
    elif ar.get("trade_rejected"):
        return f"Trade rejected: {ar.get('reason','')}"
    elif ar.get("alliance_formed"):
        return f"Allied with {ar.get('target','')}"
    elif ar.get("alliance_broken"):
        return f"Broke alliance with {ar.get('target','')}"
    elif ar.get("war_won"):
        return f"WON invasion vs {ar.get('target','')} (atk:{ar.get('attack_power',0):.0f} > def:{ar.get('defense_power',0):.0f})"
    elif ar.get("war_lost"):
        return f"LOST invasion vs {ar.get('target','')} (atk:{ar.get('attack_power',0):.0f} < def:{ar.get('defense_power',0):.0f})"
    elif ar.get("threat_complied"):
        return f"Threat: {ar.get('target','')} paid {ar.get('tribute_gained',0):.0f} tribute"
    elif ar.get("empty_threat"):
        return f"Empty threat vs {ar.get('target','')}"
    elif ar.get("peace_negotiated"):
        return f"Peace with {ar.get('target','')}"
    elif ar.get("spy_successful"):
        return f"Spied on {ar.get('target','')}"
    elif ar.get("spy_caught"):
        return f"Spy caught by {ar.get('target','')}"
    elif ar.get("special_used"):
        return f"{ar.get('ability','')}: {ar.get('effect','')}"
    elif ar.get("fallback"):
        return f"FALLBACK to WAIT ({ar.get('fallback_reason','')})"
    elif ar.get("action") == "WAIT":
        return "Resting"
    elif ar.get("action") == "DEVELOP":
        return f"Developed {ar.get('resource','')} (+{ar.get('gain',0)})"
    elif ar.get("action") == "DEFEND":
        return "Defending"
    elif ar.get("sanctioned"):
        return f"Sanctioned {ar.get('target','')}"
    elif ar.get("counter_intel_active"):
        return "Counter-intel active"
    return "Unknown"


def main():
    from server.environment import GeoPolicyEnv

    env = GeoPolicyEnv()
    all_scores = {}

    for task_id in ["task1", "task2", "task3"]:
        all_scores[task_id] = run_task(task_id, env)

    env.close()
    return all_scores


if __name__ == "__main__":
    main()
