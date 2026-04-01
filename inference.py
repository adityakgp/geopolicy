"""
inference.py — GeoPolicy LLM Agent

Runs LLM-powered country agents through all 3 tasks and produces scores.

Required environment variables:
    API_BASE_URL  - LLM API endpoint
    MODEL_NAME    - Model identifier
    HF_TOKEN      - API key

Constraints:
    - Must complete in < 20 minutes
    - Must run on 2 vCPU, 8 GB RAM
    - Must use OpenAI client library
    - Must produce scores for all 3 tasks

LLM call budget:
    Task 1: 2 countries × 10 turns = 20 calls
    Task 2: 5 countries × 15 turns = 75 calls
    Task 3: 5 countries × 20 turns = 100 calls
    Total: ~195 calls (at ~3-5s each = 10-16 minutes)
"""

import os
import json
import re
import time
from openai import OpenAI

# ---- Required environment variables (competition spec) ----
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---- Logging control ----
VERBOSE = os.environ.get("VERBOSE", "1") == "1"  # set VERBOSE=0 to silence
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")  # one timestamp per run

_log_file = None  # current task log file handle


def log(msg, indent=0):
    """Print to terminal AND write to current task log file."""
    prefix = "  " * indent
    line = f"{prefix}{msg}"
    if VERBOSE:
        print(line)
    if _log_file:
        _log_file.write(line + "\n")


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
- Only invade if your military × force > target military × (1 + 0.3 × their allies).
- Don't threaten stronger countries — it backfires.
- DEVELOP when economy is high and you have a clear weakness.
- WAIT when destabilized or broke. It's safe recovery.
- Your special ability is powerful but has 5-turn cooldown.

JSON FORMAT:
{"action_type": "ACTION_NAME", "target_country": "country_id_or_null", "resource": "resource_or_null", "amount": number_or_null, "counter_resource": "resource_or_null", "counter_amount": number_or_null}"""


def build_prompt(obs: dict, country_id: str) -> str:
    """Build a concise prompt from the observation."""
    lines = [
        f"COUNTRY: {obs['country_name']} ({obs['country_id']}) — {obs['archetype']}",
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
        f"SPECIAL ABILITY: {obs['special_ability']} — {obs['special_ability_description']}",
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

    # Economy hint to prevent wasteful actions
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
    """Clean up LLM-returned action dict before passing to GeoAction.

    Fixes common LLM issues:
    - "null" string → None
    - "None" string → None
    - String numbers → float
    """
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
        t0 = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=150,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        elapsed = time.time() - t0
        raw_text = response.choices[0].message.content.strip()
        action = parse_action(raw_text, country_id)

        log(f"[{country_id}] LLM ({elapsed:.1f}s) → {action.get('action_type', '?')}", indent=2)

        # Log details for interesting actions
        atype = action.get("action_type", "")
        if atype == "TRADE":
            log(f"  Trade: {action.get('resource')} ×{action.get('amount')} → "
                f"{action.get('target_country')} for {action.get('counter_resource')} ×{action.get('counter_amount')}", indent=2)
        elif atype == "INVADE":
            log(f"  Invade: {action.get('target_country')} with force={action.get('amount')}", indent=2)
        elif atype == "PROPOSE_ALLIANCE":
            log(f"  Alliance proposal → {action.get('target_country')}", indent=2)
        elif atype == "DEVELOP":
            log(f"  Develop: {action.get('resource')}", indent=2)
        elif atype == "SPY":
            log(f"  Spy on: {action.get('target_country')}", indent=2)
        elif atype == "USE_SPECIAL":
            log(f"  Special ability → {action.get('target_country')}", indent=2)
        elif atype == "THREATEN":
            log(f"  Threaten: {action.get('target_country')}", indent=2)
        elif atype == "SANCTION":
            log(f"  Sanction: {action.get('target_country')}", indent=2)
        elif atype == "NEGOTIATE_PEACE":
            log(f"  Peace with: {action.get('target_country')}", indent=2)

        # If LLM returned garbage, log it
        if raw_text and atype == "WAIT" and "WAIT" not in raw_text.upper():
            log(f"  [PARSE FALLBACK] Raw LLM: {raw_text[:100]}...", indent=2)

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

    # Open log file for this task (timestamped to preserve previous runs)
    log_path = os.path.join(LOG_DIR, f"{task_id}_{_RUN_TIMESTAMP}.log")
    _log_file = open(log_path, "w")

    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}  (log: logs/{task_id}.log)")
    print(f"{'='*60}")
    _log_file.write(f"{'='*60}\n  TASK: {task_id.upper()}\n{'='*60}\n")

    env.reset(task_id=task_id)

    log(f"Countries: {env.active_country_ids}")
    log(f"Max turns: {env.state.max_turns}")
    log(f"Events: {'ON' if env.global_events_enabled else 'OFF'}")
    log(f"Hidden info: {'ON' if env.hidden_info_enabled else 'OFF'}")

    # Show starting state
    log("\n--- Starting Resources ---")
    for cid in env.active_country_ids:
        c = env.countries[cid]
        log(f"  {c.name:10s} ({cid:10s}): {format_resources(c)}  NPS:{c.current_nps:.1f}")

    turn_start = time.time()

    while not env.state.done:
        turn = env.state.current_turn + 1
        print(f"\n--- Turn {turn}/{env.state.max_turns} ---")

        # Get action for each country from LLM
        actions = {}
        for cid in env.active_country_ids:
            obs = env.get_observation(cid)
            action_dict = get_agent_action(obs.model_dump(), cid)
            actions[cid] = GeoAction(**action_dict)

        # Execute all actions
        results = env.step_all(actions)

        # Log action outcomes
        for cid in env.active_country_ids:
            r = results[cid]
            ar = r.metadata.get("action_result", {})
            outcome = ""

            if ar.get("trade_successful"):
                gave = ar.get("gave", {})
                got = ar.get("received", {})
                outcome = f"✓ Traded {gave.get('amount',0):.0f} {gave.get('resource','')} for {got.get('amount',0):.0f} {got.get('resource','')}"
            elif ar.get("trade_rejected"):
                outcome = f"✗ Trade rejected: {ar.get('reason','')}"
            elif ar.get("alliance_formed"):
                outcome = f"✓ Allied with {ar.get('target','')}"
            elif ar.get("alliance_broken"):
                outcome = f"✓ Broke alliance with {ar.get('target','')}"
            elif ar.get("war_won"):
                looted = ar.get("looted", {})
                outcome = f"⚔ WON invasion vs {ar.get('target','')} (atk:{ar.get('attack_power',0):.0f} > def:{ar.get('defense_power',0):.0f}) Looted: {looted}"
            elif ar.get("war_lost"):
                outcome = f"⚔ LOST invasion vs {ar.get('target','')} (atk:{ar.get('attack_power',0):.0f} < def:{ar.get('defense_power',0):.0f})"
            elif ar.get("threat_complied"):
                outcome = f"✓ Threat: {ar.get('target','')} paid {ar.get('tribute_gained',0):.0f} tribute"
            elif ar.get("empty_threat"):
                outcome = f"✗ Empty threat vs {ar.get('target','')} — reputation lost!"
            elif ar.get("peace_negotiated"):
                outcome = f"✓ Peace with {ar.get('target','')}"
            elif ar.get("spy_successful"):
                outcome = f"✓ Spied on {ar.get('target','')} — intel for {ar.get('intel_duration',0)} turns"
            elif ar.get("spy_caught"):
                outcome = f"✗ Spy caught by {ar.get('target','')}"
            elif ar.get("special_used"):
                outcome = f"★ {ar.get('ability','')}: {ar.get('effect','')}"
            elif ar.get("fallback"):
                outcome = f"⚠ FALLBACK to WAIT ({ar.get('fallback_reason','')})"
            elif ar.get("action") == "WAIT":
                outcome = "~ Resting"
            elif ar.get("action") == "DEVELOP":
                outcome = f"↑ Developed {ar.get('resource','')} (+{ar.get('gain',0)})"
            elif ar.get("action") == "DEFEND":
                outcome = "🛡 Defending"
            elif ar.get("sanctioned"):
                outcome = f"✓ Sanctioned {ar.get('target','')} (-{ar.get('target_economy_lost',0)} eco)"
            elif ar.get("counter_intel_active"):
                outcome = "🔒 Counter-intel active"

            log(f"  {cid:10s}: {outcome}  | reward:{r.reward:.3f}", indent=1)

        # Turn summary: rankings + NPS
        rankings = env.get_rankings()
        log(f"\n  Rankings: {' > '.join(rankings)}", indent=1)
        for cid in rankings:
            c = env.countries[cid]
            status_flags = []
            if c.alliances: status_flags.append(f"allies:{c.alliances}")
            if c.at_war_with: status_flags.append(f"WAR:{c.at_war_with}")
            if c.is_bankrupt: status_flags.append("BANKRUPT")
            if c.is_collapsed: status_flags.append("COLLAPSED")
            flags = " ".join(status_flags)
            log(f"    {c.name:10s} NPS:{c.current_nps:6.1f} | {format_resources(c)} {flags}", indent=1)

        # Log global event if active
        if env.events_engine.active_event:
            log(f"\n  🌍 EVENT: {env.events_engine.active_event['name']}", indent=1)

    elapsed = time.time() - turn_start

    # Final scores
    log(f"\n{'─'*60}")
    log(f"  {task_id.upper()} COMPLETE ({elapsed:.1f}s)")
    log(f"{'─'*60}")

    scores = {}
    final_rankings = env.get_rankings()
    for rank, cid in enumerate(final_rankings, 1):
        score = env.grade_country(cid)
        scores[cid] = score
        c = env.countries[cid]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        log(f"  {medal} #{rank} {c.name:10s}: score={score:.3f}  NPS={c.current_nps:.1f}  {format_resources(c)}")

    # Close log file
    _log_file.write(f"\nFinal scores: {scores}\n")
    _log_file.close()
    _log_file = None
    print(f"  Log saved: logs/{task_id}_{_RUN_TIMESTAMP}.log")

    return scores


def main():
    from server.environment import GeoPolicyEnv

    total_start = time.time()
    env = GeoPolicyEnv()
    all_scores = {}

    print(f"GeoPolicy Inference — Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")

    for task_id in ["task1", "task2", "task3"]:
        all_scores[task_id] = run_task(task_id, env)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  FINAL SCORES (total time: {total_elapsed:.1f}s)")
    print(f"{'='*60}")
    for task_id, scores in all_scores.items():
        avg = sum(scores.values()) / len(scores)
        print(f"  {task_id}: avg={avg:.3f}  |  {scores}")

    return all_scores


if __name__ == "__main__":
    main()
