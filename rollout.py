"""
Generic rollout wrapper for GeoPolicyEnv.

Decouples the LLM (or any policy) from the env loop. Same function will be used by:
- inference.py (LLM via HF API)
- GRPO training (local Qwen 7B + LoRA, adapter on/off per country)
- Eval (any model)

Usage:
    from rollout import play_one_rollout
    from server.environment import GeoPolicyEnv

    env = GeoPolicyEnv()
    env.reset(task_id="task3")

    def my_generate(obs_dict, country_id):
        return {"action_type": "WAIT"}  # or any valid GeoAction kwargs

    history, reward = play_one_rollout(my_generate, env, country_id="ironhold")
    # history: list of per-turn dicts (obs, actions, rewards, rankings, results)
    # reward: final grade for "ironhold" in [0, 1]
"""

from typing import Callable, Dict, List, Tuple

from models.action import GeoAction
from server.environment import GeoPolicyEnv


def play_one_rollout(
    generate_fn: Callable[[dict, str], dict],
    env: GeoPolicyEnv,
    country_id: str = "ironhold",
) -> Tuple[List[dict], float]:
    """Play one full episode using `generate_fn` as the policy for ALL countries.

    Args:
        generate_fn: Callable taking (observation_dict, country_id) and returning
            an action dict (e.g. {"action_type": "WAIT"} or
            {"action_type": "INVADE", "target_country": "aqualis", "amount": 0.5}).
            `source_country` is filled in automatically — callers don't need to set it.
            If it raises or returns invalid data, the country falls back to WAIT.
        env: GeoPolicyEnv that has already been reset (env.state.done == False).
        country_id: Which country's final score to return as the rollout reward.

    Returns:
        history: list of per-turn dicts. Each entry contains:
            - turn (int)
            - obs (dict[country_id -> obs_dict] — state at the START of the turn)
            - actions (dict[country_id -> action_dict])
            - rewards (dict[country_id -> float] — per-step reward, not final grade)
            - rankings (list[country_id], best first)
            - results (dict[country_id -> action_result_dict] — what actually happened)
        final_reward: env.grade_country(country_id) after episode ends, in [0, 1].
    """
    if country_id not in env.active_country_ids:
        raise ValueError(
            f"country_id={country_id!r} not in active_country_ids={env.active_country_ids}"
        )

    history: List[dict] = []

    while not env.state.done:
        # 1. Snapshot observations for all countries BEFORE acting
        obs_dict = {
            cid: env.get_observation(cid).model_dump()
            for cid in env.active_country_ids
        }

        # 2. Generate actions (with WAIT fallback on error)
        actions: Dict[str, GeoAction] = {}
        for cid in env.active_country_ids:
            try:
                raw = generate_fn(obs_dict[cid], cid) or {}
                if not isinstance(raw, dict):
                    raise TypeError(f"generate_fn returned {type(raw).__name__}, expected dict")
                raw = {**raw, "source_country": cid}
                actions[cid] = GeoAction(**raw)
            except Exception:
                actions[cid] = GeoAction(action_type="WAIT", source_country=cid)

        # 3. Step env (advances turn, applies all 5 actions, computes rewards)
        results = env.step_all(actions)

        # 4. Record this turn's full data
        first_cid = env.active_country_ids[0]
        history.append({
            "turn": env.state.current_turn,
            "obs": obs_dict,
            "actions": {cid: actions[cid].model_dump() for cid in env.active_country_ids},
            "rewards": {cid: results[cid].reward for cid in env.active_country_ids},
            "rankings": results[first_cid].metadata.get("rankings", []),
            "results": {
                cid: results[cid].metadata.get("action_result", {})
                for cid in env.active_country_ids
            },
        })

    final_reward = env.grade_country(country_id)
    return history, final_reward
