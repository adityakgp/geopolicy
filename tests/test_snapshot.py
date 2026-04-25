"""
Test snapshot/restore on GeoPolicyEnv.

What we're verifying:
1. snapshot() returns an isolated deep copy
2. After mutation + restore, env state matches the snapshot exactly
3. The snapshot can be used multiple times (key requirement for GRPO 8-rollout setup)
4. RNG state is captured (events fire deterministically when restored)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.environment import GeoPolicyEnv
from models.action import GeoAction


def country_signature(env, cid):
    """Compact summary of a country's mutable state for comparison."""
    c = env.countries[cid]
    return (
        round(c.oil, 3), round(c.water, 3), round(c.food, 3),
        round(c.military, 3), round(c.economy, 3),
        round(c.internal_stability, 3), round(c.reputation, 3),
        round(c.current_nps, 3),
        tuple(sorted(c.alliances)),
        tuple(sorted(c.trade_agreements)),
        tuple(sorted(c.at_war_with)),
        c.special_ability_used,
        c.special_ability_cooldown,
    )


def env_signature(env):
    """Compact summary of full env state."""
    return {
        "turn": env.state.current_turn,
        "done": env.state.done,
        "task": env.state.task_id,
        "step_count": env.state.step_count,
        "active_event": env.events_engine.active_event,
        "event_history": list(env.events_engine.event_history),
        "countries": {cid: country_signature(env, cid) for cid in env.active_country_ids},
    }


def test_snapshot_isolation():
    """Snapshot must not change when env mutates."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    snap = env.snapshot()
    snap_aria_oil = snap["countries"]["aria"].oil

    actions = {cid: GeoAction(action_type="DEVELOP", source_country=cid, resource="oil") for cid in env.active_country_ids}
    env.step_all(actions)

    assert snap["countries"]["aria"].oil == snap_aria_oil, "snapshot mutated by env step!"
    assert env.countries["aria"].oil != snap_aria_oil, "env did not change after step?"
    print("✅ snapshot is isolated from env mutation")


def test_restore_matches_snapshot():
    """After mutation + restore, env state == snapshot state."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    snap = env.snapshot()
    sig_before = env_signature(env)

    # Mutate heavily
    actions = {
        "aria": GeoAction(action_type="INVADE", source_country="aria", target_country="aqualis", amount=0.5),
        "verdania": GeoAction(action_type="PROPOSE_ALLIANCE", source_country="verdania", target_country="aqualis"),
        "ironhold": GeoAction(action_type="DEVELOP", source_country="ironhold", resource="military"),
        "aqualis": GeoAction(action_type="DEFEND", source_country="aqualis"),
        "nexus": GeoAction(action_type="SPY", source_country="nexus", target_country="aria"),
    }
    env.step_all(actions)
    sig_after_mutation = env_signature(env)
    assert sig_after_mutation != sig_before, "mutation did not change state?"

    # Restore
    env.restore(snap)
    sig_after_restore = env_signature(env)
    assert sig_after_restore == sig_before, f"state mismatch after restore!\nbefore={sig_before}\nafter={sig_after_restore}"
    print("✅ restore brings env back to snapshot state")


def test_snapshot_reusable():
    """Same snapshot must produce same state across multiple restores (GRPO 8-rollout case)."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    snap = env.snapshot()

    signatures_after_restore = []
    for rollout in range(3):
        env.restore(snap)
        sig = env_signature(env)
        signatures_after_restore.append(sig)
        # Mutate so the next iteration must actually re-restore
        actions = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids}
        env.step_all(actions)
        env.step_all(actions)
        env.step_all(actions)

    # All 3 restored signatures should be identical
    assert signatures_after_restore[0] == signatures_after_restore[1] == signatures_after_restore[2], \
        "snapshot is being mutated across restore calls — GRPO would break"
    print("✅ snapshot is reusable across 3 restores")


def test_full_episode_after_restore():
    """A restored snapshot can finish a full episode and produce a valid grade."""
    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    snap = env.snapshot()

    # Mid-game mutation
    for _ in range(5):
        actions = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids}
        env.step_all(actions)

    # Restore and play full episode
    env.restore(snap)
    while not env.state.done:
        actions = {cid: GeoAction(action_type="WAIT", source_country=cid) for cid in env.active_country_ids}
        env.step_all(actions)

    score = env.grade_country("ironhold")
    assert 0.0 <= score <= 1.0, f"grade out of range: {score}"
    assert env.state.current_turn == env.state.max_turns, "episode did not run to completion"
    print(f"✅ full episode completes after restore: ironhold score = {score:.3f}")


def test_rng_captured():
    """Restoring should also restore RNG so event-based dynamics replay deterministically."""
    import random

    env = GeoPolicyEnv()
    env.reset(task_id="task3")
    snap = env.snapshot()

    # First run: take some random numbers after restore
    env.restore(snap)
    seq1 = [random.random() for _ in range(5)]

    # Second run: same restore, should give same sequence
    env.restore(snap)
    seq2 = [random.random() for _ in range(5)]

    assert seq1 == seq2, f"RNG not restored: {seq1} != {seq2}"
    print(f"✅ RNG state captured (deterministic sequence after restore)")


if __name__ == "__main__":
    test_snapshot_isolation()
    test_restore_matches_snapshot()
    test_snapshot_reusable()
    test_full_episode_after_restore()
    test_rng_captured()
    print("\n🎉 All snapshot/restore tests passed.")
