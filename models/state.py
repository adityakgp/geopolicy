"""
State Model — Subclasses openenv State base class.

The openenv State base already has:
    episode_id: Optional[str] = None
    step_count: int = 0

We add our task-specific metadata on top.
"""

from openenv.core.env_server import State


class GeoState(State):
    """Episode-level metadata. Inherits episode_id and step_count from openenv."""

    done: bool = False
    task_id: str = "task1"
    num_countries: int = 2
    current_turn: int = 0
    max_turns: int = 10
