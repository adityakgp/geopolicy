"""
Action Model — Subclasses openenv Action base class.

The openenv Action base already has: metadata (dict)
We add our geopolitical action fields on top.
"""

from openenv.core.env_server import Action
from typing import Optional


class GeoAction(Action):
    """A single geopolitical action submitted by a country-agent."""

    action_type: str = "WAIT"
    source_country: Optional[str] = None
    target_country: Optional[str] = None
    resource: Optional[str] = None
    amount: Optional[float] = None
    counter_resource: Optional[str] = None
    counter_amount: Optional[float] = None
    message: Optional[str] = None
