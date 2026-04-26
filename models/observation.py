"""
Observation Model — Subclasses openenv Observation base class.

The openenv Observation base already has:
    done: bool = False
    reward: Optional[float] = None
    metadata: dict = {}

We add our geopolitical observation fields on top.
The 'done' and 'reward' fields are inherited — no need to redeclare them.
"""

from openenv.core.env_server import Observation
from pydantic import BaseModel
from typing import Dict, List, Optional


class PublicCountryInfo(BaseModel):
    """What one country can see about another — intentionally incomplete."""

    country_id: str
    country_name: str
    archetype: str

    military_tier: str
    economy_tier: str
    oil_tier: str
    water_tier: str
    food_tier: str
    known_alliances: List[str] = []
    known_trade_agreements: List[str] = []
    at_war_with: List[str] = []
    reputation_tier: str = "neutral"

    # Hidden — only revealed by SPY action
    exact_oil: Optional[float] = None
    exact_water: Optional[float] = None
    exact_food: Optional[float] = None
    exact_military: Optional[float] = None
    exact_economy: Optional[float] = None


class GeoObservation(Observation):
    """What a country-agent sees after each turn.

    Inherits from openenv Observation which provides:
        done: bool (episode finished?)
        reward: Optional[float] (step reward)
        metadata: dict
    """

    # Identity
    country_id: str = ""
    country_name: str = ""
    archetype: str = ""
    turn: int = 0
    max_turns: int = 10

    # Own resources (FULL visibility)
    oil: float = 0.0
    water: float = 0.0
    food: float = 0.0
    military: float = 0.0
    economy: float = 0.0
    internal_stability: float = 0.0
    reputation: float = 0.0

    # Diplomatic status
    alliances: List[str] = []
    trade_agreements: List[str] = []
    at_war_with: List[str] = []
    embargoes_received: List[str] = []
    embargoes_placed: List[str] = []

    # Special ability
    special_ability: str = ""
    special_ability_description: str = ""
    special_ability_used: bool = False
    special_ability_cooldown: int = 0

    # Score
    current_nps: float = 0.0
    nps_history: List[float] = []

    # Other countries (PARTIAL visibility)
    other_countries: Dict[str, PublicCountryInfo] = {}

    # Global context
    active_global_event: Optional[str] = None
    turns_until_next_event: int = 0

    # Task info
    task_id: str = "task1"
    task_objective: str = ""

    # Hidden objective (env v2). Visible only to the country whose view this is.
    # Other countries see no hidden_objective field on PublicCountryInfo.
    hidden_objective_id: Optional[str] = None
    hidden_objective_name: Optional[str] = None
    hidden_objective_description: Optional[str] = None
