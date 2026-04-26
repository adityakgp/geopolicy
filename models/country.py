"""
Country Model — The full internal state of one country.

IMPORTANT distinction:
    Country     = the FULL truth (environment knows this)
    Observation = what ONE country can SEE (filtered view sent to agents)

The environment holds a Country object for each nation.
When an agent asks for its state, we convert Country → Observation,
hiding other countries' exact values behind tiers.
"""

from typing import Dict, List, Optional


class Country:
    """Full internal state of a single country."""

    def __init__(self, country_id: str, config: dict):
        # Identity
        self.country_id: str = country_id
        self.name: str = config["name"]
        self.archetype: str = config["archetype"]
        self.description: str = config["description"]

        # Resources (the core numbers)
        resources = config["starting_resources"]
        self.oil: float = float(resources["oil"])
        self.water: float = float(resources["water"])
        self.food: float = float(resources["food"])
        self.military: float = float(resources["military"])
        self.economy: float = float(resources["economy"])

        # Hidden reserves (only visible via SPY)
        self.hidden_reserve: dict = dict(config.get("hidden_reserve", {}))

        # Stability and reputation
        self.internal_stability: float = 70.0   # 0-100
        self.reputation: float = 50.0           # 0-100

        # Diplomatic state
        self.alliances: List[str] = []              # country_ids of allies
        self.trade_agreements: List[str] = []       # country_ids with active trades
        self.at_war_with: List[str] = []            # country_ids at war with
        self.embargoes_received: List[str] = []     # who is embargoing us
        self.embargoes_placed: List[str] = []       # who we are embargoing

        # Special ability
        self.special_ability: str = config["special_ability"]
        self.special_ability_description: str = config["special_ability_description"]
        self.special_ability_used: bool = False
        self.special_ability_cooldown: int = 0      # turns until available

        # Tracking
        self.nps_history: List[float] = []
        self.current_nps: float = 0.0
        self.actions_this_episode: List[dict] = []

        # Espionage tracking
        self.spied_on: Dict[str, int] = {}          # {country_id: turns_remaining} — known intel
        self.counter_intel_active: int = 0           # turns remaining of counter-intel

        # Status flags
        self.is_bankrupt: bool = False
        self.is_collapsed: bool = False     # stability hit 0

        # Hidden objective (env v2). Visible to this country only.
        # None means objective is disabled (Task 1) or not yet assigned.
        self.hidden_objective: Optional[str] = None

    def get_resource(self, resource_name: str) -> float:
        """Get a resource value by name."""
        return getattr(self, resource_name, 0.0)

    def set_resource(self, resource_name: str, value: float):
        """Set a resource value, clamped to >= 0."""
        setattr(self, resource_name, max(0.0, value))

    def clamp_resources(self):
        """Ensure no resource goes below 0."""
        self.oil = max(0.0, self.oil)
        self.water = max(0.0, self.water)
        self.food = max(0.0, self.food)
        self.military = max(0.0, self.military)
        self.economy = max(0.0, self.economy)
        self.internal_stability = max(0.0, min(100.0, self.internal_stability))
        self.reputation = max(0.0, min(100.0, self.reputation))

    def to_dict(self) -> dict:
        """Full serialization (for internal use / debugging)."""
        return {
            "country_id": self.country_id,
            "name": self.name,
            "archetype": self.archetype,
            "oil": self.oil,
            "water": self.water,
            "food": self.food,
            "military": self.military,
            "economy": self.economy,
            "internal_stability": self.internal_stability,
            "reputation": self.reputation,
            "alliances": self.alliances,
            "trade_agreements": self.trade_agreements,
            "at_war_with": self.at_war_with,
            "embargoes_received": self.embargoes_received,
            "embargoes_placed": self.embargoes_placed,
            "special_ability": self.special_ability,
            "special_ability_used": self.special_ability_used,
            "special_ability_cooldown": self.special_ability_cooldown,
            "current_nps": self.current_nps,
            "nps_history": self.nps_history,
            "is_bankrupt": self.is_bankrupt,
            "is_collapsed": self.is_collapsed,
            "hidden_objective": self.hidden_objective,
        }
