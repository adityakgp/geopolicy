"""
GeoPolicyEnv — Subclasses openenv Environment base class.

Implements the three required methods:
    reset(**kwargs) → GeoObservation
    step(action)    → GeoObservation  (with .done and .reward set)
    state           → GeoState (property)

The openenv framework expects step() to return an Observation (not a dict).
The Observation's .done and .reward fields carry the RL signals.
"""

import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from models.action import GeoAction
from models.observation import GeoObservation, PublicCountryInfo
from models.state import GeoState
from models.country import Country
from config.countries import COUNTRIES, TASK_COUNTRIES
from config.constants import (
    TASK_CONFIG,
    NPS_WEIGHTS,
    NPS_ALLIANCE_BONUS,
    NPS_TRADE_BONUS,
    NPS_WAR_PENALTY,
    STABILITY_WEIGHT,
    RESOURCE_TIERS,
    STARTING_STABILITY,
    STARTING_REPUTATION,
    NATURAL_RECOVERY,
)
from server.actions import resolve_action
from server.rewards import calculate_step_reward
from server.events import EventsEngine
from tasks.graders import grade


def value_to_tier(value: float) -> str:
    """Convert an exact resource value to a tier string."""
    for tier_name, (low, high) in RESOURCE_TIERS.items():
        if low <= value < high:
            return tier_name
    return "very_high"


def calculate_nps(country: Country) -> float:
    """Calculate National Power Score for a country."""
    resource_score = (
        country.oil * NPS_WEIGHTS["oil"]
        + country.water * NPS_WEIGHTS["water"]
        + country.food * NPS_WEIGHTS["food"]
        + country.military * NPS_WEIGHTS["military"]
        + country.economy * NPS_WEIGHTS["economy"]
    )
    diplomacy_score = (
        len(country.alliances) * NPS_ALLIANCE_BONUS
        + len(country.trade_agreements) * NPS_TRADE_BONUS
        - len(country.at_war_with) * NPS_WAR_PENALTY
    )
    stability_score = country.internal_stability * STABILITY_WEIGHT
    return round(resource_score + diplomacy_score + stability_score, 2)


class GeoPolicyEnv(Environment[GeoAction, GeoObservation, GeoState]):
    """
    OpenEnv-compliant geopolitical simulation environment.

    Subclasses openenv.Environment with typed generics:
        ActT = GeoAction
        ObsT = GeoObservation
        StateT = GeoState
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._episode_id: str = ""
        self._task_id: str = "task1"
        self._current_turn: int = 0
        self._max_turns: int = 10
        self._done: bool = False
        self._step_count: int = 0

        self.countries: Dict[str, Country] = {}
        self.active_country_ids: List[str] = []

        self.hidden_info_enabled: bool = False
        self.global_events_enabled: bool = False
        self.special_abilities_enabled: bool = False

        self.events_engine: EventsEngine = EventsEngine()

    # ==================== REQUIRED: reset() ====================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GeoObservation:
        """Start a new episode. Returns initial observation."""
        task_id = kwargs.get("task_id", "task1")

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_id = task_id
        self._current_turn = 0
        self._done = False
        self._step_count = 0

        config = TASK_CONFIG.get(task_id, TASK_CONFIG["task1"])
        self._max_turns = config["max_turns"]
        self.hidden_info_enabled = config["hidden_info_enabled"]
        self.global_events_enabled = config["global_events_enabled"]
        self.special_abilities_enabled = config["special_abilities_enabled"]

        self.active_country_ids = TASK_COUNTRIES.get(task_id, TASK_COUNTRIES["task1"])
        self.countries = {}

        for cid in self.active_country_ids:
            country_config = COUNTRIES[cid]
            country = Country(cid, country_config)
            country.internal_stability = STARTING_STABILITY
            country.reputation = STARTING_REPUTATION
            country.current_nps = calculate_nps(country)
            country.nps_history = [country.current_nps]
            self.countries[cid] = country

        self.events_engine = EventsEngine()
        if self.global_events_enabled:
            self.events_engine.schedule_next()

        first_country = self.active_country_ids[0]
        return self._build_observation(first_country)

    # ==================== REQUIRED: step() ====================

    def step(
        self,
        action: GeoAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GeoObservation:
        """Process one action and advance the environment.

        Returns a GeoObservation with .done and .reward set
        (these are inherited from openenv Observation base).
        """
        # Auto-reset if step called before reset
        if not self.active_country_ids:
            self.reset(task_id="task1")

        acting_country = action.source_country or self.active_country_ids[0]
        if acting_country not in self.countries:
            acting_country = self.active_country_ids[0]

        # Save pre-step NPS
        prev_nps = {cid: c.current_nps for cid, c in self.countries.items()}

        # Resolve the action
        action_result = resolve_action(action, self.countries)

        # Natural recovery + decay timers
        self._current_turn += 1
        self._step_count += 1
        self._tick_timers()

        # Global events
        if self.global_events_enabled:
            self.events_engine.tick(self.countries, self._current_turn)

        # Recalculate NPS + check bankruptcy/collapse
        self._update_nps_and_status()

        # Check done
        self._done = self._current_turn >= self._max_turns
        solvent = [cid for cid, c in self.countries.items() if not c.is_bankrupt]
        if len(solvent) <= 1 and len(self.countries) > 1:
            self._done = True

        # Calculate reward
        rankings = self.get_rankings()
        reward = calculate_step_reward(
            country_id=acting_country,
            prev_nps=prev_nps[acting_country],
            new_nps=self.countries[acting_country].current_nps,
            action_result=action_result,
            rankings=rankings,
        )

        # Build observation with done and reward set
        obs = self._build_observation(acting_country)
        obs.done = self._done
        obs.reward = round(reward, 4)
        obs.metadata = {
            "action_type": action.action_type,
            "action_result": action_result,
            "turn": self._current_turn,
            "rankings": rankings,
        }
        return obs

    # ==================== REQUIRED: state (property) ====================

    @property
    def state(self) -> GeoState:
        """Return current episode metadata."""
        return GeoState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            done=self._done,
            task_id=self._task_id,
            num_countries=len(self.active_country_ids),
            current_turn=self._current_turn,
            max_turns=self._max_turns,
        )

    # ==================== REQUIRED: close() ====================

    def close(self) -> None:
        """Clean up resources (nothing to clean for our env)."""
        pass

    # ==================== step_all (our custom method for inference) ====================

    def step_all(self, actions: Dict[str, GeoAction]) -> Dict[str, GeoObservation]:
        """Process ALL countries' actions for one turn simultaneously.

        This is NOT part of the openenv spec — it's our helper for inference.py
        where all 5 countries act each turn.
        """
        prev_nps = {cid: c.current_nps for cid, c in self.countries.items()}

        # Resolve all actions
        action_results = {}
        for cid in self.active_country_ids:
            if cid in actions:
                action_results[cid] = resolve_action(actions[cid], self.countries)
            else:
                fallback = GeoAction(action_type="WAIT", source_country=cid)
                action_results[cid] = resolve_action(fallback, self.countries)

        # Advance turn once
        self._current_turn += 1
        self._step_count += 1
        self._tick_timers()

        # Global events
        if self.global_events_enabled:
            self.events_engine.tick(self.countries, self._current_turn)

        # Update NPS
        self._update_nps_and_status()

        # Check done
        self._done = self._current_turn >= self._max_turns
        solvent = [cid for cid, c in self.countries.items() if not c.is_bankrupt]
        if len(solvent) <= 1 and len(self.countries) > 1:
            self._done = True

        # Build results for each country
        rankings = self.get_rankings()
        results = {}
        for cid in self.active_country_ids:
            reward = calculate_step_reward(
                country_id=cid,
                prev_nps=prev_nps[cid],
                new_nps=self.countries[cid].current_nps,
                action_result=action_results.get(cid, {}),
                rankings=rankings,
            )
            obs = self._build_observation(cid)
            obs.done = self._done
            obs.reward = round(reward, 4)
            obs.metadata = {
                "action_result": action_results.get(cid, {}),
                "turn": self._current_turn,
                "rankings": rankings,
            }
            results[cid] = obs

        return results

    # ==================== Helper methods ====================

    def _tick_timers(self):
        """Natural recovery + decay all timers. Called once per turn."""
        for cid, country in self.countries.items():
            if not country.is_bankrupt:
                country.oil += NATURAL_RECOVERY
                country.water += NATURAL_RECOVERY
                country.food += NATURAL_RECOVERY
                country.military += NATURAL_RECOVERY
                country.economy += NATURAL_RECOVERY

            # Decay spy intel
            expired = [k for k, v in country.spied_on.items() if v <= 1]
            for k in expired:
                del country.spied_on[k]
            for k in country.spied_on:
                country.spied_on[k] -= 1

            # Decay counter-intel
            if country.counter_intel_active > 0:
                country.counter_intel_active -= 1

            # Decay special ability cooldown
            if country.special_ability_cooldown > 0:
                country.special_ability_cooldown -= 1
                if country.special_ability_cooldown <= 0:
                    country.special_ability_used = False

            # Clear defend flag
            country._defend_active = False
            country.clamp_resources()

    def _update_nps_and_status(self):
        """Recalculate NPS and check bankruptcy/collapse for all countries."""
        for cid, country in self.countries.items():
            if country.economy <= 0:
                country.is_bankrupt = True
            if country.internal_stability <= 0:
                country.is_collapsed = True
            country.current_nps = calculate_nps(country)
            country.nps_history.append(country.current_nps)

    def _build_observation(self, country_id: str) -> GeoObservation:
        """Build an observation for one country with info asymmetry."""
        country = self.countries[country_id]

        other_countries = {}
        for other_id, other in self.countries.items():
            if other_id == country_id:
                continue

            if self.hidden_info_enabled:
                has_intel = other_id in country.spied_on and country.spied_on[other_id] > 0
                public_info = PublicCountryInfo(
                    country_id=other_id,
                    country_name=other.name,
                    archetype=other.archetype,
                    military_tier=value_to_tier(other.military),
                    economy_tier=value_to_tier(other.economy),
                    oil_tier=value_to_tier(other.oil),
                    water_tier=value_to_tier(other.water),
                    food_tier=value_to_tier(other.food),
                    known_alliances=list(other.alliances),
                    known_trade_agreements=list(other.trade_agreements),
                    at_war_with=list(other.at_war_with),
                    reputation_tier=value_to_tier(other.reputation),
                    exact_oil=other.oil if has_intel else None,
                    exact_water=other.water if has_intel else None,
                    exact_food=other.food if has_intel else None,
                    exact_military=other.military if has_intel else None,
                    exact_economy=other.economy if has_intel else None,
                )
            else:
                public_info = PublicCountryInfo(
                    country_id=other_id,
                    country_name=other.name,
                    archetype=other.archetype,
                    military_tier=value_to_tier(other.military),
                    economy_tier=value_to_tier(other.economy),
                    oil_tier=value_to_tier(other.oil),
                    water_tier=value_to_tier(other.water),
                    food_tier=value_to_tier(other.food),
                    known_alliances=list(other.alliances),
                    known_trade_agreements=list(other.trade_agreements),
                    at_war_with=list(other.at_war_with),
                    reputation_tier=value_to_tier(other.reputation),
                    exact_oil=other.oil,
                    exact_water=other.water,
                    exact_food=other.food,
                    exact_military=other.military,
                    exact_economy=other.economy,
                )
            other_countries[other_id] = public_info

        task_config = TASK_CONFIG.get(self._task_id, TASK_CONFIG["task1"])

        return GeoObservation(
            country_id=country_id,
            country_name=country.name,
            archetype=country.archetype,
            turn=self._current_turn,
            max_turns=self._max_turns,
            oil=country.oil,
            water=country.water,
            food=country.food,
            military=country.military,
            economy=country.economy,
            internal_stability=country.internal_stability,
            reputation=country.reputation,
            alliances=list(country.alliances),
            trade_agreements=list(country.trade_agreements),
            at_war_with=list(country.at_war_with),
            embargoes_received=list(country.embargoes_received),
            embargoes_placed=list(country.embargoes_placed),
            special_ability=country.special_ability,
            special_ability_description=country.special_ability_description,
            special_ability_used=country.special_ability_used,
            special_ability_cooldown=country.special_ability_cooldown,
            current_nps=country.current_nps,
            nps_history=list(country.nps_history),
            other_countries=other_countries,
            active_global_event=(
                self.events_engine.active_event["description"]
                if self.events_engine.active_event else None
            ),
            turns_until_next_event=self.events_engine.turns_until_next,
            task_id=self._task_id,
            task_objective=task_config["description"],
        )

    def get_observation(self, country_id: str) -> GeoObservation:
        """Public alias for _build_observation (used by inference.py)."""
        return self._build_observation(country_id)

    def get_rankings(self) -> List[str]:
        """Return country IDs sorted by NPS (highest first)."""
        return sorted(
            self.active_country_ids,
            key=lambda cid: self.countries[cid].current_nps,
            reverse=True,
        )

    def get_final_results(self) -> dict:
        """Collect all data needed by task graders."""
        rankings = self.get_rankings()
        alliances_ever = {}
        for cid, country in self.countries.items():
            count = sum(1 for a in country.actions_this_episode if a.get("alliance_formed"))
            alliances_ever[cid] = count

        return {
            "final_rankings": rankings,
            "final_nps": {cid: c.current_nps for cid, c in self.countries.items()},
            "bankrupt": {cid: c.is_bankrupt for cid, c in self.countries.items()},
            "ever_collapsed": {cid: c.is_collapsed for cid, c in self.countries.items()},
            "final_alliances": {cid: len(c.alliances) for cid, c in self.countries.items()},
            "alliances_ever_formed": alliances_ever,
            "final_stability": {cid: c.internal_stability for cid, c in self.countries.items()},
            "event_history": list(self.events_engine.event_history),
        }

    def grade_country(self, country_id: str) -> float:
        """Grade a single country's performance. Returns 0.0-1.0."""
        results = self.get_final_results()
        return grade(country_id, self._task_id, results)
