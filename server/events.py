"""
Events Engine — Manages global events during an episode.

Responsibilities:
  - Schedule when the next event fires (random 3-5 turn interval)
  - Pick a random event from the pool
  - Apply the event's effects to all countries
  - Track active event and its remaining duration

HOW IT WORKS:
  Each turn, the environment calls events_engine.tick().
  If it's time for an event, one fires and affects all countries.
  The event stays active for its duration (2-5 turns).
  After it ends, a new event is scheduled.
"""

import random
from typing import Dict, Optional, List
from models.country import Country
from config.events import GLOBAL_EVENTS
from config.constants import EVENT_MIN_INTERVAL, EVENT_MAX_INTERVAL


class EventsEngine:
    """Manages the global events lifecycle."""

    def __init__(self):
        self.active_event: Optional[dict] = None     # currently active event
        self.active_event_turns_left: int = 0         # turns remaining
        self.turns_until_next: int = 0                # turns until next event fires
        self.event_history: List[str] = []            # IDs of past events
        self.used_events: List[str] = []              # avoid repeats in same episode

    def schedule_next(self):
        """Schedule when the next event will fire."""
        self.turns_until_next = random.randint(EVENT_MIN_INTERVAL, EVENT_MAX_INTERVAL)

    def tick(self, countries: Dict[str, Country], current_turn: int) -> Optional[dict]:
        """
        Called every turn. Returns event info dict if an event fires, else None.

        Logic:
        1. If an active event is running, decrement its timer
        2. If no active event, decrement countdown to next event
        3. If countdown hits 0, fire a new random event
        """
        # If active event still running, count down
        if self.active_event and self.active_event_turns_left > 0:
            self.active_event_turns_left -= 1
            if self.active_event_turns_left <= 0:
                self.active_event = None
                self.schedule_next()
            return self.active_event  # return current event (or None if just ended)

        # Count down to next event
        self.turns_until_next -= 1
        if self.turns_until_next <= 0:
            return self._fire_event(countries)

        return None

    def _fire_event(self, countries: Dict[str, Country]) -> dict:
        """Pick and apply a random event."""
        # Pick from events we haven't used yet (avoid repeats)
        available = [e for e in GLOBAL_EVENTS if e["id"] not in self.used_events]
        if not available:
            # All events used — reset pool
            self.used_events = []
            available = GLOBAL_EVENTS

        event = random.choice(available)
        self.active_event = event
        self.active_event_turns_left = event["duration"]
        self.event_history.append(event["id"])
        self.used_events.append(event["id"])

        # Apply effects
        self._apply_event(event, countries)

        return event

    def _apply_event(self, event: dict, countries: Dict[str, Country]):
        """Apply an event's effects to all countries."""
        eid = event["id"]
        effects = event["effects"]

        if eid == "oil_price_shock":
            for cid, c in countries.items():
                if c.oil >= effects["oil_rich_threshold"]:
                    c.economy += effects["oil_rich_economy_bonus"]
                elif c.oil < effects["oil_poor_threshold"]:
                    c.economy -= effects["oil_poor_economy_penalty"]
                    c.internal_stability -= effects["oil_poor_stability_penalty"]
                c.clamp_resources()

        elif eid == "famine_crisis":
            for cid, c in countries.items():
                c.food -= effects["all_food_loss"]
                if c.food < effects["food_poor_threshold"]:
                    c.internal_stability -= effects["food_poor_stability_penalty"]
                c.clamp_resources()

        elif eid == "water_wars":
            for cid, c in countries.items():
                if c.water < effects["water_poor_threshold"]:
                    c.food -= effects["water_poor_food_penalty"]
                    c.internal_stability -= effects["water_poor_stability_penalty"]
                if cid == "aqualis":
                    c.economy += effects["aqualis_bonus"]
                c.clamp_resources()

        elif eid == "global_recession":
            for cid, c in countries.items():
                c.economy -= effects["all_economy_loss"]
                if cid == "nexus":
                    c.economy -= effects["nexus_extra_economy_loss"]
                c.clamp_resources()

        elif eid == "un_sanctions_vote":
            # Find most aggressive country (most wars started)
            war_counts = {cid: len(c.at_war_with) for cid, c in countries.items()}
            if any(v > 0 for v in war_counts.values()):
                aggressor = max(war_counts, key=war_counts.get)
                countries[aggressor].economy -= effects["aggressor_economy_penalty"]
                countries[aggressor].reputation -= effects["aggressor_reputation_penalty"]
                countries[aggressor].clamp_resources()

            # Peaceful countries (no wars) gain reputation
            for cid, c in countries.items():
                if len(c.at_war_with) == 0:
                    c.reputation += effects["peaceful_reputation_bonus"]
                    c.clamp_resources()

        elif eid == "technology_breakthrough":
            for cid, c in countries.items():
                # High-economy countries benefit
                if c.economy >= effects["high_economy_threshold"]:
                    c.economy += effects["high_economy_bonus"]
                # Oil-heavy countries suffer (oil less valuable)
                if c.oil > 60:
                    c.economy -= effects["oil_dependent_penalty"]
                c.clamp_resources()

        elif eid == "military_escalation":
            for cid, c in countries.items():
                c.military += effects["all_military_bonus"]
                if cid == "ironhold":
                    c.military += effects["ironhold_military_bonus"]
                c.clamp_resources()

    def get_status(self) -> dict:
        """Return current event status for observations."""
        return {
            "active_event": self.active_event["name"] if self.active_event else None,
            "active_event_description": self.active_event["description"] if self.active_event else None,
            "turns_remaining": self.active_event_turns_left,
            "turns_until_next": self.turns_until_next if not self.active_event else 0,
            "event_history": list(self.event_history),
        }
