"""
Action Resolution — The rules of the game.

All 13 action types: WAIT, DEVELOP, TRADE, PROPOSE_ALLIANCE, BREAK_ALLIANCE,
INVADE, DEFEND, SANCTION, THREATEN, NEGOTIATE_PEACE, SPY, COUNTER_INTEL, USE_SPECIAL
"""

import random
from typing import Dict
from models.country import Country
from config.constants import (
    ACTION_COSTS,
    WAIT_RECOVERY, WAIT_STABILITY_BONUS, WAIT_REPUTATION_BONUS,
    DEVELOP_GAIN, DEVELOP_DIMINISHING,
    TRADE_REPUTATION_GAIN, TRADE_REJECT_PENALTY, TRADE_MAX_AMOUNT,
    ALLIANCE_DEFENSE_BONUS, ALLIANCE_REPUTATION_GAIN,
    ALLIANCE_BREAK_REP_PENALTY, ALLIANCE_MIN_REPUTATION,
    INVADE_WIN_RESOURCE_LOOT, INVADE_WIN_TARGET_MIL_LOSS,
    INVADE_WIN_TARGET_ECO_LOSS, INVADE_WIN_SOURCE_MIL_LOSS,
    INVADE_WIN_SOURCE_REP_LOSS, INVADE_LOSE_SOURCE_MIL_LOSS,
    INVADE_LOSE_SOURCE_ECO_LOSS, INVADE_LOSE_SOURCE_REP_LOSS,
    INVADE_LOSE_TARGET_REP_GAIN, WAR_DURATION,
    DEFEND_BONUS,
    SANCTION_TARGET_ECO_LOSS, SANCTION_SOURCE_ECO_LOSS, SANCTION_REPUTATION_LOSS,
    THREATEN_COMPLY_TRIBUTE, THREATEN_EMPTY_REP_LOSS,
    PEACE_STABILITY_GAIN, PEACE_REPUTATION_GAIN,
    SPY_INTEL_DURATION, SPY_CATCH_CHANCE, SPY_CAUGHT_REP_LOSS,
    COUNTER_INTEL_DURATION,
    SPECIAL_COOLDOWN, OIL_EMBARGO_DURATION, OIL_EMBARGO_ECO_PENALTY,
    FOOD_DIPLOMACY_REP_GAIN, WATER_CUTOFF_DURATION, WATER_CUTOFF_FOOD_PENALTY,
    INTIMIDATION_TRIBUTE, TRADE_MULTIPLIER_BONUS,
)

VALID_RESOURCES = {"oil", "water", "food", "military", "economy"}

# All 13 action types
VALID_ACTIONS = {
    "WAIT", "DEVELOP", "TRADE",
    "PROPOSE_ALLIANCE", "BREAK_ALLIANCE",
    "INVADE", "DEFEND",
    "SANCTION", "THREATEN", "NEGOTIATE_PEACE",
    "SPY", "COUNTER_INTEL", "USE_SPECIAL",
}

TARGETED_ACTIONS = {
    "TRADE", "PROPOSE_ALLIANCE", "BREAK_ALLIANCE",
    "INVADE", "SANCTION", "THREATEN", "NEGOTIATE_PEACE",
    "SPY", "USE_SPECIAL",
}


def validate_action(action, countries: Dict[str, Country]) -> dict:
    """Check if an action is valid."""
    # Normalize resource names to lowercase (LLMs sometimes return "Oil" instead of "oil")
    if action.resource:
        action.resource = action.resource.lower()
    if action.counter_resource:
        action.counter_resource = action.counter_resource.lower()

    atype = action.action_type.upper()

    if atype not in VALID_ACTIONS:
        return {"valid": False, "reason": f"Unknown action type: {atype}"}

    if action.source_country not in countries:
        return {"valid": False, "reason": f"Unknown source country: {action.source_country}"}

    source = countries[action.source_country]

    # Bankrupt countries can only WAIT
    if source.is_bankrupt and atype != "WAIT":
        return {"valid": False, "reason": f"{action.source_country} is bankrupt, can only WAIT"}

    # Economy cost check
    cost = ACTION_COSTS.get(atype, 0)
    if source.economy < cost:
        return {"valid": False, "reason": f"Not enough economy ({source.economy:.0f}) for {atype} (costs {cost})"}

    # Target validation for actions that need a target
    if atype in TARGETED_ACTIONS:
        if not action.target_country or action.target_country not in countries:
            return {"valid": False, "reason": f"{atype} requires a valid target_country"}
        if action.target_country == action.source_country:
            return {"valid": False, "reason": f"Cannot {atype} yourself"}

    # Action-specific checks
    if atype == "DEVELOP":
        if action.resource not in VALID_RESOURCES:
            return {"valid": False, "reason": f"Invalid resource: {action.resource}"}

    elif atype == "TRADE":
        if action.resource not in VALID_RESOURCES:
            return {"valid": False, "reason": f"Invalid trade resource: {action.resource}"}
        if action.counter_resource not in VALID_RESOURCES:
            return {"valid": False, "reason": f"Invalid counter resource: {action.counter_resource}"}
        if not action.amount or action.amount <= 0:
            return {"valid": False, "reason": "Trade amount must be positive"}
        if not action.counter_amount or action.counter_amount <= 0:
            return {"valid": False, "reason": "Counter amount must be positive"}
        if action.amount > TRADE_MAX_AMOUNT:
            return {"valid": False, "reason": f"Trade amount exceeds max ({TRADE_MAX_AMOUNT})"}
        if source.get_resource(action.resource) < action.amount:
            return {"valid": False, "reason": f"Not enough {action.resource} to trade"}

    elif atype == "BREAK_ALLIANCE":
        target = countries[action.target_country]
        if target.country_id not in source.alliances:
            return {"valid": False, "reason": f"No alliance with {action.target_country} to break"}

    elif atype == "NEGOTIATE_PEACE":
        target = countries[action.target_country]
        if target.country_id not in source.at_war_with:
            return {"valid": False, "reason": f"Not at war with {action.target_country}"}

    elif atype == "INVADE":
        if source.military <= 0:
            return {"valid": False, "reason": "No military to invade with"}
        if action.target_country in source.alliances:
            return {"valid": False, "reason": f"Cannot invade ally {action.target_country}. Break alliance first."}

    elif atype == "USE_SPECIAL":
        if source.special_ability_cooldown > 0:
            return {"valid": False, "reason": f"Special ability on cooldown ({source.special_ability_cooldown} turns)"}

    return {"valid": True}


# ==================== RESOLVERS ====================


def resolve_wait(source: Country) -> dict:
    """WAIT — Passive recovery. +3 all resources, +5 stability, +2 reputation."""
    source.oil += WAIT_RECOVERY
    source.water += WAIT_RECOVERY
    source.food += WAIT_RECOVERY
    source.military += WAIT_RECOVERY
    source.economy += WAIT_RECOVERY
    source.internal_stability += WAIT_STABILITY_BONUS
    source.reputation += WAIT_REPUTATION_BONUS
    source.clamp_resources()

    return {"action": "WAIT", "success": True}


def resolve_develop(source: Country, resource: str) -> dict:
    """DEVELOP — Spend 20 economy, grow one resource by 15 (or 8 if diminishing)."""
    source.economy -= ACTION_COSTS["DEVELOP"]

    recent_develops = [
        a for a in source.actions_this_episode[-3:]
        if a.get("action") == "DEVELOP" and a.get("resource") == resource
    ]
    gain = DEVELOP_DIMINISHING if recent_develops else DEVELOP_GAIN
    diminished = bool(recent_develops)

    source.set_resource(resource, source.get_resource(resource) + gain)
    source.clamp_resources()

    return {
        "action": "DEVELOP", "success": True,
        "resource": resource, "gain": gain, "diminished": diminished,
        "economy_spent": ACTION_COSTS["DEVELOP"],
    }


def resolve_trade(
    source: Country, target: Country, resource: str, amount: float,
    counter_resource: str, counter_amount: float,
) -> dict:
    """TRADE — Exchange resources. Uses fairness heuristic for acceptance."""
    source.economy -= ACTION_COSTS["TRADE"]

    # Rejection checks
    if target.get_resource(counter_resource) < counter_amount:
        source.clamp_resources()
        return {"action": "TRADE", "success": False, "trade_rejected": True,
                "reason": f"{target.country_id} lacks {counter_resource}"}

    if source.country_id in target.at_war_with:
        source.clamp_resources()
        return {"action": "TRADE", "success": False, "trade_rejected": True,
                "reason": "At war — cannot trade"}

    ratio = amount / max(counter_amount, 0.1)
    if ratio > 3.0 or ratio < 0.33:
        source.clamp_resources()
        return {"action": "TRADE", "success": False, "trade_rejected": True,
                "reason": f"Unfair ratio ({ratio:.1f}x)"}

    # Transfer resources
    source.set_resource(resource, source.get_resource(resource) - amount)
    source.set_resource(counter_resource, source.get_resource(counter_resource) + counter_amount)
    target.set_resource(counter_resource, target.get_resource(counter_resource) - counter_amount)
    target.set_resource(resource, target.get_resource(resource) + amount)

    source.reputation += TRADE_REPUTATION_GAIN
    target.reputation += TRADE_REPUTATION_GAIN

    if target.country_id not in source.trade_agreements:
        source.trade_agreements.append(target.country_id)
    if source.country_id not in target.trade_agreements:
        target.trade_agreements.append(source.country_id)

    source.clamp_resources()
    target.clamp_resources()

    return {"action": "TRADE", "success": True, "trade_successful": True,
            "target": target.country_id,
            "gave": {"resource": resource, "amount": amount},
            "received": {"resource": counter_resource, "amount": counter_amount}}


def resolve_propose_alliance(source: Country, target: Country) -> dict:
    """
    PROPOSE_ALLIANCE — Offer a mutual defense pact.

    Acceptance: target accepts if:
      - Not already allied
      - Not at war with source
      - Target reputation >= 30 (they have reasonable standing)

    If accepted:
      - Both add each other to alliances list
      - Both gain +5 reputation
      - Allies get +30% defense bonus when invaded

    WHY alliances matter:
      Aqualis (military=25) alone is easy to invade.
      Aqualis + Ironhold alliance? Attacker faces 25 + (95 * 0.3) = 53.5 defense.
      Alliances turn weak countries into viable ones.
    """
    # Already allied?
    if target.country_id in source.alliances:
        return {"action": "PROPOSE_ALLIANCE", "success": False,
                "reason": "Already allied", "alliance_rejected": True}

    # At war?
    if target.country_id in source.at_war_with:
        return {"action": "PROPOSE_ALLIANCE", "success": False,
                "reason": "At war — cannot ally", "alliance_rejected": True}

    # Reputation check — target won't ally with someone disreputable
    if source.reputation < ALLIANCE_MIN_REPUTATION:
        return {"action": "PROPOSE_ALLIANCE", "success": False,
                "reason": f"Source reputation too low ({source.reputation:.0f} < {ALLIANCE_MIN_REPUTATION})",
                "alliance_rejected": True}

    # Accepted!
    source.alliances.append(target.country_id)
    target.alliances.append(source.country_id)
    source.reputation += ALLIANCE_REPUTATION_GAIN
    target.reputation += ALLIANCE_REPUTATION_GAIN
    source.clamp_resources()
    target.clamp_resources()

    return {"action": "PROPOSE_ALLIANCE", "success": True,
            "alliance_formed": True, "target": target.country_id}


def resolve_break_alliance(source: Country, target: Country) -> dict:
    """
    BREAK_ALLIANCE — Exit an existing alliance.

    Effects:
      - Alliance removed for both
      - Source reputation -15 (betrayer penalty)
      - Target's reputation with source drops (they'll remember)

    WHY this exists:
      Alliances are powerful but not permanent. If your ally is losing
      and dragging you down, or if you want to invade someone your
      ally is protecting, you might need to break the alliance first.
      But it costs reputation — do it too often and nobody will trust you.
    """
    source.alliances.remove(target.country_id)
    target.alliances.remove(source.country_id)
    source.reputation -= ALLIANCE_BREAK_REP_PENALTY
    source.clamp_resources()

    return {"action": "BREAK_ALLIANCE", "success": True,
            "alliance_broken": True, "target": target.country_id}


def resolve_invade(source: Country, target: Country, force_level: float,
                   countries: Dict[str, Country]) -> dict:
    """
    INVADE — Military attack to seize resources.

    Combat math:
      attack_power  = source.military * force_level
      defense_power = target.military * (1 + num_allies * 0.3 + defend_bonus)

    If attacker wins (attack > defense):
      - Attacker takes 30% of target's oil, food, economy
      - Target military -30, economy -25
      - Attacker military -15 (war cost even when winning)
      - Attacker reputation -25 (aggressor penalty)

    If attacker loses:
      - Attacker military -20, economy -15
      - Attacker reputation -15
      - Defender reputation +10

    Both enter war status for 2 turns.

    WHY invasion is risky:
      Even winning costs you military and reputation. And if the target
      has allies, their combined defense might beat your attack.
      A failed invasion is devastating — you lose troops AND look weak.
    """
    source.economy -= ACTION_COSTS["INVADE"]

    # Force level: how much military to commit (0.1 to 1.0)
    force = max(0.1, min(1.0, force_level if force_level else 0.5))

    # Attack power
    attack_power = source.military * force

    # Defense power: base + alliance bonuses
    # Check if target used DEFEND this turn (tracked via defend_active flag)
    defend_bonus = DEFEND_BONUS if getattr(target, '_defend_active', False) else 0.0
    ally_count = len(target.alliances)
    defense_power = target.military * (1.0 + ally_count * ALLIANCE_DEFENSE_BONUS + defend_bonus)

    attacker_wins = attack_power > defense_power

    if attacker_wins:
        # Loot resources from target
        oil_loot = target.oil * INVADE_WIN_RESOURCE_LOOT
        food_loot = target.food * INVADE_WIN_RESOURCE_LOOT
        eco_loot = target.economy * INVADE_WIN_RESOURCE_LOOT

        source.oil += oil_loot
        source.food += food_loot
        source.economy += eco_loot

        target.oil -= oil_loot
        target.food -= food_loot
        target.economy -= eco_loot
        target.military -= INVADE_WIN_TARGET_MIL_LOSS
        target.economy -= INVADE_WIN_TARGET_ECO_LOSS

        source.military -= INVADE_WIN_SOURCE_MIL_LOSS
        source.reputation -= INVADE_WIN_SOURCE_REP_LOSS

        # Target stability drops from being invaded
        target.internal_stability -= 15

        result = {
            "action": "INVADE", "success": True, "war_won": True,
            "target": target.country_id,
            "attack_power": round(attack_power, 1),
            "defense_power": round(defense_power, 1),
            "looted": {"oil": round(oil_loot, 1), "food": round(food_loot, 1),
                       "economy": round(eco_loot, 1)},
        }
    else:
        # Failed invasion
        source.military -= INVADE_LOSE_SOURCE_MIL_LOSS
        source.economy -= INVADE_LOSE_SOURCE_ECO_LOSS
        source.reputation -= INVADE_LOSE_SOURCE_REP_LOSS
        target.reputation += INVADE_LOSE_TARGET_REP_GAIN

        # Attacker stability drops from humiliation
        source.internal_stability -= 10

        result = {
            "action": "INVADE", "success": False, "war_lost": True,
            "target": target.country_id,
            "attack_power": round(attack_power, 1),
            "defense_power": round(defense_power, 1),
        }

    # Both enter war status
    if target.country_id not in source.at_war_with:
        source.at_war_with.append(target.country_id)
    if source.country_id not in target.at_war_with:
        target.at_war_with.append(source.country_id)

    # Remove any trade agreements between warring countries
    if target.country_id in source.trade_agreements:
        source.trade_agreements.remove(target.country_id)
    if source.country_id in target.trade_agreements:
        target.trade_agreements.remove(source.country_id)

    source.clamp_resources()
    target.clamp_resources()
    return result


def resolve_defend(source: Country) -> dict:
    """
    DEFEND — Boost defense power for this turn.

    Sets a flag so if source is invaded this turn, they get +25% defense.
    No economy cost. Purely defensive.

    WHY: If you suspect someone will invade, DEFEND makes you harder to beat.
    But if nobody invades, you wasted a turn (could have DEVELOPED instead).
    It's a prediction/bluffing mechanic.
    """
    source._defend_active = True
    return {"action": "DEFEND", "success": True, "defend_active": True}


def resolve_sanction(source: Country, target: Country) -> dict:
    """
    SANCTION — Economic pressure. Both sides pay a cost.

    Effects:
      - Target economy -15
      - Source economy -5 (cost of enforcing)
      - Source reputation -10 globally (aggressive economic action)
      - Existing trade agreements between them suspended

    WHY sanctions exist:
      A non-military way to weaken a rival. Cheaper than invasion
      but also damages your reputation. Good for pressuring
      someone without starting a war.
    """
    source.economy -= ACTION_COSTS["SANCTION"]

    target.economy -= SANCTION_TARGET_ECO_LOSS
    source.economy -= SANCTION_SOURCE_ECO_LOSS
    source.reputation -= SANCTION_REPUTATION_LOSS

    # Add embargo tracking
    if target.country_id not in source.embargoes_placed:
        source.embargoes_placed.append(target.country_id)
    if source.country_id not in target.embargoes_received:
        target.embargoes_received.append(source.country_id)

    # Suspend trade agreements
    if target.country_id in source.trade_agreements:
        source.trade_agreements.remove(target.country_id)
    if source.country_id in target.trade_agreements:
        target.trade_agreements.remove(source.country_id)

    source.clamp_resources()
    target.clamp_resources()

    return {"action": "SANCTION", "success": True, "sanctioned": True,
            "target": target.country_id,
            "target_economy_lost": SANCTION_TARGET_ECO_LOSS}


def resolve_threaten(source: Country, target: Country) -> dict:
    """
    THREATEN — Issue an ultimatum. Target auto-responds based on power balance.

    If source military > target military:
      Target complies — pays 15 economy tribute to source.
    If source military <= target military:
      Target defies — source loses reputation (empty threat).

    WHY threats exist:
      Cheaper than invasion. Ironhold (military=95) can threaten Aqualis (military=25)
      and extract tribute without fighting. But threatening a stronger country
      backfires — you look weak.
    """
    source.economy -= ACTION_COSTS["THREATEN"]

    # Does the target comply? Based on raw military comparison.
    if source.military > target.military:
        # Target complies — pays tribute
        tribute = min(THREATEN_COMPLY_TRIBUTE, target.economy)
        target.economy -= tribute
        source.economy += tribute
        target.clamp_resources()
        source.clamp_resources()

        return {"action": "THREATEN", "success": True, "threat_complied": True,
                "target": target.country_id, "tribute_gained": tribute}
    else:
        # Target defies — source looks foolish
        source.reputation -= THREATEN_EMPTY_REP_LOSS
        source.clamp_resources()

        return {"action": "THREATEN", "success": False, "empty_threat": True,
                "target": target.country_id,
                "reason": "Target military is equal or stronger — threat defied"}


def resolve_negotiate_peace(source: Country, target: Country) -> dict:
    """
    NEGOTIATE_PEACE — End an ongoing war.

    Acceptance: target accepts if the war has lasted 1+ turns
    (newly started wars are too heated for peace).

    If accepted:
      - War status cleared for both
      - Both gain +5 stability, +8 reputation

    If rejected:
      - Source gains +5 reputation anyway (good look for trying)

    WHY peace matters:
      Being at war costs -15 NPS per enemy. Ending a war removes that penalty
      and lets both countries recover. Sometimes losing a war early and
      making peace is better than dragging it out.
    """
    source.economy -= ACTION_COSTS["NEGOTIATE_PEACE"]

    # Accept peace (simple heuristic: always accept if both at war)
    if source.country_id in target.at_war_with:
        source.at_war_with.remove(target.country_id)
        target.at_war_with.remove(source.country_id)

        source.internal_stability += PEACE_STABILITY_GAIN
        target.internal_stability += PEACE_STABILITY_GAIN
        source.reputation += PEACE_REPUTATION_GAIN
        target.reputation += PEACE_REPUTATION_GAIN

        source.clamp_resources()
        target.clamp_resources()

        return {"action": "NEGOTIATE_PEACE", "success": True,
                "peace_negotiated": True, "target": target.country_id}

    source.clamp_resources()
    return {"action": "NEGOTIATE_PEACE", "success": False,
            "reason": "Not at war with target"}


# ==================== ESPIONAGE ====================


def resolve_spy(source: Country, target: Country) -> dict:
    """
    SPY — Intelligence operation to reveal a rival's hidden resources.

    If target has COUNTER_INTEL active: spy always fails.
    Otherwise: 90% success, 10% caught.

    On success: source can see target's exact resources for 3 turns.
    On caught: source loses 20 reputation, target is alerted.

    WHY spying matters:
      In Task 2/3, you only see rival tiers ("high", "low").
      Before invading Ironhold, you want to know their EXACT military.
      Is "very_high" 81 or 120? That changes your invasion decision.
    """
    source.economy -= ACTION_COSTS["SPY"]

    # Counter-intel blocks spy completely
    if target.counter_intel_active > 0:
        source.reputation -= SPY_CAUGHT_REP_LOSS
        source.clamp_resources()
        return {"action": "SPY", "success": False, "spy_caught": True,
                "reason": "Counter-intelligence blocked the spy",
                "target": target.country_id}

    # Random catch chance
    if random.random() < SPY_CATCH_CHANCE:
        source.reputation -= SPY_CAUGHT_REP_LOSS
        source.clamp_resources()
        return {"action": "SPY", "success": False, "spy_caught": True,
                "reason": "Spy was caught",
                "target": target.country_id}

    # Success — record intel
    source.spied_on[target.country_id] = SPY_INTEL_DURATION
    source.clamp_resources()

    return {"action": "SPY", "success": True, "spy_successful": True,
            "target": target.country_id,
            "intel_duration": SPY_INTEL_DURATION}


def resolve_counter_intel(source: Country) -> dict:
    """
    COUNTER_INTEL — Activate counter-intelligence for 3 turns.

    While active, any SPY action against this country automatically fails.
    Source is also notified who tried.

    WHY: If you suspect a rival is about to spy on you before an invasion,
    counter-intel protects your hidden info.
    """
    source.economy -= ACTION_COSTS["COUNTER_INTEL"]
    source.counter_intel_active = COUNTER_INTEL_DURATION
    source.clamp_resources()

    return {"action": "COUNTER_INTEL", "success": True,
            "counter_intel_active": True,
            "duration": COUNTER_INTEL_DURATION}


# ==================== SPECIAL ABILITIES ====================


def resolve_use_special(source: Country, target: Country,
                        countries: Dict[str, Country]) -> dict:
    """
    USE_SPECIAL — Activate country's unique ability.

    Each country has one special ability with a 5-turn cooldown:

    ARIA - OIL_EMBARGO: Cut oil to target, their economy -30%/turn for 2 turns
    VERDANIA - FOOD_DIPLOMACY: Give food aid, gain +15 reputation with target
    IRONHOLD - INTIMIDATION: Force tribute of 20 economy from target
    AQUALIS - WATER_CUTOFF: Halve target's food production for 3 turns
    NEXUS - TRADE_MULTIPLIER: All trades this turn yield 20% more
    """
    source.economy -= ACTION_COSTS["USE_SPECIAL"]
    ability = source.special_ability

    # Set cooldown
    source.special_ability_used = True
    source.special_ability_cooldown = SPECIAL_COOLDOWN

    if ability == "OIL_EMBARGO":
        # Aria cuts oil — target economy drops 30% immediately
        eco_loss = target.economy * OIL_EMBARGO_ECO_PENALTY
        target.economy -= eco_loss
        target.clamp_resources()
        source.clamp_resources()
        return {"action": "USE_SPECIAL", "success": True, "ability": "OIL_EMBARGO",
                "special_used": True, "target": target.country_id,
                "effect": f"Economy embargo: target lost {eco_loss:.0f} economy"}

    elif ability == "FOOD_DIPLOMACY":
        # Verdania gives food aid — massive reputation boost
        food_gift = min(20, source.food)
        source.food -= food_gift
        target.food += food_gift
        source.reputation += FOOD_DIPLOMACY_REP_GAIN
        target.reputation += 5  # small gratitude
        source.clamp_resources()
        target.clamp_resources()
        return {"action": "USE_SPECIAL", "success": True, "ability": "FOOD_DIPLOMACY",
                "special_used": True, "target": target.country_id,
                "effect": f"Food aid: gave {food_gift:.0f} food, +{FOOD_DIPLOMACY_REP_GAIN} reputation"}

    elif ability == "INTIMIDATION":
        # Ironhold extracts tribute via military threat
        tribute = min(INTIMIDATION_TRIBUTE, target.economy)
        target.economy -= tribute
        source.economy += tribute
        target.clamp_resources()
        source.clamp_resources()
        return {"action": "USE_SPECIAL", "success": True, "ability": "INTIMIDATION",
                "special_used": True, "target": target.country_id,
                "effect": f"Intimidation: extracted {tribute:.0f} economy tribute"}

    elif ability == "WATER_CUTOFF":
        # Aqualis cuts water — target food production halved
        food_loss = target.food * WATER_CUTOFF_FOOD_PENALTY
        target.food -= food_loss
        target.internal_stability -= 15
        target.clamp_resources()
        source.clamp_resources()
        return {"action": "USE_SPECIAL", "success": True, "ability": "WATER_CUTOFF",
                "special_used": True, "target": target.country_id,
                "effect": f"Water cutoff: target lost {food_loss:.0f} food, -15 stability"}

    elif ability == "TRADE_MULTIPLIER":
        # Nexus boosts trade — bonus economy for self
        bonus = source.economy * TRADE_MULTIPLIER_BONUS
        source.economy += bonus
        source.clamp_resources()
        return {"action": "USE_SPECIAL", "success": True, "ability": "TRADE_MULTIPLIER",
                "special_used": True, "target": target.country_id,
                "effect": f"Trade multiplier: gained {bonus:.0f} bonus economy"}

    # Unknown ability fallback
    source.clamp_resources()
    return {"action": "USE_SPECIAL", "success": False,
            "reason": f"Unknown ability: {ability}"}


# ==================== DISPATCHER ====================


def resolve_action(action, countries: Dict[str, Country]) -> dict:
    """Main dispatcher — validate and resolve any action."""

    validation = validate_action(action, countries)
    if not validation["valid"]:
        source = countries.get(action.source_country)
        if source:
            result = resolve_wait(source)
            result["original_action"] = action.action_type
            result["fallback"] = True
            result["fallback_reason"] = validation["reason"]
            source.actions_this_episode.append(result)
            return result
        return {"action": "INVALID", "success": False, "reason": validation["reason"]}

    source = countries[action.source_country]
    atype = action.action_type.upper()

    if atype == "WAIT":
        result = resolve_wait(source)

    elif atype == "DEVELOP":
        result = resolve_develop(source, action.resource)

    elif atype == "TRADE":
        target = countries[action.target_country]
        result = resolve_trade(
            source, target,
            action.resource, action.amount,
            action.counter_resource, action.counter_amount,
        )

    elif atype == "PROPOSE_ALLIANCE":
        target = countries[action.target_country]
        result = resolve_propose_alliance(source, target)

    elif atype == "BREAK_ALLIANCE":
        target = countries[action.target_country]
        result = resolve_break_alliance(source, target)

    elif atype == "INVADE":
        target = countries[action.target_country]
        result = resolve_invade(source, target, action.amount or 0.5, countries)

    elif atype == "DEFEND":
        result = resolve_defend(source)

    elif atype == "SANCTION":
        target = countries[action.target_country]
        result = resolve_sanction(source, target)

    elif atype == "THREATEN":
        target = countries[action.target_country]
        result = resolve_threaten(source, target)

    elif atype == "NEGOTIATE_PEACE":
        target = countries[action.target_country]
        result = resolve_negotiate_peace(source, target)

    elif atype == "SPY":
        target = countries[action.target_country]
        result = resolve_spy(source, target)

    elif atype == "COUNTER_INTEL":
        result = resolve_counter_intel(source)

    elif atype == "USE_SPECIAL":
        target = countries[action.target_country]
        result = resolve_use_special(source, target, countries)

    else:
        result = resolve_wait(source)
        result["fallback"] = True

    source.actions_this_episode.append(result)
    return result
