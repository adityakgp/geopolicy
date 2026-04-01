"""
Game Constants — All the numbers that control the simulation.

Keeping these in one place makes it easy to tune the game balance.
If rewards feel too high or wars are too strong, we change numbers HERE,
not scattered across 10 files.
"""

# --- Task Configuration ---
TASK_CONFIG = {
    "task1": {
        "max_turns": 8,
        "global_events_enabled": False,
        "hidden_info_enabled": False,       # full transparency for easy task
        "special_abilities_enabled": False,
        "description": "Bilateral Survival: 2 countries, 8 turns, finish with higher NPS",
    },
    "task2": {
        "max_turns": 10,
        "global_events_enabled": True,
        "hidden_info_enabled": True,        # partial visibility (tiers)
        "special_abilities_enabled": True,
        "description": "Coalition Wars: 5 countries, 10 turns, alliances + events",
    },
    "task3": {
        "max_turns": 12,
        "global_events_enabled": True,
        "hidden_info_enabled": True,
        "special_abilities_enabled": True,
        "description": "Full Simulation: 5 countries, 12 turns, everything active",
    },
}

# --- NPS (National Power Score) Weights ---
# These control how much each resource matters for the final ranking
NPS_WEIGHTS = {
    "oil": 0.25,
    "water": 0.20,
    "food": 0.20,
    "military": 0.20,
    "economy": 0.15,
}

NPS_ALLIANCE_BONUS = 10        # +10 NPS per alliance
NPS_TRADE_BONUS = 5            # +5 NPS per active trade agreement
NPS_WAR_PENALTY = 15           # -15 NPS per active war
STABILITY_WEIGHT = 0.3         # how much internal stability affects NPS

# --- Starting Values ---
STARTING_STABILITY = 70.0      # all countries start with 70/100 stability
STARTING_REPUTATION = 50.0     # all countries start with 50/100 reputation

# --- Resource Tiers (for public info) ---
# When country A looks at country B, they see tiers, not exact numbers
# This is the "fog of war" — you need SPY to see exact values
RESOURCE_TIERS = {
    "very_low": (0, 20),
    "low": (20, 40),
    "medium": (40, 60),
    "high": (60, 80),
    "very_high": (80, 150),
}

# --- Special Ability Cooldown ---
SPECIAL_ABILITY_COOLDOWN = 5   # turns until special ability can be used again

# --- Action Costs (economy spent to perform an action) ---
ACTION_COSTS = {
    "WAIT": 0,
    "DEVELOP": 20,
    "TRADE": 5,
    "SANCTION": 10,
    "INVADE": 30,
    "DEFEND": 0,
    "PROPOSE_ALLIANCE": 0,
    "BREAK_ALLIANCE": 0,
    "NEGOTIATE_PEACE": 10,
    "THREATEN": 5,
    "SPY": 25,
    "COUNTER_INTEL": 15,
    "USE_SPECIAL": 20,
}

# --- WAIT action ---
WAIT_RECOVERY = 3              # all resources +3 per WAIT
WAIT_STABILITY_BONUS = 5       # stability +5 per WAIT
WAIT_REPUTATION_BONUS = 2      # reputation +2 per WAIT

# --- DEVELOP action ---
DEVELOP_GAIN = 15              # resource gains +15 on next step
DEVELOP_DIMINISHING = 8        # 2nd consecutive develop on same resource gives +8

# --- TRADE action ---
TRADE_REPUTATION_GAIN = 3      # both parties gain +3 reputation on successful trade
TRADE_REJECT_PENALTY = 5       # source loses 5 economy if trade rejected
TRADE_MIN_REPUTATION = 15      # target must have at least this reputation with source to accept
TRADE_MAX_AMOUNT = 50          # can't trade more than 50 of any resource at once

# --- Natural Recovery (happens every turn for all countries) ---
NATURAL_RECOVERY = 2           # all resources +2 per turn (baseline growth)

# --- ALLIANCE action ---
ALLIANCE_DEFENSE_BONUS = 0.3   # each ally gives +30% defense power when invaded
ALLIANCE_REPUTATION_GAIN = 5   # both parties gain +5 reputation on alliance formed
ALLIANCE_BREAK_REP_PENALTY = 15  # reputation cost of breaking an alliance
ALLIANCE_MIN_REPUTATION = 30   # target must have >=30 reputation to accept alliance

# --- INVADE action ---
INVADE_WIN_RESOURCE_LOOT = 0.30   # winner takes 30% of loser's oil, food, economy
INVADE_WIN_TARGET_MIL_LOSS = 30   # target loses 30 military on invasion loss
INVADE_WIN_TARGET_ECO_LOSS = 25   # target loses 25 economy on invasion loss
INVADE_WIN_SOURCE_MIL_LOSS = 15   # attacker loses 15 military even on win (war cost)
INVADE_WIN_SOURCE_REP_LOSS = 25   # attacker reputation penalty (aggressor)
INVADE_LOSE_SOURCE_MIL_LOSS = 20  # attacker loses 20 military on failed invasion
INVADE_LOSE_SOURCE_ECO_LOSS = 15  # attacker loses 15 economy on failed invasion
INVADE_LOSE_SOURCE_REP_LOSS = 15  # attacker reputation loss on failure
INVADE_LOSE_TARGET_REP_GAIN = 10  # defender gains reputation for surviving
WAR_DURATION = 2                  # war status lasts 2 turns after invasion

# --- DEFEND action ---
DEFEND_BONUS = 0.25               # +25% defense power this turn

# --- SANCTION action ---
SANCTION_TARGET_ECO_LOSS = 15     # target loses 15 economy per turn under sanctions
SANCTION_SOURCE_ECO_LOSS = 5      # source loses 5 economy per turn (cost of sanctions)
SANCTION_REPUTATION_LOSS = 10     # global reputation penalty for sanctioning

# --- THREATEN action ---
THREATEN_COMPLY_TRIBUTE = 15      # tribute paid if target complies
THREATEN_EMPTY_REP_LOSS = 20      # reputation loss if source doesn't follow through

# --- NEGOTIATE_PEACE action ---
PEACE_STABILITY_GAIN = 5          # both parties gain stability on peace
PEACE_REPUTATION_GAIN = 8         # both parties gain reputation (peacemakers)

# --- SPY action ---
SPY_INTEL_DURATION = 3            # revealed info lasts 3 turns
SPY_CATCH_CHANCE = 0.10           # 10% chance spy is caught
SPY_CAUGHT_REP_LOSS = 20         # reputation loss if caught
SPY_CAUGHT_ALERT = True          # target is notified who tried

# --- COUNTER_INTEL action ---
COUNTER_INTEL_DURATION = 3        # counter-intel active for 3 turns

# --- SPECIAL ABILITIES ---
SPECIAL_COOLDOWN = 5              # turns until ability can be used again
OIL_EMBARGO_DURATION = 2          # embargo lasts 2 turns
OIL_EMBARGO_ECO_PENALTY = 0.30    # target economy -30% per turn
FOOD_DIPLOMACY_REP_GAIN = 15     # reputation gain with target
WATER_CUTOFF_DURATION = 3         # water cutoff lasts 3 turns
WATER_CUTOFF_FOOD_PENALTY = 0.50  # target food production halved
INTIMIDATION_TRIBUTE = 20         # tribute extracted (economy)
TRADE_MULTIPLIER_BONUS = 0.20     # 20% more resources on trades

# --- GLOBAL EVENTS ---
EVENT_MIN_INTERVAL = 3            # minimum turns between events
EVENT_MAX_INTERVAL = 5            # maximum turns between events
