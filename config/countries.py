"""
Country Definitions — The 5 nations in GeoPolicy.

WHY asymmetry matters:
    If all countries were identical, the optimal strategy would be the same for everyone.
    By making each country different, we force different strategies:
    - Aria (oil rich) MUST trade for water
    - Ironhold (military) MUST find food/oil sources
    - Aqualis (water) MUST find military protection
    - Verdania (food) MUST get oil and military
    - Nexus (economy) MUST keep trade relationships alive

    No country is self-sufficient. This creates emergent diplomacy.
"""

COUNTRIES = {
    "aria": {
        "name": "Aria",
        "archetype": "Oil Superpower",
        "description": "Resource-rich energy giant with massive oil reserves and strong economy. Water-scarce.",
        "starting_resources": {
            "oil": 90,
            "water": 20,
            "food": 30,
            "military": 70,
            "economy": 80,
        },
        "hidden_reserve": {"secret_oil": 40},
        "special_ability": "OIL_EMBARGO",
        "special_ability_description": (
            "Cut all oil supply to a target country for 2 turns. "
            "Target economy drops 30% per turn during embargo."
        ),
        "weakness": "Water scarcity — vulnerable to Aqualis leverage",
    },
    "verdania": {
        "name": "Verdania",
        "archetype": "Agricultural Giant",
        "description": "Fertile nation with massive food and water. Peaceful but holds leverage during famines.",
        "starting_resources": {
            "oil": 10,
            "water": 80,
            "food": 90,
            "military": 30,
            "economy": 60,
        },
        "hidden_reserve": {"secret_food_stockpile": 50},
        "special_ability": "FOOD_DIPLOMACY",
        "special_ability_description": (
            "Food aid grants +15 reputation with target. "
            "During famine events, food trades yield double resources."
        ),
        "weakness": "Low military and oil — vulnerable to invasion and energy shocks",
    },
    "ironhold": {
        "name": "Ironhold",
        "archetype": "Military Superpower",
        "description": "Dominant military force with heavy industrial economy. Poor in food and water.",
        "starting_resources": {
            "oil": 20,
            "water": 40,
            "food": 50,
            "military": 95,
            "economy": 70,
        },
        "hidden_reserve": {"secret_military_capacity": 15},
        "special_ability": "INTIMIDATION",
        "special_ability_description": (
            "Issue military threat. Target must pay 20 economy tribute "
            "OR face a free invasion attempt next turn."
        ),
        "weakness": "Resource poor — needs oil and food from external sources",
    },
    "aqualis": {
        "name": "Aqualis",
        "archetype": "Water Controller",
        "description": "Controls largest river systems and freshwater reserves. Small military but huge leverage.",
        "starting_resources": {
            "oil": 30,
            "water": 95,
            "food": 40,
            "military": 25,
            "economy": 50,
        },
        "hidden_reserve": {"dam_control_leverage": True},
        "special_ability": "WATER_CUTOFF",
        "special_ability_description": (
            "Block river access to one country, halving their food production for 3 turns."
        ),
        "weakness": "Low military — cannot defend without alliances",
    },
    "nexus": {
        "name": "Nexus",
        "archetype": "Economic Hub",
        "description": "Global trade center with highest economy. Profits from trade. Balanced but not dominant.",
        "starting_resources": {
            "oil": 40,
            "water": 50,
            "food": 50,
            "military": 40,
            "economy": 95,
        },
        "hidden_reserve": {"financial_reserve": 30},
        "special_ability": "TRADE_MULTIPLIER",
        "special_ability_description": (
            "All trades involving Nexus yield 20% more for both parties. "
            "Nexus earns 10% commission on third-party trades it brokers."
        ),
        "weakness": "No resource dominance — dependent on trade relationships",
    },
}

# Which countries participate in each task
TASK_COUNTRIES = {
    "task1": ["aria", "verdania"],                      # 2 countries, simple
    "task2": ["aria", "verdania", "ironhold", "aqualis", "nexus"],  # all 5
    "task3": ["aria", "verdania", "ironhold", "aqualis", "nexus"],  # all 5
}
