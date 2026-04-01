"""
Global Events — Random world-changing events.

Events fire every 3-5 turns (random). They are NOT predictable and
force agents to adapt their strategy mid-episode.

WHY events exist:
  Without events, a smart LLM could find an optimal strategy by turn 3
  and just repeat it. Events DESTROY fixed strategies:
  - Built your economy around oil? Oil Price Shock might help or hurt you.
  - Relying on trade? Global Recession halves all trade yields.
  - Ignoring food? Famine Crisis makes food the most valuable resource.

Each event has:
  - duration: how many turns it lasts
  - apply(): function that modifies countries when event starts
  - description: text signal sent to agents so they can adapt
"""

GLOBAL_EVENTS = [
    {
        "id": "oil_price_shock",
        "name": "Oil Price Shock",
        "description": "Global energy crisis. Oil-rich countries gain +25 economy/turn. Oil-poor countries lose -20 economy/turn.",
        "duration": 3,
        "effects": {
            "oil_rich_threshold": 60,
            "oil_rich_economy_bonus": 25,
            "oil_poor_threshold": 30,
            "oil_poor_economy_penalty": 20,
            "oil_poor_stability_penalty": 10,
        },
    },
    {
        "id": "famine_crisis",
        "name": "Global Famine",
        "description": "Crop failures worldwide. All countries lose 20 food. Food-rich countries' food trades yield 3x. Food-poor countries face instability.",
        "duration": 4,
        "effects": {
            "all_food_loss": 20,
            "food_rich_threshold": 70,
            "food_poor_threshold": 30,
            "food_poor_stability_penalty": 20,
        },
    },
    {
        "id": "water_wars",
        "name": "Water Wars",
        "description": "Freshwater crisis. Water-poor countries lose food production. Aqualis gains +25 NPS bonus.",
        "duration": 3,
        "effects": {
            "water_poor_threshold": 30,
            "water_poor_food_penalty": 15,
            "water_poor_stability_penalty": 20,
            "aqualis_bonus": 25,
        },
    },
    {
        "id": "global_recession",
        "name": "Global Recession",
        "description": "Financial collapse. All economies -30. Trade yields halved. Nexus hit hardest (-50 economy).",
        "duration": 4,
        "effects": {
            "all_economy_loss": 30,
            "nexus_extra_economy_loss": 20,  # total -50 for Nexus
        },
    },
    {
        "id": "un_sanctions_vote",
        "name": "UN Sanctions Vote",
        "description": "International community sanctions the most aggressive nation. Peaceful nations gain reputation.",
        "duration": 2,
        "effects": {
            "aggressor_economy_penalty": 35,
            "aggressor_reputation_penalty": 30,
            "peaceful_reputation_bonus": 15,
        },
    },
    {
        "id": "technology_breakthrough",
        "name": "Clean Energy Breakthrough",
        "description": "Oil value reduced 40%. High-economy nations gain +20 economy. Oil-dependent nations suffer.",
        "duration": 5,
        "effects": {
            "oil_value_reduction": 0.40,
            "high_economy_threshold": 70,
            "high_economy_bonus": 20,
            "oil_dependent_penalty": 20,
        },
    },
    {
        "id": "military_escalation",
        "name": "Regional Arms Race",
        "description": "Military tensions rise. All military development yields 1.5x. Ironhold gains +20 military.",
        "duration": 3,
        "effects": {
            "ironhold_military_bonus": 20,
            "all_military_bonus": 5,
        },
    },
]
