---
title: GeoPolicy — Geopolitical Resource Competition
emoji: "\U0001F30D"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
---

# GeoPolicy — Geopolitical Resource Competition

A multi-agent OpenEnv environment where 5 country-agents compete for global dominance through trade, diplomacy, military action, espionage, and alliance management.

## Environment Overview

Each country has unique resources (oil, water, food, military, economy) and a special ability. Countries must trade, ally, threaten, and spy to maximize their National Power Score (NPS) across multiple turns while adapting to random global events.

### Countries

| Country | Archetype | Strength | Weakness | Special Ability |
|---------|-----------|----------|----------|-----------------|
| Aria | Oil Superpower | Oil (90), Economy (80) | Water (20) | OIL_EMBARGO |
| Verdania | Agricultural Giant | Food (90), Water (80) | Oil (10), Military (30) | FOOD_DIPLOMACY |
| Ironhold | Military Superpower | Military (95) | Oil (20), Water (40) | INTIMIDATION |
| Aqualis | Water Controller | Water (95) | Military (25), Economy (50) | WATER_CUTOFF |
| Nexus | Economic Hub | Economy (95) | No dominant resource | TRADE_MULTIPLIER |

## Tasks

| Task | Difficulty | Countries | Turns | Features |
|------|-----------|-----------|-------|----------|
| task1 — Bilateral Survival | Easy | 2 | 8 | Basic resources, full transparency |
| task2 — Coalition Wars | Medium | 5 | 10 | Alliances, global events, hidden info |
| task3 — Full Simulation | Hard | 5 | 12 | All features: espionage, special abilities, events |

## Action Space (13 actions)

| Action | Cost | Description |
|--------|------|-------------|
| WAIT | 0 | Passive recovery (+3 all resources) |
| DEVELOP | 20 | Grow one resource (+15) |
| TRADE | 5 | Exchange resources with another country |
| PROPOSE_ALLIANCE | 0 | Form mutual defense pact (+30% defense) |
| BREAK_ALLIANCE | 0 | Exit an alliance (-15 reputation) |
| INVADE | 30 | Military attack to seize resources |
| DEFEND | 0 | Boost defense this turn (+25%) |
| SANCTION | 10 | Economic pressure on a rival |
| THREATEN | 5 | Demand tribute (works if stronger) |
| NEGOTIATE_PEACE | 10 | End an ongoing war |
| SPY | 25 | Reveal rival's exact resources |
| COUNTER_INTEL | 15 | Block spy attempts for 3 turns |
| USE_SPECIAL | 20 | Activate unique country ability |

## Observation Space

Each country sees its own full state (resources, alliances, wars) and partial information about rivals (tiers like "high"/"low" instead of exact numbers). Exact rival data requires SPY actions.

## Reward Function

Per-turn reward combining:
- NPS delta (did your score improve?)
- Ranking bonus (where are you vs others?)
- Action quality (trade succeeded? war won? alliance formed?)

All rewards in [0.0, 1.0] range with partial progress signals.

## Global Events (Task 2 & 3)

Random world-changing events fire every 3-5 turns: Oil Price Shock, Global Famine, Water Wars, Global Recession, UN Sanctions Vote, Clean Energy Breakthrough, Regional Arms Race.

## Setup

```bash
pip install -r requirements.txt
python server/app.py
```

## API

- `POST /reset` — Start new episode (`{"task_id": "task1"}`)
- `POST /step` — Submit action (`{"action": {"action_type": "WAIT", "source_country": "aria"}}`)
- `GET /state` — Get episode metadata

## Running Inference

```bash
export API_BASE_URL="your-llm-endpoint"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"
python inference.py
```

## Baseline Scores

Scores from running `inference.py` with `gpt-4o-mini` (April 2026):

| Task | Difficulty | Country Scores | Average |
|------|-----------|---------------|---------|
| task1 — Bilateral Survival | Easy | aria: 0.800, verdania: 0.400 | 0.600 |
| task2 — Coalition Wars | Medium | verdania: 0.700, aqualis: 0.600, aria: 0.600, nexus: 0.300, ironhold: 0.200 | 0.480 |
| task3 — Full Simulation | Hard | verdania: 0.890, aria: 0.775, nexus: 0.491, ironhold: 0.421, aqualis: 0.306 | 0.577 |

Key observations:
- Diplomatic countries (Verdania, Aqualis) outperform aggressive ones (Ironhold)
- Alliance formation is the strongest strategy across tasks
- Ironhold's invasion-first approach consistently backfires due to economy drain
- Scores decrease with task complexity as expected (easy > hard for top performers)
