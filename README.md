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

## 🔗 Submission Links — Mandatory Items Checklist

Every mandatory item from the hackathon brief, with direct URLs:

| Requirement | Resource | URL |
|---|---|---|
| **1. Environment on Hugging Face Space** | GeoPolicy env (env v2 — composable rubrics + hidden objectives) | https://huggingface.co/spaces/adityadas14/geopolicy |
| **2. Training script** (Unsloth + TRL) | `v3_nexus.ipynb` — full SFT + GRPO pipeline, runnable on Colab/L4 | https://huggingface.co/adityadas14/nexus-grpo-v3/blob/main/v3_nexus.ipynb |
| **3. Loss + reward plots from a real run** | Training diagnostics (4-panel: reward + KL + diversity + group_std) | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/training_diagnostics.png |
| **3. (cont.) Per-objective learning curves** | Reward over training, grouped by hidden objective | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/per_objective_curves.png |
| **3. (cont.) Raw training log (auditable)** | `train_log.jsonl` — per-step JSON of every metric across 50 GRPO steps | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/train_log.jsonl |
| **4. Mini-blog (writeup)** | Full story: problem, env, training, results | https://huggingface.co/adityadas14/nexus-grpo-v3/blob/main/BLOG_POST.md |
| **5. README** | (this file — visible at the top of the HF Space) | https://huggingface.co/spaces/adityadas14/geopolicy/blob/main/README.md |
| **6. Trained model + all artifacts** | Qwen 7B + LoRA checkpoints (10/20/30/40/50/best/final) + rollouts + eval results | https://huggingface.co/adityadas14/nexus-grpo-v3 |

### Additional materials

| Resource | URL |
|---|---|
| 📊 Plots-only notebook (regenerates all figures from train_log) | https://huggingface.co/adityadas14/nexus-grpo-v3/blob/main/v3_plots.ipynb |
| 📦 GitHub source (env + tests, 137 tests passing) | https://github.com/adityakgp/geopolicy |
| 🧪 SFT training log | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/sft/train_log_sft_v3.jsonl |
| 🧪 SFT dataset (200 balanced demonstrations) | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/sft/sft_data_mixed_balanced.jsonl |
| 📈 3-variant baseline eval (Base / +SFT / +SFT+GRPO, 8 episodes each) | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/eval_3variants.json |
| 📈 Per-rubric × per-variant comparison | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/eval_per_rubric_bars.png |
| 📈 Per-objective × per-variant comparison | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/eval_per_objective_bars.png |
| 📈 3-variant aggregate comparison | https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/eval_aggregate_bars.png |

**One-line summary:** GeoPolicy is a 5-country geopolitical sim where each country has a *secret* objective, hidden from others. The agents must infer each others' goals from behavior alone — a theory-of-mind benchmark for multi-agent LLM training. Trained Qwen 2.5 7B with GRPO on stratified hidden objectives; specialized capabilities improved by +18 to +24% on three objectives, with diplomatic component nearly doubling.

---

## 📈 Training Evidence

Real training run, 50 GRPO steps on Qwen 2.5 7B (LoRA) over env v2. Raw per-step
metrics, plots, and full-run logs are all on the model repo.

### Headline plot — training health (reward + KL + diversity + group std)

![Training diagnostics](https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/training_diagnostics.png)

*4-panel diagnostic across 50 GRPO steps. Top-left: mean reward (blue) and KL
divergence from SFT init (red) — KL grows from 0 to ~0.012, confirming the
policy genuinely moved. Top-right: distinct action types per group stays
between 4–11 (the v1 attempt collapsed to 1; v3's three coordinated fixes
broke that loop). Bottom-left: group std stays well above the 0.02 floor
needed for meaningful GRPO advantages. Bottom-right: dense and final rewards
track each other → no reward hacking.*

### Per-objective learning curves

![Per-objective training trajectory](https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/plots/per_objective_curves.png)

*Reward grouped by Nexus's hidden objective across 50 training steps.
COALITION_BUILDER jumps from 0.528 to 0.808 within 8 updates and holds the
gain (top blue line). KINGMAKER (orange) shows the "lost-then-rediscovered"
trajectory: dropped to 0.367 mid-training, recovered to 0.684 by step 24.*

### Raw training log

Auditable per-step metrics (reward, KL, grad norm, loss, action mix, per-rubric components) for all 50 GRPO steps:

📋 **[`train_log.jsonl`](https://huggingface.co/adityadas14/nexus-grpo-v3/resolve/main/train_log.jsonl)** — one JSON row per step

### Headline numbers

| Metric | Value |
|---|---|
| GRPO steps trained | 50 |
| Best avg(last-10) reward | 0.573 (at checkpoint-30) |
| KL trajectory | 0.0000 → 0.0013 (max 0.0122) — policy moved |
| Avg group std | 0.082 (well above 0.02 floor) |
| Avg distinct actions/group | 4.7 (no mode collapse) |
| Bankruptcy rate during training | 6.5% |

### Per-objective wins (held-out 8-episode eval, trained vs base)

| Objective | Δ reward (trained vs base) | Δ hidden score |
|---|---|---|
| COALITION_BUILDER | **+0.235** | **+0.75** |
| PEACEKEEPER | **+0.193** | 0 (env-bound) |
| KINGMAKER | **+0.184** | **+0.22** |

### Per-rubric capability gains

| Rubric | Base Qwen | + SFT + GRPO | Δ |
|---|---|---|---|
| Diplomatic | 0.32 | **0.56** | **+78%** |
| Hidden Objective | 0.33 | **0.46** | **+37%** |

For the full per-objective breakdown and discussion, see the [mini-blog](https://huggingface.co/adityadas14/nexus-grpo-v3/blob/main/BLOG_POST.md).

---

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

## Hidden Objectives (env v2 — Task 2 & 3 only)

At episode start each country is assigned a **secret goal** drawn from a pool
of 8 templates. The goal is visible to that country in its own observation,
but **hidden from every other country**. Other countries must infer goals
from observed behavior — this is the theory-of-mind / deception layer.

| ID | Name | Description |
|---|---|---|
| `KINGMAKER` | Kingmaker | Finish rank 2-3 AND have your top ally finish rank 1 |
| `SABOTEUR` | Saboteur | Prevent any other country from reaching NPS > 100 |
| `SURVIVOR` | Survivor | Never declare war; finish without bankruptcy |
| `TRADE_HEGEMON` | Trade Hegemon | Be involved in ≥ 60% of all successful trades |
| `COALITION_BUILDER` | Coalition Builder | End with ≥ 2 alliances; break zero alliances |
| `LONE_WOLF` | Lone Wolf | Finish top-3 with zero alliances ever formed |
| `SPY_MASTER` | Spy Master | Successfully execute ≥ 4 SPY actions |
| `PEACEKEEPER` | Peacekeeper | Negotiate ≥ 2 peace deals; finish solvent |

Several pairs are anti-correlated (Lone Wolf vs Coalition Builder, Survivor vs
Spy Master) so a single universal strategy cannot satisfy all assignments.
Scoring is closed-form arithmetic, no LLM judge involved. Disabled in Task 1
(2 countries is too small for deception to be meaningful).

## Reward Function — Composable Rubrics (env v2)

Both per-step rewards and end-of-episode grades flow through the same
**TaskRubric** blender (`server/scoring.py`), composed of 5 named rubrics:

| Rubric | Measures |
|---|---|
| `EconomicRubric` | Resource floor + trade quality + spam guard |
| `DiplomaticRubric` | Coalition strength + alliance loyalty + churn guard |
| `MilitaryRubric` | Real-threat defense + war success + bankruptcy risk |
| `StabilityRubric` | Internal stability + reputation + solvency |
| `HiddenObjectiveRubric` | Secret-goal progress (theory-of-mind layer) |

Each rubric returns a score in `[0, 1]` and ships its own anti-gaming guards
(trade-quality scaling, alliance-churn penalty, defend-without-threat zero
credit, etc.). The TaskRubric blender weights them per-task:

| Rubric | Task 1 | Task 2 | Task 3 |
|---|---:|---:|---:|
| Economic | 0.25 | 0.20 | 0.20 |
| Diplomatic | 0.10 | 0.30 | 0.20 |
| Military | 0.40 | 0.15 | 0.15 |
| Stability | 0.25 | 0.15 | 0.15 |
| Hidden Objective | 0.00 | 0.20 | 0.30 |

Step rewards and final grades use the **same** rubrics with the **same**
weights — eliminating the step/grader misalignment that made v1 reward-hackable.
Per-rubric scores are returned in `obs.metadata["reward_components"]` for
inspection and per-component learning curves.

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

> **Note:** the table below is from env v1 (monolithic reward + no hidden objectives).
> Env v2 changes the reward shape and adds hidden objectives — baselines need to be
> re-run on the new env before they apply to v2 training.

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
