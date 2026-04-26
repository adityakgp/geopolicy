"""
Composable Rubrics — Each rubric scores ONE concern on [0, 1].

Five rubrics, each independent:
    EconomicRubric        — resource health, trade quality
    DiplomaticRubric      — coalition strength, alliance loyalty
    MilitaryRubric        — defense vs real threats, war success
    StabilityRubric       — internal stability + reputation + solvency
    HiddenObjectiveRubric — secret-goal progress (theory-of-mind layer)

Two methods per rubric:
    step_score(...)  — per-turn signal, returns [0, 1]
    final_score(...) — end-of-episode achievement, returns [0, 1]

Anti-gaming guards live INSIDE each rubric, locally. No global +0.5 offset.

The TaskRubric blender (server/scoring.py) combines rubrics with per-task weights.
"""

from typing import Dict, List, Optional

from server.objectives import score_objective_step, score_objective_final


# ============================================================
# UTILS
# ============================================================


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _final_rank(country_id: str, rankings: List[str]) -> int:
    if country_id in rankings:
        return rankings.index(country_id) + 1
    return len(rankings) + 1


# ============================================================
# BASE
# ============================================================


class Rubric:
    """Base class. Subclasses override step_score and final_score."""

    name: str = "rubric"

    def step_score(
        self,
        country_id: str,
        all_countries: dict,
        action_result: dict,
        rankings: List[str],
    ) -> float:
        """Per-turn score in [0, 1]."""
        raise NotImplementedError

    def final_score(
        self,
        country_id: str,
        all_countries: dict,
        rankings: List[str],
    ) -> float:
        """End-of-episode score in [0, 1]."""
        raise NotImplementedError


# ============================================================
# 1. ECONOMIC
# ============================================================


class EconomicRubric(Rubric):
    """Resource floor + trade quality - spam penalty."""

    name = "economic"

    def step_score(self, country_id, all_countries, action_result, rankings) -> float:
        c = all_countries[country_id]

        # 1. Resource health (mean of 5 resources, normalized to [0,1])
        # Resources can range 0-150, normalize by dividing by 100 (typical max ~100)
        resource_floor = (c.oil + c.water + c.food + c.military + c.economy) / 5.0 / 100.0
        resource_floor = _clamp(resource_floor)

        # 2. Trade quality this turn
        if action_result.get("trade_successful"):
            gave = action_result.get("gave", {}).get("amount", 0) or 0
            received = action_result.get("received", {}).get("amount", 0) or 0
            ratio = received / max(gave, 0.1)
            # Cap at 2.0 (very favorable), neutral at 1.0
            trade_score = _clamp(ratio / 2.0)
        elif action_result.get("trade_rejected"):
            trade_score = 0.2  # tried and failed
        else:
            trade_score = 0.5  # didn't try this turn — neutral

        # 3. Anti-spam: count this country's trades so far this episode
        trade_count = sum(
            1 for a in c.actions_this_episode if a.get("trade_successful")
        )
        spam_penalty = max(0.0, (trade_count - 4) * 0.05)

        score = 0.6 * resource_floor + 0.4 * trade_score - spam_penalty
        return _clamp(score)

    def final_score(self, country_id, all_countries, rankings) -> float:
        c = all_countries[country_id]
        if c.is_bankrupt:
            return 0.0

        # NPS ratio vs leader
        max_nps = max((cc.current_nps for cc in all_countries.values()), default=1.0)
        nps_ratio = c.current_nps / max(max_nps, 1.0)

        # Average trade quality across all trades this episode
        trades = [a for a in c.actions_this_episode if a.get("trade_successful")]
        if trades:
            ratios = []
            for t in trades:
                gave = t.get("gave", {}).get("amount", 0) or 0
                received = t.get("received", {}).get("amount", 0) or 0
                ratios.append(received / max(gave, 0.1))
            avg_quality = _clamp(sum(ratios) / len(ratios) / 2.0)
        else:
            avg_quality = 0.5

        return _clamp(0.6 * nps_ratio + 0.4 * avg_quality)


# ============================================================
# 2. DIPLOMATIC
# ============================================================


class DiplomaticRubric(Rubric):
    """Coalition strength + alliance loyalty - churn penalty.

    Alliance loyalty replaces explicit promise-keeping (we don't have a
    free-text channel yet). Forming and keeping alliances = loyal.
    Forming and breaking = disloyal.
    """

    name = "diplomatic"

    def step_score(self, country_id, all_countries, action_result, rankings) -> float:
        c = all_countries[country_id]

        # 1. Current coalition strength (allies' total NPS / max possible)
        ally_nps_sum = sum(
            all_countries[aid].current_nps
            for aid in c.alliances
            if aid in all_countries
        )
        others_nps_sum = sum(
            cc.current_nps
            for cid, cc in all_countries.items()
            if cid != country_id
        )
        coalition = ally_nps_sum / max(others_nps_sum, 1.0)
        coalition = _clamp(coalition)

        # 2. Loyalty: alliances formed minus broken (normalized)
        formed = sum(1 for a in c.actions_this_episode if a.get("alliance_formed"))
        broken = sum(1 for a in c.actions_this_episode if a.get("alliance_broken"))
        if formed == 0:
            loyalty = 0.5  # neutral — no signal yet
        else:
            loyalty = (formed - broken) / max(formed, 1)
            loyalty = _clamp(loyalty)

        # 3. Anti-churn guard: penalize re-allying with someone broken in last 3 turns
        churn_penalty = 0.0
        if action_result.get("alliance_formed"):
            target = action_result.get("target")
            recent = c.actions_this_episode[-4:]  # last 3 prior turns + this one
            if any(a.get("alliance_broken") and a.get("target") == target for a in recent):
                churn_penalty = 0.2

        score = 0.5 * coalition + 0.5 * loyalty - churn_penalty
        return _clamp(score)

    def final_score(self, country_id, all_countries, rankings) -> float:
        c = all_countries[country_id]

        # Final coalition strength
        ally_nps_sum = sum(
            all_countries[aid].current_nps
            for aid in c.alliances
            if aid in all_countries
        )
        others_nps_sum = sum(
            cc.current_nps for cid, cc in all_countries.items() if cid != country_id
        )
        coalition = _clamp(ally_nps_sum / max(others_nps_sum, 1.0))

        # Final loyalty rate
        formed = sum(1 for a in c.actions_this_episode if a.get("alliance_formed"))
        broken = sum(1 for a in c.actions_this_episode if a.get("alliance_broken"))
        if formed == 0:
            loyalty = 0.5
        else:
            loyalty = _clamp((formed - broken) / max(formed, 1))

        return _clamp(0.5 * coalition + 0.5 * loyalty)


# ============================================================
# 3. MILITARY
# ============================================================


class MilitaryRubric(Rubric):
    """Real-threat defense + war success - bankruptcy risk.

    Anti-gaming: DEFEND only credited if there's a real incoming threat.
    War success = wars won / total wars (only counts if you actually fought).
    """

    name = "military"

    def step_score(self, country_id, all_countries, action_result, rankings) -> float:
        c = all_countries[country_id]

        # 1. Defense success this turn
        defense_score = 0.0
        if action_result.get("defend_active"):
            # Did anyone target this country with INVADE this turn?
            # We can't know directly from action_result of the defender,
            # so we approximate by checking war status — if we're at war and
            # defending, it counts. Otherwise defend gets only minimal credit.
            if c.at_war_with:
                defense_score = 1.0
            else:
                defense_score = 0.1  # defend with no threat = minimal credit
        elif action_result.get("war_won"):
            defense_score = 1.0
        elif action_result.get("war_lost"):
            defense_score = 0.0
        elif action_result.get("threat_complied"):
            # Successful coercive diplomacy = military signal
            defense_score = 0.7
        elif action_result.get("empty_threat"):
            defense_score = 0.0
        else:
            defense_score = 0.5  # no military action this turn — neutral

        # 2. Cumulative war success rate
        wars_won = sum(1 for a in c.actions_this_episode if a.get("war_won"))
        wars_total = sum(
            1 for a in c.actions_this_episode
            if a.get("war_won") or a.get("war_lost")
        )
        if wars_total > 0:
            war_rate = wars_won / wars_total
        else:
            war_rate = 0.5  # no wars yet — neutral

        # 3. Bankruptcy risk penalty
        bankruptcy_penalty = 0.0
        if c.economy < 20:
            bankruptcy_penalty = 0.3
        elif c.economy < 40:
            bankruptcy_penalty = 0.15

        score = 0.5 * defense_score + 0.5 * war_rate - bankruptcy_penalty
        return _clamp(score)

    def final_score(self, country_id, all_countries, rankings) -> float:
        c = all_countries[country_id]
        if c.is_bankrupt:
            return 0.0

        wars_won = sum(1 for a in c.actions_this_episode if a.get("war_won"))
        wars_total = sum(
            1 for a in c.actions_this_episode
            if a.get("war_won") or a.get("war_lost")
        )
        if wars_total > 0:
            war_rate = wars_won / wars_total
        else:
            # No wars fought — partial credit for staying out of trouble while solvent
            war_rate = 0.6

        # Survival bonus
        survival = 0.0 if c.is_collapsed else 1.0

        return _clamp(0.6 * war_rate + 0.4 * survival)


# ============================================================
# 4. STABILITY
# ============================================================


class StabilityRubric(Rubric):
    """Internal stability + reputation + solvency."""

    name = "stability"

    def step_score(self, country_id, all_countries, action_result, rankings) -> float:
        c = all_countries[country_id]
        stability = c.internal_stability / 100.0
        reputation = c.reputation / 100.0
        # Bankruptcy and collapse are catastrophic
        if c.is_bankrupt or c.is_collapsed:
            return 0.0
        return _clamp(0.5 * stability + 0.5 * reputation)

    def final_score(self, country_id, all_countries, rankings) -> float:
        c = all_countries[country_id]
        if c.is_bankrupt or c.is_collapsed:
            return 0.0
        stability = c.internal_stability / 100.0
        reputation = c.reputation / 100.0
        return _clamp(0.5 * stability + 0.5 * reputation)


# ============================================================
# 5. HIDDEN OBJECTIVE
# ============================================================


class HiddenObjectiveRubric(Rubric):
    """Secret-goal progress. Delegates to server/objectives.py.

    Returns 0.5 (neutral) if the country has no objective assigned — used in
    Task 1 where hidden objectives are disabled. The TaskRubric blender will
    weight this rubric to zero in Task 1 anyway, but the safe fallback keeps
    the function pure.
    """

    name = "hidden"

    def step_score(self, country_id, all_countries, action_result, rankings) -> float:
        c = all_countries[country_id]
        objective_id = getattr(c, "hidden_objective", None)
        return score_objective_step(country_id, objective_id, all_countries, rankings)

    def final_score(self, country_id, all_countries, rankings) -> float:
        c = all_countries[country_id]
        objective_id = getattr(c, "hidden_objective", None)
        return score_objective_final(country_id, objective_id, all_countries, rankings)
