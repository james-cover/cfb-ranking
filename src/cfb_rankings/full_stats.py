"""Pregame, opponent-conditioned profiles for all thirteen requested stat groups.

Ratings use a diagonal Gaussian-filter approximation, not an exact joint Bayesian
posterior. Updates are delayed until the end of a week. No current-week outcome
can change another current-week expectation. Fixed scales are modeling priors,
not estimates fitted to the complete dataset.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

# metric: league-center prior, game noise SD, team effect prior SD
METRICS = {
    "points": (27., 14., 8.), "margin": (0., 20., 10.),
    "passing": (215., 95., 55.), "rushing": (160., 70., 40.),
    "yards": (375., 115., 70.), "yards_per_pass": (7., 2.5, 1.3),
    "yards_per_rush": (4.2, 1.8, 1.), "turnover_margin": (0., 2., .7),
    "possession": (30., 5., 2.), "third_down": (.38, .15, .07),
    "first_downs": (20., 6., 3.), "penalty_yards": (50., 30., 12.),
    "ppa": (.1, .3, .15), "success_rate": (.4, .1, .05),
}
KINDS = ("raw_for", "raw_against", "off_rating", "def_rating", "off_residual", "def_residual")
FULL_TEAM_FEATURES = [f"full_{metric}_{kind}" for metric in METRICS for kind in KINDS]
FULL_FEATURES = [f"{name}_diff" for name in FULL_TEAM_FEATURES]
ADJUSTED_FEATURES = [name for name in FULL_FEATURES if "_raw_" not in name]
RAW_FEATURES = [name for name in FULL_FEATURES if "_raw_" in name]
NONREDUNDANT_FEATURES = [name for name in ADJUSTED_FEATURES
                         if not name.startswith(("full_margin_", "full_yards_"))]


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def ratio(value):
    if value is None:
        return None
    parts = re.split(r"[-/]", str(value))
    if len(parts) == 2:
        made, attempts = map(number, parts)
        return made / attempts if made is not None and attempts and 0 <= made <= attempts else None
    result = number(str(value).rstrip("%"))
    if result is not None and ("%" in str(value) or result > 1):
        result /= 100
    return result if result is not None and 0 <= result <= 1 else None


def observations(stats, advanced, points, opponent_points, opponent_stats):
    """Parse genuine values only; no zero-filling or fabricated opponent TOP."""
    values = {"points": number(points)}
    values["margin"] = points - opponent_points
    aliases = {
        "passing": "netPassingYards", "rushing": "rushingYards", "yards": "totalYards",
        "yards_per_pass": "yardsPerPass", "yards_per_rush": "yardsPerRushAttempt",
        "first_downs": "firstDowns",
    }
    for metric, key in aliases.items():
        values[metric] = number(stats.get(key))
    if values["yards"] is None and all(values[k] is not None for k in ("passing", "rushing")):
        values["yards"] = values["passing"] + values["rushing"]
    for metric, numerator, attempt_key in (
        ("yards_per_rush", "rushing", "rushingAttempts"),
        ("yards_per_pass", "passing", "passingAttempts"),
    ):
        attempts = number(stats.get(attempt_key))
        if metric == "yards_per_pass" and attempts is None:
            parts = str(stats.get("completionAttempts", "")).split("-")
            attempts = number(parts[-1]) if len(parts) == 2 else None
        if values[metric] is None and attempts and attempts > 0 and values[numerator] is not None:
            values[metric] = values[numerator] / attempts
    own_to = number(stats.get("turnovers", stats.get("turnoversLost")))
    opp_to = number(opponent_stats.get("turnovers", opponent_stats.get("turnoversLost")))
    values["turnover_margin"] = opp_to - own_to if own_to is not None and opp_to is not None else None
    values["third_down"] = ratio(stats.get("thirdDownEff"))
    penalty = str(stats.get("totalPenaltiesYards", "")).split("-")
    values["penalty_yards"] = number(penalty[-1]) if len(penalty) == 2 else number(stats.get("penaltyYards"))
    possession = str(stats.get("possessionTime", "")).split(":")
    values["possession"] = None
    if len(possession) == 2:
        minutes, seconds = map(number, possession)
        if minutes is not None and seconds is not None and minutes >= 0 and 0 <= seconds < 60:
            values["possession"] = minutes + seconds / 60
    elif len(possession) == 1:
        values["possession"] = number(possession[0])
    values["ppa"] = number(advanced.get("offense_ppa"))
    values["success_rate"] = ratio(advanced.get("offense_success_rate"))
    return values


class FullStatProfiles:
    def __init__(self, teams, previous=None):
        self.teams = sorted(teams)
        self.previous = previous or {}
        self.off, self.defense, self.ov, self.dv = {}, {}, {}, {}
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.pending = []
        self.audit = []
        for metric, (_, _, prior_sd) in METRICS.items():
            for team in self.teams:
                key = (team, metric)
                prior = self.previous.get(key, {})
                self.off[key] = .65 * prior.get("off_rating", 0.)
                self.defense[key] = .65 * prior.get("def_rating", 0.)
                self.ov[key] = self.dv[key] = prior_sd ** 2

    def predict(self, team, opponent, metric):
        return METRICS[metric][0] + self.off[team, metric] - self.defense[opponent, metric]

    def record(self, team, opponent, values, **context):
        for metric, actual in values.items():
            if actual is None or not math.isfinite(actual):
                continue
            expected = self.predict(team, opponent, metric)
            self.pending.append((team, opponent, metric, actual, expected))
            self.audit.append({**context, "team": team, "opponent": opponent, "stat": metric,
                               "actual": actual, "pregame_expected": expected,
                               "residual": actual - expected})

    def finish_week(self):
        # Predictions/residuals above were frozen before any of this week's updates.
        for team, opponent, metric, actual, expected in self.pending:
            key, other = (team, metric), (opponent, metric)
            noise = METRICS[metric][1] ** 2
            # Sequential filter updates use current posterior, but never rewrite
            # the archived pregame expectation or residual.
            innovation = actual - self.predict(team, opponent, metric)
            variance = noise + self.ov[key] + self.dv[other]
            og, dg = self.ov[key] / variance, self.dv[other] / variance
            self.off[key] += og * innovation
            self.defense[other] -= dg * innovation
            self.ov[key] *= 1 - og
            self.dv[other] *= 1 - dg
            for who, kind, value in ((team, "raw_for", actual), (opponent, "raw_against", actual),
                                     (team, "off_residual", actual - expected),
                                     (opponent, "def_residual", expected - actual)):
                stat_key = (who, metric, kind)
                self.sums[stat_key] += value
                self.counts[stat_key] += 1
        self.pending.clear()

    def snapshot(self, team):
        result = {}
        for metric, (center, _, _) in METRICS.items():
            prior = self.previous.get((team, metric), {})
            for kind in KINDS:
                if kind == "off_rating":
                    value = self.off[team, metric]
                elif kind == "def_rating":
                    value = self.defense[team, metric]
                else:
                    key = (team, metric, kind)
                    n = self.counts[key]
                    baseline = center if kind.startswith("raw_") else 0.
                    prior_value = baseline + .65 * (prior.get(kind, baseline) - baseline)
                    value = (self.sums[key] + 4 * prior_value) / (n + 4)
                    result[f"full_{metric}_{kind}_games"] = n
                    result[f"full_{metric}_{kind}_observed"] = self.sums[key] / n if n else float("nan")
                result[f"full_{metric}_{kind}"] = value
        return result

    def profiles(self, eligible):
        snapshots = {team: self.snapshot(team) for team in eligible if team in self.teams}
        return {(team, metric): {kind: snapshots[team][f"full_{metric}_{kind}"] for kind in KINDS}
                for team in eligible for metric in METRICS if team in self.teams}
