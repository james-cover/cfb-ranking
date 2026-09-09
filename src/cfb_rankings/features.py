from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

AP_PATTERN = re.compile(r"associated press|ap top|\bap\b", re.IGNORECASE)
EARLY_SEASON_PRIOR_GAMES = 2.0
MODEL_FEATURE_BASELINES = {
    "win_pct": 0.5,
    "points_per_game": 27.0,
    "points_allowed_per_game": 27.0,
    "avg_margin": 0.0,
    # sos_elo intentionally excluded — SOS should always reflect actual
    # opponents faced, even after a single game. Regressing it toward 1500
    # washes out early-season schedule-strength signal.
    "yards_per_game": 375.0,
    "yards_allowed_per_game": 375.0,
    "turnover_margin_per_game": 0.0,
    "offense_ppa": 0.0,
    "defense_ppa": 0.0,
    "offense_success_rate": 0.4,
    "defense_success_rate": 0.4,
    "recent_margin_3": 0.0,
    # Quality-adjusted stats share the same baselines as their raw counterparts.
    # The opponent weighting already discounts weak-schedule results naturally,
    # so the prior here is just a safety net for zero-game edge cases.
    "quality_adj_margin": 0.0,
    "quality_adj_ppg": 27.0,
    "quality_adj_ppg_allowed": 27.0,
}


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).replace(",", "").replace("%", "").strip()
    if "-" in text and not text.startswith("-"):
        # Conversion strings such as 5-12 are not scalar measurements.
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_stat(stats: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    normalized = {re.sub(r"[^a-z0-9]", "", key.lower()): value for key, value in stats.items()}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in normalized:
            return _safe_float(normalized[key])
    return None


@dataclass
class TeamState:
    team: str
    elo: float = 1500.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    margin_sum: float = 0.0
    opponent_elo_sum: float = 0.0
    road_wins: int = 0
    ranked_wins: int = 0
    quality_wins: int = 0
    bad_losses: int = 0
    yards_for_sum: float = 0.0
    yards_against_sum: float = 0.0
    turnovers_forced_sum: float = 0.0
    turnovers_lost_sum: float = 0.0
    stat_games: int = 0
    offense_ppa_sum: float = 0.0
    defense_ppa_sum: float = 0.0
    offense_success_rate_sum: float = 0.0
    defense_success_rate_sum: float = 0.0
    advanced_games: int = 0
    season_games: int = 0  # games played this season only — resets each year
    recent_margins: list[float] = field(default_factory=list)
    results: list[tuple[str, int]] = field(default_factory=list)
    # Opponent-quality-weighted accumulators.  Each game's contribution is
    # scaled by (opponent_elo / 1500) so that a 49-point blowout of a 1327-Elo
    # cupcake inflates these stats far less than 49 points against a 1600-Elo team.
    quality_margin_sum: float = 0.0
    quality_score_sum: float = 0.0
    quality_allowed_sum: float = 0.0
    quality_weight_sum: float = 0.0

    def snapshot(self) -> dict[str, float | int | str]:
        games = max(self.games, 1)
        stat_games = max(self.stat_games, 1)
        advanced_games = max(self.advanced_games, 1)
        return {
            "team": self.team,
            "elo": self.elo,
            "games": self.games,
            "season_games": self.season_games,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_pct": (self.wins + 0.5 * self.ties) / games if self.games else 0.5,
            "points_per_game": self.points_for / games if self.games else 0.0,
            "points_allowed_per_game": self.points_against / games if self.games else 0.0,
            "avg_margin": self.margin_sum / games if self.games else 0.0,
            "sos_elo": self.opponent_elo_sum / games if self.games else 1500.0,
            "road_wins": self.road_wins,
            "ranked_wins": self.ranked_wins,
            "quality_wins": self.quality_wins,
            "bad_losses": self.bad_losses,
            "yards_per_game": self.yards_for_sum / stat_games if self.stat_games else 0.0,
            "yards_allowed_per_game": (
                self.yards_against_sum / stat_games if self.stat_games else 0.0
            ),
            "turnover_margin_per_game": (
                (self.turnovers_forced_sum - self.turnovers_lost_sum) / stat_games
                if self.stat_games
                else 0.0
            ),
            "offense_ppa": self.offense_ppa_sum / advanced_games if self.advanced_games else 0.0,
            "defense_ppa": self.defense_ppa_sum / advanced_games if self.advanced_games else 0.0,
            "offense_success_rate": (
                self.offense_success_rate_sum / advanced_games if self.advanced_games else 0.0
            ),
            "defense_success_rate": (
                self.defense_success_rate_sum / advanced_games if self.advanced_games else 0.0
            ),
            "recent_margin_3": (
                float(np.mean(self.recent_margins[-3:])) if self.recent_margins else 0.0
            ),
            # Quality-adjusted stats: weighted by opponent Elo / 1500.
            # After 1 game vs a 1327-Elo team, these are heavily discounted vs
            # raw stats, giving the model an explicit signal about schedule strength.
            "quality_adj_margin": (
                self.quality_margin_sum / self.quality_weight_sum
                if self.quality_weight_sum > 0 else 0.0
            ),
            "quality_adj_ppg": (
                self.quality_score_sum / self.quality_weight_sum
                if self.quality_weight_sum > 0 else 0.0
            ),
            "quality_adj_ppg_allowed": (
                self.quality_allowed_sum / self.quality_weight_sum
                if self.quality_weight_sum > 0 else 0.0
            ),
        }


TEAM_NUMERIC_FEATURES = [
    "elo",
    "games",
    "season_games",
    "wins",
    "losses",
    "win_pct",
    "points_per_game",
    "points_allowed_per_game",
    "avg_margin",
    "sos_elo",
    "road_wins",
    "ranked_wins",
    "quality_wins",
    "bad_losses",
    "yards_per_game",
    "yards_allowed_per_game",
    "turnover_margin_per_game",
    "offense_ppa",
    "defense_ppa",
    "offense_success_rate",
    "defense_success_rate",
    "recent_margin_3",
    "quality_adj_margin",
    "quality_adj_ppg",
    "quality_adj_ppg_allowed",
]

MATCHUP_FEATURES = [
    "elo_diff",
    "games_diff",
    "season_games_diff",
    "wins_diff",
    "losses_diff",
    "win_pct_diff",
    "points_per_game_diff",
    "points_allowed_per_game_diff",
    "avg_margin_diff",
    "sos_elo_diff",
    "road_wins_diff",
    "ranked_wins_diff",
    "quality_wins_diff",
    "bad_losses_diff",
    "yards_per_game_diff",
    "yards_allowed_per_game_diff",
    "turnover_margin_per_game_diff",
    "offense_ppa_diff",
    "defense_ppa_diff",
    "offense_success_rate_diff",
    "defense_success_rate_diff",
    "recent_margin_3_diff",
    "quality_adj_margin_diff",
    "quality_adj_ppg_diff",
    "quality_adj_ppg_allowed_diff",
    "neutral_site",
    "home_field",
    "season_progress",
]

AP_FEATURES = [
    "previous_ap_rank_filled",
    "previous_ap_points",
    "previous_first_place_votes",
    "previously_ranked",
    "elo",
    "games",
    "wins",
    "losses",
    "win_pct",
    "points_per_game",
    "points_allowed_per_game",
    "avg_margin",
    "sos_elo",
    "road_wins",
    "ranked_wins",
    "quality_wins",
    "bad_losses",
    "yards_per_game",
    "yards_allowed_per_game",
    "turnover_margin_per_game",
    "offense_ppa",
    "defense_ppa",
    "offense_success_rate",
    "defense_success_rate",
    "recent_margin_3",
    "last_week_won",
    "last_week_margin",
    "last_week_opponent_elo",
    "last_week_road",
    "last_week_ranked_opponent",
]


def ap_poll_rows(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return rankings.copy()
    names = rankings["poll"].fillna("").astype(str)
    return rankings[names.map(lambda value: bool(AP_PATTERN.search(value)))].copy()


def _pregame_ap_rank_lookup(rankings: pd.DataFrame) -> dict[tuple[int, int, str], int]:
    ap = ap_poll_rows(rankings)
    if ap.empty:
        return {}
    regular = ap[ap["season_type"].astype(str).str.lower().eq("regular")].copy()
    return {
        (int(row.season), int(row.poll_week), str(row.team)): int(row.rank)
        for row in regular.itertuples()
    }


def _team_stats_lookup(team_game_stats: pd.DataFrame) -> dict[tuple[Any, str], dict[str, Any]]:
    lookup: dict[tuple[Any, str], dict[str, Any]] = {}
    if team_game_stats.empty or "stats_json" not in team_game_stats:
        return lookup
    for row in team_game_stats.itertuples():
        try:
            values = json.loads(row.stats_json) if isinstance(row.stats_json, str) else {}
        except json.JSONDecodeError:
            values = {}
        lookup[(row.game_id, str(row.team))] = values
    return lookup


def _advanced_stats_lookup(
    advanced_game_stats: pd.DataFrame,
) -> dict[tuple[Any, str], dict[str, float]]:
    lookup: dict[tuple[Any, str], dict[str, float]] = {}
    if advanced_game_stats.empty or "game_id" not in advanced_game_stats or "team" not in advanced_game_stats:
        return lookup
    canonical_columns = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in advanced_game_stats.columns
    }
    aliases = {
        "offense_ppa": ("offenseppa",),
        "defense_ppa": ("defenseppa",),
        "offense_success_rate": ("offensesuccessrate",),
        "defense_success_rate": ("defensesuccessrate",),
    }
    for row in advanced_game_stats.to_dict(orient="records"):
        values: dict[str, float] = {}
        for output_name, candidates in aliases.items():
            for candidate in candidates:
                source = canonical_columns.get(candidate)
                value = _safe_float(row.get(source)) if source else None
                if value is not None:
                    values[output_name] = value
                    break
        lookup[(row["game_id"], str(row["team"]))] = values
    return lookup


def _initial_elo(
    team: str,
    prior_elos: dict[str, float],
    reversion: float,
    baseline: float = 1500.0,
) -> float:
    prior = prior_elos.get(team, baseline)
    return baseline + reversion * (prior - baseline)


def _elo_expected(home_elo: float, away_elo: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + home_advantage - away_elo) / 400.0))


def _update_elo(
    home: TeamState,
    away: TeamState,
    home_margin: float,
    neutral: bool,
    k_factor: float,
) -> None:
    actual = 1.0 if home_margin > 0 else 0.0 if home_margin < 0 else 0.5
    expected = _elo_expected(home.elo, away.elo, 0.0 if neutral else 55.0)
    margin_multiplier = math.log(abs(home_margin) + 1.0) * (
        2.2 / ((abs(home.elo - away.elo) * 0.001) + 2.2)
    )
    change = k_factor * max(1.0, margin_multiplier) * (actual - expected)
    home.elo += change
    away.elo -= change


def _prefix_snapshot(snapshot: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in snapshot.items() if key != "team"}


def model_adjusted_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Shrink volatile rate stats toward neutral values early in a season.

    Raw records and displayed statistics stay untouched. Only model inputs are adjusted,
    using a four-game empirical prior so one September blowout is not treated as a stable
    season-long average.
    """
    adjusted = dict(snapshot)
    games = max(float(adjusted.get("games", 0.0)), 0.0)
    reliability = games / (games + EARLY_SEASON_PRIOR_GAMES)
    for feature, baseline in MODEL_FEATURE_BASELINES.items():
        value = float(adjusted.get(feature, baseline))
        adjusted[feature] = baseline + reliability * (value - baseline)
    return adjusted


def _matchup_row(
    home: TeamState,
    away: TeamState,
    neutral: bool,
    season_progress: float,
) -> dict[str, float]:
    home_values = model_adjusted_snapshot(home.snapshot())
    away_values = model_adjusted_snapshot(away.snapshot())
    features: dict[str, float] = {}
    for name in TEAM_NUMERIC_FEATURES:
        features[f"{name}_diff"] = float(home_values[name]) - float(away_values[name])
    features["neutral_site"] = float(neutral)
    features["home_field"] = 0.0 if neutral else 1.0
    features["season_progress"] = season_progress
    return {name: features.get(name, 0.0) for name in MATCHUP_FEATURES}


def build_sequential_features(
    games: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
    team_game_stats: pd.DataFrame | None = None,
    advanced_game_stats: pd.DataFrame | None = None,
    *,
    k_factor: float = 20.0,
    offseason_reversion: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pregame matchup features and post-week snapshots without future leakage."""
    if games.empty:
        return pd.DataFrame(), pd.DataFrame()
    rankings = rankings if rankings is not None else pd.DataFrame()
    teams = teams if teams is not None else pd.DataFrame()
    team_game_stats = team_game_stats if team_game_stats is not None else pd.DataFrame()
    advanced_game_stats = (
        advanced_game_stats if advanced_game_stats is not None else pd.DataFrame()
    )
    ap_lookup = _pregame_ap_rank_lookup(rankings)
    stats_lookup = _team_stats_lookup(team_game_stats)
    advanced_lookup = _advanced_stats_lookup(advanced_game_stats)

    work = games.copy()
    work = work[work["completed"].fillna(False).astype(bool)].copy()
    work = work.dropna(subset=["season", "week", "home_team", "away_team", "home_points", "away_points"])
    work["season"] = work["season"].astype(int)
    work["week"] = work["week"].astype(int)
    work["start_ts"] = pd.to_datetime(work["start_date"], errors="coerce", utc=True)
    work["season_type_order"] = (
        work["season_type"].fillna("regular").astype(str).str.lower().map({"regular": 0, "postseason": 1}).fillna(2)
    )
    work = work.sort_values(["season", "season_type_order", "week", "start_ts", "game_id"])

    game_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    previous_season_elos: dict[str, float] = {}

    for season, season_games in work.groupby("season", sort=True):
        known_teams: set[str] = set()
        season_fbs: set[str] = set()
        if not teams.empty:
            season_teams = teams[pd.to_numeric(teams["season"], errors="coerce").eq(season)]
            season_fbs.update(season_teams["team"].dropna().astype(str))
            known_teams.update(season_fbs)
        known_teams.update(season_games["home_team"].dropna().astype(str))
        known_teams.update(season_games["away_team"].dropna().astype(str))
        states = {
            team: TeamState(
                team=team,
                elo=_initial_elo(
                    team,
                    previous_season_elos,
                    offseason_reversion,
                    baseline=1500.0 if team in season_fbs else 1350.0,
                ),
            )
            for team in known_teams
        }

        group_columns = ["season_type_order", "season_type", "week"]
        for (_, season_type, week), week_games in season_games.groupby(group_columns, sort=True):
            last_week_results: dict[str, dict[str, Any]] = {}
            for game in week_games.itertuples():
                home = states.setdefault(str(game.home_team), TeamState(str(game.home_team)))
                away = states.setdefault(str(game.away_team), TeamState(str(game.away_team)))
                home_pre = home.snapshot()
                away_pre = away.snapshot()
                neutral = bool(game.neutral_site)
                progress = min(max(int(week), 0) / 15.0, 1.0)
                matchup = _matchup_row(home, away, neutral, progress)
                home_margin = float(game.home_points) - float(game.away_points)
                game_rows.append(
                    {
                        "game_id": game.game_id,
                        "season": season,
                        "season_type": season_type,
                        "week": int(week),
                        "start_date": game.start_date,
                        "home_team": game.home_team,
                        "away_team": game.away_team,
                        **_prefix_snapshot(home_pre, "home"),
                        **_prefix_snapshot(away_pre, "away"),
                        **matchup,
                        "home_margin": home_margin,
                        "game_total": float(game.home_points) + float(game.away_points),
                    }
                )

                home_rank = ap_lookup.get((season, int(week), str(game.home_team)))
                away_rank = ap_lookup.get((season, int(week), str(game.away_team)))
                home_elo_before, away_elo_before = home.elo, away.elo
                home.games += 1
                away.games += 1
                home.season_games += 1
                away.season_games += 1
                home.points_for += float(game.home_points)
                home.points_against += float(game.away_points)
                away.points_for += float(game.away_points)
                away.points_against += float(game.home_points)
                home.margin_sum += home_margin
                away.margin_sum -= home_margin
                home.opponent_elo_sum += away_elo_before
                away.opponent_elo_sum += home_elo_before
                home.recent_margins.append(home_margin)
                away.recent_margins.append(-home_margin)

                # Quality-weighted accumulators: scale each game's contribution
                # by the opponent's Elo relative to the 1500 baseline so that
                # blowouts against weak opponents don't inflate stats as much.
                home_opp_quality = away_elo_before / 1500.0
                away_opp_quality = home_elo_before / 1500.0
                home.quality_margin_sum += home_margin * home_opp_quality
                home.quality_score_sum += float(game.home_points) * home_opp_quality
                home.quality_allowed_sum += float(game.away_points) * home_opp_quality
                home.quality_weight_sum += home_opp_quality
                away.quality_margin_sum += (-home_margin) * away_opp_quality
                away.quality_score_sum += float(game.away_points) * away_opp_quality
                away.quality_allowed_sum += float(game.home_points) * away_opp_quality
                away.quality_weight_sum += away_opp_quality

                if home_margin > 0:
                    home.wins += 1
                    away.losses += 1
                    if away_rank and away_rank <= 25:
                        home.ranked_wins += 1
                    if away_elo_before >= 1600:
                        home.quality_wins += 1
                    if home_elo_before < 1450:
                        away.bad_losses += 1
                elif home_margin < 0:
                    away.wins += 1
                    home.losses += 1
                    away.road_wins += int(not neutral)
                    if home_rank and home_rank <= 25:
                        away.ranked_wins += 1
                    if home_elo_before >= 1600:
                        away.quality_wins += 1
                    if away_elo_before < 1450:
                        home.bad_losses += 1
                else:
                    home.ties += 1
                    away.ties += 1
                home.results.append(
                    (
                        str(game.away_team),
                        1 if home_margin > 0 else -1 if home_margin < 0 else 0,
                    )
                )
                away.results.append(
                    (
                        str(game.home_team),
                        1 if home_margin < 0 else -1 if home_margin > 0 else 0,
                    )
                )

                home_stats = stats_lookup.get((game.game_id, str(game.home_team)), {})
                away_stats = stats_lookup.get((game.game_id, str(game.away_team)), {})
                home_yards = _find_stat(home_stats, ("totalYards", "total yards"))
                away_yards = _find_stat(away_stats, ("totalYards", "total yards"))
                home_turnovers = _find_stat(home_stats, ("turnovers", "turnoversLost"))
                away_turnovers = _find_stat(away_stats, ("turnovers", "turnoversLost"))
                if home_yards is not None and away_yards is not None:
                    home.stat_games += 1
                    away.stat_games += 1
                    home.yards_for_sum += home_yards
                    home.yards_against_sum += away_yards
                    away.yards_for_sum += away_yards
                    away.yards_against_sum += home_yards
                    if home_turnovers is not None and away_turnovers is not None:
                        home.turnovers_lost_sum += home_turnovers
                        home.turnovers_forced_sum += away_turnovers
                        away.turnovers_lost_sum += away_turnovers
                        away.turnovers_forced_sum += home_turnovers

                home_advanced = advanced_lookup.get((game.game_id, str(game.home_team)), {})
                away_advanced = advanced_lookup.get((game.game_id, str(game.away_team)), {})
                if home_advanced and away_advanced:
                    home.advanced_games += 1
                    away.advanced_games += 1
                    home.offense_ppa_sum += home_advanced.get("offense_ppa", 0.0)
                    home.defense_ppa_sum += home_advanced.get("defense_ppa", 0.0)
                    away.offense_ppa_sum += away_advanced.get("offense_ppa", 0.0)
                    away.defense_ppa_sum += away_advanced.get("defense_ppa", 0.0)
                    home.offense_success_rate_sum += home_advanced.get(
                        "offense_success_rate", 0.0
                    )
                    home.defense_success_rate_sum += home_advanced.get(
                        "defense_success_rate", 0.0
                    )
                    away.offense_success_rate_sum += away_advanced.get(
                        "offense_success_rate", 0.0
                    )
                    away.defense_success_rate_sum += away_advanced.get(
                        "defense_success_rate", 0.0
                    )

                _update_elo(home, away, home_margin, neutral, k_factor)
                last_week_results[str(game.home_team)] = {
                    "last_week_won": int(home_margin > 0),
                    "last_week_margin": home_margin,
                    "last_week_opponent_elo": away_elo_before,
                    "last_week_road": 0,
                    "last_week_ranked_opponent": int(bool(away_rank and away_rank <= 25)),
                }
                last_week_results[str(game.away_team)] = {
                    "last_week_won": int(home_margin < 0),
                    "last_week_margin": -home_margin,
                    "last_week_opponent_elo": home_elo_before,
                    "last_week_road": int(not neutral),
                    "last_week_ranked_opponent": int(bool(home_rank and home_rank <= 25)),
                }

            # Revalue each résumé using opponents' strength as it exists this week.
            for state in states.values():
                opponent_elos = [
                    states[opponent].elo
                    for opponent, _ in state.results
                    if opponent in states
                ]
                state.opponent_elo_sum = float(sum(opponent_elos))
                state.quality_wins = sum(
                    result > 0 and opponent in states and states[opponent].elo >= 1600
                    for opponent, result in state.results
                )
                state.bad_losses = sum(
                    result < 0 and opponent in states and states[opponent].elo < 1450
                    for opponent, result in state.results
                )

            for team_name, state in states.items():
                weekly_rows.append(
                    {
                        "season": season,
                        "season_type": season_type,
                        "game_week": int(week),
                        **state.snapshot(),
                        **last_week_results.get(
                            team_name,
                            {
                                "last_week_won": 0,
                                "last_week_margin": 0.0,
                                "last_week_opponent_elo": 1500.0,
                                "last_week_road": 0,
                                "last_week_ranked_opponent": 0,
                            },
                        ),
                    }
                )
        # Only FBS membership carries an Elo into the next season. This prevents years of
        # lower-division results from becoming an inherited top-FBS prior after promotion.
        previous_season_elos = {
            team: states[team].elo for team in season_fbs if team in states
        }

    game_features = pd.DataFrame(game_rows)
    team_week_features = pd.DataFrame(weekly_rows)
    return game_features, team_week_features


def build_ap_training_frame(
    team_week_features: pd.DataFrame,
    rankings: pd.DataFrame,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align end-of-game-week features to the following AP poll."""
    if team_week_features.empty or rankings.empty:
        return pd.DataFrame()
    ap = ap_poll_rows(rankings)
    if ap.empty:
        return pd.DataFrame()
    ap = ap.copy()
    ap["season"] = pd.to_numeric(ap["season"], errors="coerce").astype("Int64")
    ap["poll_week"] = pd.to_numeric(ap["poll_week"], errors="coerce").astype("Int64")
    features = team_week_features.copy()
    features["target_poll_week"] = features["game_week"].astype(int) + 1

    # Limit candidates to FBS teams when a roster is available.
    if teams is not None and not teams.empty:
        fbs = teams[["season", "team"]].drop_duplicates()
        fbs["season"] = pd.to_numeric(fbs["season"], errors="coerce").astype("Int64")
        features = features.merge(fbs.assign(is_fbs=1), on=["season", "team"], how="inner")

    targets = ap[[
        "season", "season_type", "poll_week", "team", "rank", "points", "first_place_votes"
    ]].rename(
        columns={
            "poll_week": "target_poll_week",
            "rank": "actual_ap_rank",
            "points": "actual_ap_points",
            "first_place_votes": "actual_first_place_votes",
        }
    )
    prior = ap[[
        "season", "season_type", "poll_week", "team", "rank", "points", "first_place_votes"
    ]].copy()
    prior["target_poll_week"] = prior["poll_week"].astype(int) + 1
    prior = prior.rename(
        columns={
            "rank": "previous_ap_rank",
            "points": "previous_ap_points",
            "first_place_votes": "previous_first_place_votes",
        }
    ).drop(columns="poll_week")

    output = features.merge(
        prior,
        on=["season", "season_type", "target_poll_week", "team"],
        how="left",
    ).merge(
        targets,
        on=["season", "season_type", "target_poll_week", "team"],
        how="left",
    )
    # Keep only weeks for which the target AP poll exists.
    existing_groups = targets[["season", "season_type", "target_poll_week"]].drop_duplicates()
    output = output.merge(
        existing_groups.assign(target_poll_exists=1),
        on=["season", "season_type", "target_poll_week"],
        how="inner",
    )
    output["previously_ranked"] = output["previous_ap_rank"].notna().astype(int)
    output["previous_ap_rank_filled"] = output["previous_ap_rank"].fillna(40).clip(upper=40)
    output["previous_ap_points"] = output["previous_ap_points"].fillna(0)
    output["previous_first_place_votes"] = output["previous_first_place_votes"].fillna(0)
    output["actual_ap_points"] = output["actual_ap_points"].fillna(0)
    output["actual_first_place_votes"] = output["actual_first_place_votes"].fillna(0)
    output["actual_ap_rank"] = output["actual_ap_rank"].fillna(999).astype(int)
    output["poll_group"] = (
        output["season"].astype(str)
        + "-"
        + output["season_type"].astype(str)
        + "-"
        + output["target_poll_week"].astype(str)
    )
    return output.sort_values(["season", "target_poll_week", "team"]).reset_index(drop=True)


def current_ap_feature_frame(
    team_week_features: pd.DataFrame,
    rankings: pd.DataFrame,
    season: int,
    teams: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    season_features = team_week_features[
        pd.to_numeric(team_week_features["season"], errors="coerce").eq(season)
    ].copy()
    if season_features.empty:
        raise ValueError(f"No completed-game features exist for {season}")
    regular = season_features[
        season_features["season_type"].astype(str).str.lower().eq("regular")
    ]
    latest_game_week = int(regular["game_week"].max())
    target_week = latest_game_week + 1
    current = regular[regular["game_week"].eq(latest_game_week)].copy()
    if teams is not None and not teams.empty:
        fbs_names = set(
            teams[pd.to_numeric(teams["season"], errors="coerce").eq(season)]["team"]
            .dropna()
            .astype(str)
        )
        current = current[current["team"].isin(fbs_names)]

    ap = ap_poll_rows(rankings)
    prior = ap[
        pd.to_numeric(ap["season"], errors="coerce").eq(season)
        & pd.to_numeric(ap["poll_week"], errors="coerce").eq(target_week - 1)
    ][["team", "rank", "points", "first_place_votes"]].rename(
        columns={
            "rank": "previous_ap_rank",
            "points": "previous_ap_points",
            "first_place_votes": "previous_first_place_votes",
        }
    )
    current = current.merge(prior, on="team", how="left")
    current["previously_ranked"] = current["previous_ap_rank"].notna().astype(int)
    current["previous_ap_rank_filled"] = current["previous_ap_rank"].fillna(40).clip(upper=40)
    current["previous_ap_points"] = current["previous_ap_points"].fillna(0)
    current["previous_first_place_votes"] = current["previous_first_place_votes"].fillna(0)
    current["target_poll_week"] = target_week
    return current.reset_index(drop=True), target_week


def latest_team_states(team_week_features: pd.DataFrame, season: int) -> pd.DataFrame:
    current = team_week_features[
        pd.to_numeric(team_week_features["season"], errors="coerce").eq(season)
    ].copy()
    if current.empty:
        return current
    regular = current[current["season_type"].astype(str).str.lower().eq("regular")]
    latest_week = int(regular["game_week"].max())
    return regular[regular["game_week"].eq(latest_week)].copy().reset_index(drop=True)


def matchup_features_from_snapshots(
    home: pd.Series | dict[str, Any],
    away: pd.Series | dict[str, Any],
    *,
    neutral_site: bool,
    week: int,
) -> dict[str, float]:
    home_values = model_adjusted_snapshot(dict(home))
    away_values = model_adjusted_snapshot(dict(away))
    output: dict[str, float] = {}
    for name in TEAM_NUMERIC_FEATURES:
        output[f"{name}_diff"] = float(home_values.get(name, 0.0)) - float(
            away_values.get(name, 0.0)
        )
    output["neutral_site"] = float(neutral_site)
    output["home_field"] = 0.0 if neutral_site else 1.0
    output["season_progress"] = min(max(week, 0) / 15.0, 1.0)
    return {name: output.get(name, 0.0) for name in MATCHUP_FEATURES}
