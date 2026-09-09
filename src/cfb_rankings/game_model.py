from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import DMatrix, XGBRegressor

from .features import MATCHUP_FEATURES, model_adjusted_snapshot


@dataclass
class GameModelBundle:
    margin_model: XGBRegressor
    total_model: XGBRegressor
    imputer: SimpleImputer
    features: list[str]
    residual_mean: float
    residual_std: float
    residual_q10: float
    residual_q90: float
    selected_parameters: dict[str, Any]
    probability_calibration: tuple[float, float] = (0.0, 0.0)
    schema_version: int = 6

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(frame.reindex(columns=self.features))
        return np.asarray(self.margin_model.predict(matrix), dtype=float)

    def predict_total(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(frame.reindex(columns=self.features))
        return np.asarray(self.total_model.predict(matrix), dtype=float)

    def predict_margin_contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """Return TreeSHAP contributions in the same order as ``self.features``.

        XGBoost appends the expected-value bias as the final column. Keeping it here lets
        callers verify that feature contributions sum back to the prediction.
        """
        matrix = self.imputer.transform(frame.reindex(columns=self.features))
        return np.asarray(
            self.margin_model.get_booster().predict(DMatrix(matrix), pred_contribs=True),
            dtype=float,
        )

    def win_probability(self, predicted_margin: np.ndarray) -> np.ndarray:
        from .validated_training import apply_probability
        return apply_probability(predicted_margin, self.probability_calibration)

    def fundamental_cover_probability(self, predicted_margin: np.ndarray) -> np.ndarray:
        scale = max(self.residual_std, 1e-6)
        z = (np.asarray(predicted_margin, dtype=float) + self.residual_mean) / scale
        return np.asarray([0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in z])


def select_and_train_game_model(
    game_features: pd.DataFrame,
    models_dir: Path,
    *,
    validation_seasons: int = 4,
    validation_cutoff_season: int | None = None,
    random_state: int = 42,
) -> tuple[GameModelBundle, pd.DataFrame]:
    from .validated_training import select_and_train_game_model as train

    return train(
        game_features, models_dir,
        validation_seasons=validation_seasons,
        validation_cutoff_season=validation_cutoff_season,
        random_state=random_state,
    )


def load_game_model(models_dir: Path) -> GameModelBundle:
    path = models_dir / "independent_xgboost.joblib"
    if not path.exists():
        raise FileNotFoundError("Independent XGBoost model has not been trained yet")
    bundle = joblib.load(path)
    if bundle.__dict__.get("schema_version") != 6:
        raise ValueError("Model predates the v0.6 feature schema. Run cfb build-features and cfb train.")
    return bundle


def _pair_frame(team_a: pd.Series, team_b: pd.Series) -> pd.DataFrame:
    team_a = pd.Series(model_adjusted_snapshot(team_a.to_dict()))
    team_b = pd.Series(model_adjusted_snapshot(team_b.to_dict()))
    row = {}
    for name in [feature.removesuffix("_diff") for feature in MATCHUP_FEATURES if feature.endswith("_diff")]:
        row[f"{name}_diff"] = float(team_a.get(name, 0.0)) - float(team_b.get(name, 0.0))
    current_week = max(float(team_a.get("game_week", 0.0)), float(team_b.get("game_week", 0.0)))
    row.update(
        {
            "neutral_site": 1.0,
            "home_field": 0.0,
            "season_progress": min(max(current_week, 0.0) / 15.0, 1.0),
        }
    )
    return pd.DataFrame([{name: row.get(name, 0.0) for name in MATCHUP_FEATURES}])


CONTRIBUTION_GROUPS = {
    "power_contribution": {"elo_diff"},
    "resume_contribution": {
        "games_diff",
        "wins_diff",
        "losses_diff",
        "win_pct_diff",
        "road_wins_diff",
    },
    "efficiency_contribution": {
        "points_per_game_diff",
        "points_allowed_per_game_diff",
        "avg_margin_diff",
        "yards_per_game_diff",
        "yards_allowed_per_game_diff",
        "turnover_margin_per_game_diff",
        "offense_ppa_diff",
        "defense_ppa_diff",
        "offense_success_rate_diff",
        "defense_success_rate_diff",
    },
    "sos_contribution": {"sos_elo_diff", "quality_adj_margin_diff",
                         "quality_adj_ppg_diff", "quality_adj_ppg_allowed_diff"},
    "quality_wins_contribution": {"ranked_wins_diff", "quality_wins_diff"},
    "bad_losses_contribution": {"bad_losses_diff"},
    "recent_form_contribution": {"recent_margin_3_diff"},
}

CONTRIBUTION_GROUPS["efficiency_contribution"].update({
    f"{name}_diff" for name in (
        "rushing_ypg", "rushing_ypg_allowed", "passing_ypg", "passing_ypg_allowed",
        "yards_per_rush", "yards_per_rush_allowed", "yards_per_pass", "yards_per_pass_allowed",
        "third_down_pct", "third_down_pct_allowed", "first_downs_pg", "first_downs_pg_allowed",
        "penalty_yards_pg", "possession_time_pg",
    )
})


def build_independent_rankings(
    bundle: GameModelBundle,
    current_team_features: pd.DataFrame,
    fbs_teams: set[str] | None = None,
) -> pd.DataFrame:
    work = current_team_features.copy()
    if fbs_teams:
        work = work[work["team"].astype(str).isin(fbs_teams)]
    work = work.drop_duplicates("team").set_index("team", drop=False)
    names = sorted(work.index.astype(str))
    margin_sums = {team: 0.0 for team in names}
    comparisons = {team: 0 for team in names}
    feature_sums = {team: np.zeros(len(bundle.features), dtype=float) for team in names}
    pair_names: list[tuple[str, str]] = []
    comparison_rows: list[dict[str, float]] = []
    for team_a, team_b in combinations(names, 2):
        pair_names.append((team_a, team_b))
        comparison_rows.append(_pair_frame(work.loc[team_a], work.loc[team_b]).iloc[0].to_dict())
        comparison_rows.append(_pair_frame(work.loc[team_b], work.loc[team_a]).iloc[0].to_dict())
    predicted_margins = (
        bundle.predict_margin(pd.DataFrame(comparison_rows))
        if comparison_rows
        else np.array([], dtype=float)
    )
    predicted_contributions = (
        bundle.predict_margin_contributions(pd.DataFrame(comparison_rows))
        if comparison_rows
        else np.empty((0, len(bundle.features) + 1), dtype=float)
    )
    for pair_index, (team_a, team_b) in enumerate(pair_names):
        forward = float(predicted_margins[2 * pair_index])
        reverse = float(predicted_margins[2 * pair_index + 1])
        antisymmetric_margin = (forward - reverse) / 2.0
        antisymmetric_contributions = (
            predicted_contributions[2 * pair_index, :-1]
            - predicted_contributions[2 * pair_index + 1, :-1]
        ) / 2.0
        margin_sums[team_a] += antisymmetric_margin
        margin_sums[team_b] -= antisymmetric_margin
        feature_sums[team_a] += antisymmetric_contributions
        feature_sums[team_b] -= antisymmetric_contributions
        comparisons[team_a] += 1
        comparisons[team_b] += 1
    rows = []
    for team in names:
        state = work.loc[team]
        average_contributions = feature_sums[team] / max(comparisons[team], 1)
        by_feature = dict(zip(bundle.features, average_contributions, strict=True))
        grouped_contributions = {
            output_name: float(sum(by_feature.get(feature, 0.0) for feature in features))
            for output_name, features in CONTRIBUTION_GROUPS.items()
        }
        rows.append(
            {
                "team": team,
                "model_rating": margin_sums[team] / max(comparisons[team], 1),
                "wins": int(state.get("wins", 0)),
                "losses": int(state.get("losses", 0)),
                "elo": float(state.get("elo", 1500.0)),
                "sos_elo": float(state.get("sos_elo", 1500.0)),
                "ranked_wins": int(state.get("ranked_wins", 0)),
                "bad_losses": int(state.get("bad_losses", 0)),
                **grouped_contributions,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["model_rating", "elo"], ascending=[False, False]
    ).reset_index(drop=True)
    output.insert(0, "independent_rank", np.arange(1, len(output) + 1))
    return output


def consensus_current_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "market_home_margin",
                "consensus_spread",
                "home_spread",
                "away_spread",
                "home_spread_odds",
                "away_spread_odds",
                "spread_odds_sportsbook",
                "spread_odds_estimated",
                "home_moneyline",
                "away_moneyline",
                "home_moneyline_sportsbook",
                "away_moneyline_sportsbook",
                "line_fetched_at",
                "sportsbooks",
            ]
        )
    work = lines.copy()
    for column in (
        "spread",
        "home_spread_odds",
        "away_spread_odds",
        "home_moneyline",
        "away_moneyline",
    ):
        if column not in work:
            work[column] = np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["fetched_ts"] = pd.to_datetime(work["fetched_at"], errors="coerce", utc=True)
    work = work.dropna(subset=["game_id", "fetched_ts"])
    work = work[
        work[["spread", "home_moneyline", "away_moneyline"]].notna().any(axis=1)
    ]
    if work.empty:
        return consensus_current_lines(pd.DataFrame())
    # Keep each provider's most recent quote. Use the median spread and the best
    # available moneyline because EV should reflect the price a bettor can actually take.
    work = work.sort_values("fetched_ts").drop_duplicates(["game_id", "provider"], keep="last")
    rows: list[dict[str, Any]] = []
    for game_id, quotes in work.groupby("game_id", sort=False):
        consensus_spread = quotes["spread"].median()
        spread_quotes = quotes.dropna(subset=["spread"]).copy()
        if not spread_quotes.empty:
            spread_quotes["distance_from_consensus"] = (
                spread_quotes["spread"] - consensus_spread
            ).abs()
            spread_quote = spread_quotes.sort_values(
                ["distance_from_consensus", "fetched_ts"],
                ascending=[True, False],
            ).iloc[0]
        else:
            spread_quote = None
        actual_home_spread_odds = (
            spread_quote.get("home_spread_odds") if spread_quote is not None else np.nan
        )
        actual_away_spread_odds = (
            spread_quote.get("away_spread_odds") if spread_quote is not None else np.nan
        )
        spread_odds_estimated = bool(
            pd.isna(actual_home_spread_odds) or pd.isna(actual_away_spread_odds)
        )
        home_spread_odds = (
            -110.0 if pd.isna(actual_home_spread_odds) and pd.notna(consensus_spread)
            else actual_home_spread_odds
        )
        away_spread_odds = (
            -110.0 if pd.isna(actual_away_spread_odds) and pd.notna(consensus_spread)
            else actual_away_spread_odds
        )
        home_quotes = quotes.dropna(subset=["home_moneyline"])
        away_quotes = quotes.dropna(subset=["away_moneyline"])
        best_home = (
            home_quotes.loc[home_quotes["home_moneyline"].idxmax()]
            if not home_quotes.empty
            else None
        )
        best_away = (
            away_quotes.loc[away_quotes["away_moneyline"].idxmax()]
            if not away_quotes.empty
            else None
        )
        rows.append(
            {
                "game_id": game_id,
                "consensus_spread": consensus_spread,
                "home_spread": spread_quote["spread"] if spread_quote is not None else np.nan,
                "away_spread": -spread_quote["spread"] if spread_quote is not None else np.nan,
                "home_spread_odds": home_spread_odds,
                "away_spread_odds": away_spread_odds,
                "spread_odds_sportsbook": (
                    spread_quote["provider"]
                    if spread_quote is not None and not spread_odds_estimated
                    else "Standard -110 estimate"
                ),
                "spread_odds_estimated": spread_odds_estimated,
                "home_moneyline": (
                    best_home["home_moneyline"] if best_home is not None else np.nan
                ),
                "away_moneyline": (
                    best_away["away_moneyline"] if best_away is not None else np.nan
                ),
                "home_moneyline_sportsbook": (
                    best_home["provider"] if best_home is not None else None
                ),
                "away_moneyline_sportsbook": (
                    best_away["provider"] if best_away is not None else None
                ),
                "line_fetched_at": quotes["fetched_ts"].max(),
                "sportsbooks": ", ".join(sorted(set(map(str, quotes["provider"])))),
            }
        )
    aggregated = pd.DataFrame(rows)
    # CFBD's spread is the home-team handicap, so -3 implies a home margin of +3.
    # Attach prices to their actual quoted handicap, not an unbettable median.
    aggregated["market_home_margin"] = -aggregated["home_spread"]
    return aggregated


def _moneyline_profit_per_100(odds: pd.Series) -> pd.Series:
    valid = odds.where(odds.ne(0))
    return pd.Series(
        np.where(valid > 0, valid, 10000.0 / valid.abs()),
        index=odds.index,
        dtype=float,
    ).where(valid.notna())


def _moneyline_expected_value(probability: pd.Series, odds: pd.Series) -> pd.Series:
    profit = _moneyline_profit_per_100(odds)
    return probability * profit - (1.0 - probability) * 100.0


def predict_upcoming_games(
    bundle: GameModelBundle,
    games: pd.DataFrame,
    current_team_features: pd.DataFrame,
    lines: pd.DataFrame,
    season: int,
    *,
    next_slate_only: bool = True,
) -> pd.DataFrame:
    schedule = games[pd.to_numeric(games["season"], errors="coerce").eq(season)].copy()
    completed = schedule["completed"].fillna(False)
    if completed.dtype != bool:
        completed = completed.astype(str).str.lower().isin({"true", "1", "yes"})
    schedule = schedule[~completed].copy()
    if next_slate_only and not schedule.empty:
        schedule = schedule[
            pd.to_numeric(schedule["week"], errors="coerce").eq(
                pd.to_numeric(schedule["week"], errors="coerce").min()
            )
        ].copy()
    if schedule.empty:
        return pd.DataFrame()
    states = current_team_features.drop_duplicates("team").set_index("team")
    rows: list[dict[str, Any]] = []
    for game in schedule.itertuples():
        if str(game.home_team) not in states.index or str(game.away_team) not in states.index:
            continue
        home = states.loc[str(game.home_team)]
        away = states.loc[str(game.away_team)]
        neutral = bool(game.neutral_site)
        matchup = _pair_frame(home, away).iloc[0].to_dict()
        matchup["neutral_site"] = float(neutral)
        matchup["home_field"] = 0.0 if neutral else 1.0
        matchup["season_progress"] = min(max(int(game.week), 0) / 15.0, 1.0)
        rows.append(
            {
                "game_id": game.game_id,
                "start_date": game.start_date,
                "week": int(game.week),
                "home_team": game.home_team,
                "away_team": game.away_team,
                "neutral_site": neutral,
                **matchup,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["model_home_margin"] = bundle.predict_margin(output)
    output["predicted_total"] = np.maximum(bundle.predict_total(output), 10.0)
    output["predicted_home_score"] = np.maximum(
        (output["predicted_total"] + output["model_home_margin"]) / 2.0, 0.0
    )
    output["predicted_away_score"] = np.maximum(
        (output["predicted_total"] - output["model_home_margin"]) / 2.0, 0.0
    )
    output["home_win_probability"] = bundle.win_probability(
        output["model_home_margin"].to_numpy()
    )
    output["margin_low_80"] = output["model_home_margin"] + bundle.residual_q10
    output["margin_high_80"] = output["model_home_margin"] + bundle.residual_q90
    output = output.merge(consensus_current_lines(lines), on="game_id", how="left")
    output["home_moneyline_ev_100"] = _moneyline_expected_value(
        output["home_win_probability"], output["home_moneyline"]
    )
    output["away_moneyline_ev_100"] = _moneyline_expected_value(
        1.0 - output["home_win_probability"], output["away_moneyline"]
    )
    home_is_best = output["home_moneyline_ev_100"].fillna(-np.inf).ge(
        output["away_moneyline_ev_100"].fillna(-np.inf)
    )
    output["best_moneyline_team"] = np.where(
        home_is_best, output["home_team"], output["away_team"]
    )
    output["best_moneyline"] = np.where(
        home_is_best, output["home_moneyline"], output["away_moneyline"]
    )
    output["best_ev_100"] = np.where(
        home_is_best, output["home_moneyline_ev_100"], output["away_moneyline_ev_100"]
    )
    no_moneyline = output[["home_moneyline", "away_moneyline"]].isna().all(axis=1)
    output.loc[no_moneyline, ["best_moneyline_team", "best_moneyline", "best_ev_100"]] = np.nan
    output["model_edge_home"] = output["model_home_margin"] - output["market_home_margin"]
    output["fundamental_edge_home"] = output["model_edge_home"]
    output["edge_model_validated"] = False
    output["betting_model_home_margin"] = np.nan
    # A winner probability is not a cover probability. Populate spread EV only
    # after the separately calibrated market-aware model passes its guardrails.
    output["cover_probability_home"] = np.nan
    output["cover_probability_away"] = 1.0 - output["cover_probability_home"]
    output["home_spread_ev_100"] = _moneyline_expected_value(
        output["cover_probability_home"], output["home_spread_odds"]
    )
    output["away_spread_ev_100"] = _moneyline_expected_value(
        output["cover_probability_away"], output["away_spread_odds"]
    )
    output["model_ats_lean"] = np.where(
        output["model_edge_home"].isna(),
        "No line",
        np.where(output["model_edge_home"] >= 0, output["home_team"], output["away_team"]),
    )
    output = refresh_best_props(output)
    keep = [
        "game_id", "start_date", "week", "away_team", "home_team", "neutral_site",
        "predicted_away_score", "predicted_home_score", "model_home_margin",
        "margin_low_80", "margin_high_80", "home_win_probability", "consensus_spread",
        "market_home_margin", "model_edge_home", "cover_probability_home",
        "cover_probability_away", "model_ats_lean", "home_spread", "away_spread",
        "home_spread_odds", "away_spread_odds", "spread_odds_sportsbook",
        "spread_odds_estimated", "home_spread_ev_100", "away_spread_ev_100",
        "home_moneyline", "away_moneyline", "home_moneyline_sportsbook",
        "away_moneyline_sportsbook", "home_moneyline_ev_100", "away_moneyline_ev_100",
        "best_moneyline_team", "best_moneyline", "best_ev_100", "sportsbooks",
        "best_prop_team", "best_prop_type", "best_prop_line", "best_prop_odds",
        "best_prop_ev_100", "line_fetched_at", "fundamental_edge_home",
        "edge_model_validated", "betting_model_home_margin",
    ]
    # Preserve pregame inputs for the edge model which runs after this function.
    keep = list(dict.fromkeys(keep + [f for f in MATCHUP_FEATURES if f in output]))
    return output[keep].sort_values("start_date").reset_index(drop=True)


def refresh_best_props(output: pd.DataFrame) -> pd.DataFrame:
    """Recompute displayed best plays after any probability/EV update."""
    if output.empty:
        return output
    output = output.drop(columns=[c for c in output if c.startswith("best_prop_")])
    prop_specs = [
        ("home", "spread", "home_spread", "home_spread_odds", "home_spread_ev_100"),
        ("away", "spread", "away_spread", "away_spread_odds", "away_spread_ev_100"),
        ("home", "moneyline", None, "home_moneyline", "home_moneyline_ev_100"),
        ("away", "moneyline", None, "away_moneyline", "away_moneyline_ev_100"),
    ]
    best_prop_rows: list[dict[str, Any]] = []
    for row in output.to_dict(orient="records"):
        candidates: list[dict[str, Any]] = []
        for side, prop_type, line_column, odds_column, ev_column in prop_specs:
            ev = row.get(ev_column)
            odds_value = row.get(odds_column)
            if pd.isna(ev) or pd.isna(odds_value):
                continue
            candidates.append(
                {
                    "best_prop_team": row[f"{side}_team"],
                    "best_prop_type": prop_type,
                    "best_prop_line": row.get(line_column) if line_column else np.nan,
                    "best_prop_odds": odds_value,
                    "best_prop_ev_100": ev,
                }
            )
        best_prop_rows.append(
            max(candidates, key=lambda candidate: candidate["best_prop_ev_100"])
            if candidates
            else {
                "best_prop_team": None,
                "best_prop_type": None,
                "best_prop_line": np.nan,
                "best_prop_odds": np.nan,
                "best_prop_ev_100": np.nan,
            }
        )
    output = pd.concat([output.reset_index(drop=True), pd.DataFrame(best_prop_rows)], axis=1)
    return output


def apply_edge_predictions(output: pd.DataFrame, bundle: EdgeModelBundle | None) -> pd.DataFrame:
    """Use the validated calibration and retain a distinct independent margin."""
    if output.empty or bundle is None or not bundle.validated:
        return refresh_best_props(output)
    output = output.copy()
    mask = output["market_home_margin"].notna()
    if mask.any():
        edge = bundle.predict_edge(_build_edge_features(output.loc[mask].copy()))
        probability = bundle.cover_probability(edge)
        output.loc[mask, "model_edge_home"] = edge
        output.loc[mask, "betting_model_home_margin"] = output.loc[mask, "market_home_margin"] + edge
        output.loc[mask, "edge_model_validated"] = True
        output.loc[mask, "cover_probability_home"] = probability
        output.loc[mask, "cover_probability_away"] = 1 - probability
        output.loc[mask, "model_ats_lean"] = np.where(
            probability >= .5, output.loc[mask, "home_team"], output.loc[mask, "away_team"]
        )
        for side in ("home", "away"):
            output.loc[mask, f"{side}_spread_ev_100"] = _moneyline_expected_value(
                output.loc[mask, f"cover_probability_{side}"], output.loc[mask, f"{side}_spread_odds"]
            )
            # Calibration conditions on a decisive result. Without a calibrated
            # push probability, do not advertise dollar EV for integer lines.
            integer_line = mask & output["home_spread"].mod(1).eq(0)
            output.loc[integer_line, f"{side}_spread_ev_100"] = np.nan
    return refresh_best_props(output)


def build_season_schedule(
    games: pd.DataFrame,
    lines: pd.DataFrame,
    future_predictions: pd.DataFrame,
    season: int,
    fbs_teams: set[str] | None = None,
) -> pd.DataFrame:
    """Build one row per current-season game for team schedule pages."""
    schedule = games[pd.to_numeric(games["season"], errors="coerce").eq(season)].copy()
    if fbs_teams:
        schedule = schedule[
            schedule["home_team"].astype(str).isin(fbs_teams)
            | schedule["away_team"].astype(str).isin(fbs_teams)
        ]
    if schedule.empty:
        return schedule
    completed = schedule["completed"].fillna(False)
    if completed.dtype != bool:
        completed = completed.astype(str).str.lower().isin({"true", "1", "yes"})
    schedule["completed"] = completed
    keep = [
        "game_id",
        "season",
        "season_type",
        "week",
        "start_date",
        "completed",
        "neutral_site",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "venue",
    ]
    schedule = schedule[[column for column in keep if column in schedule.columns]]
    schedule = schedule.merge(consensus_current_lines(lines), on="game_id", how="left")
    if not future_predictions.empty:
        prediction_columns = [
            "game_id",
            "predicted_away_score",
            "predicted_home_score",
            "model_home_margin",
            "margin_low_80",
            "margin_high_80",
            "home_win_probability",
            "model_edge_home",
            "cover_probability_home",
            "cover_probability_away",
            "model_ats_lean",
            "home_spread_ev_100",
            "away_spread_ev_100",
            "home_moneyline_ev_100",
            "away_moneyline_ev_100",
            "best_moneyline_team",
            "best_moneyline",
            "best_ev_100",
            "best_prop_team",
            "best_prop_type",
            "best_prop_line",
            "best_prop_odds",
            "best_prop_ev_100",
        ]
        schedule = schedule.merge(
            future_predictions[prediction_columns], on="game_id", how="left"
        )
    return schedule.sort_values(["start_date", "game_id"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Edge model — learns where the market systematically misprices games.
#  Uses SITUATIONAL features (spread size, week, home dog, Elo vs spread
#  disagreement) alongside team stats. These capture known market biases
#  that raw team quality stats can't — because Vegas already prices those in.
# ---------------------------------------------------------------------------


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Safely extract a numeric column, returning a filled Series even if missing."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def _build_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer situational columns that expose known market inefficiencies."""
    out = df.copy()
    mhm = _safe_col(out, "market_home_margin")
    out["spread_magnitude"] = mhm.abs()
    out["home_is_underdog"] = (mhm < 0).astype(float)
    out["is_large_spread"] = (mhm.abs() >= 20).astype(float)
    week = _safe_col(out, "week", 5.0)
    out["early_season"] = (week <= 3).astype(float)
    elo_margin = _safe_col(out, "elo_diff") * 0.04
    out["elo_spread_disagreement"] = elo_margin - mhm
    out["season_games_diff_feat"] = _safe_col(out, "season_games_diff")
    out["season_maturity"] = _safe_col(out, "season_progress", 0.33)
    recent_home = _safe_col(out, "recent_margin_3_diff")
    out["recent_form_vs_spread"] = recent_home - mhm
    return out


# The edge model uses all MATCHUP_FEATURES (team quality) plus the market
# spread and engineered situational features.
EDGE_SITUATIONAL = [
    "market_home_margin",
    "spread_magnitude",
    "home_is_underdog",
    "is_large_spread",
    "early_season",
    "elo_spread_disagreement",
    "recent_form_vs_spread",
]

EDGE_FEATURES = MATCHUP_FEATURES + EDGE_SITUATIONAL

@dataclass
class EdgeModelBundle:
    model: XGBRegressor
    imputer: SimpleImputer
    features: list[str]
    residual_std: float
    shrinkage: tuple[float, float] = (0.0, 0.0)
    probability_calibration: tuple[float, float] = (0.0, 0.0)
    validated: bool = False
    schema_version: int = 6

    def predict_edge(self, frame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing live edge inputs: {sorted(missing)}")
        matrix = self.imputer.transform(frame[self.features])
        raw = np.asarray(self.model.predict(matrix), float)
        return self.shrinkage[0] + self.shrinkage[1]*raw

    def cover_probability(self, edge):
        from .validated_training import apply_probability
        return apply_probability(edge, self.probability_calibration)


def train_edge_model(
    game_features: pd.DataFrame,
    models_dir: Path,
    *,
    validation_seasons: int = 4,
    validation_cutoff_season: int | None = None,
    random_state: int = 99,
) -> tuple[EdgeModelBundle | None, dict[str, Any]]:
    from .validated_training import train_edge_model as train

    return train(
        game_features, models_dir,
        validation_seasons=validation_seasons,
        validation_cutoff_season=validation_cutoff_season,
        random_state=random_state,
    )


def load_edge_model(models_dir: Path) -> EdgeModelBundle | None:
    path = models_dir / "edge_xgboost.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return bundle if bundle is not None and bundle.__dict__.get("schema_version") == 6 else None
