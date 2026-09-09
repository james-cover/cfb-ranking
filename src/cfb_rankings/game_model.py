from __future__ import annotations

import json
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

from .evaluation import evaluate_margin_predictions
from .features import MATCHUP_FEATURES, model_adjusted_snapshot
from .storage import atomic_write_csv


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
        scale = max(self.residual_std, 1e-6)
        z = (np.asarray(predicted_margin, dtype=float) + self.residual_mean) / scale
        return np.asarray([0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in z])


PARAMETER_CANDIDATES = [
    {
        "max_depth": 3,
        "min_child_weight": 6,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
    },
    {
        "max_depth": 4,
        "min_child_weight": 6,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
    },
    {
        "max_depth": 5,
        "min_child_weight": 8,
        "learning_rate": 0.025,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_lambda": 6.0,
    },
]


def _new_regressor(parameters: dict[str, Any], random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        n_jobs=-1,
        random_state=random_state,
        **parameters,
    )


def select_and_train_game_model(
    game_features: pd.DataFrame,
    models_dir: Path,
    *,
    validation_seasons: int = 4,
    validation_cutoff_season: int | None = None,
    random_state: int = 42,
) -> tuple[GameModelBundle, pd.DataFrame]:
    if game_features.empty:
        raise ValueError("Game training frame is empty")
    work = game_features.dropna(subset=["home_margin", "game_total", "season"]).copy()
    selection_work = work
    if validation_cutoff_season is not None:
        selection_work = work[
            pd.to_numeric(work["season"], errors="coerce").lt(validation_cutoff_season)
        ]
    seasons = [int(value) for value in sorted(selection_work["season"].astype(int).unique())]
    if len(seasons) < 3:
        raise ValueError("At least three seasons are required for game-model validation")
    folds = seasons[-min(validation_seasons, len(seasons) - 1) :]
    evidence_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    validation_residuals: dict[int, list[float]] = {
        candidate_id: [] for candidate_id in range(1, len(PARAMETER_CANDIDATES) + 1)
    }

    for validation_season in folds:
        validate = selection_work[
            selection_work["season"].astype(int).eq(validation_season)
        ]
        if validate.empty:
            continue
        elo_margin = (
            0.04 * pd.to_numeric(validate["elo_diff"], errors="coerce").fillna(0)
            + 2.5 * pd.to_numeric(validate["home_field"], errors="coerce").fillna(0)
        )
        baseline_rows.append(
            {
                "model": "elo_margin_baseline",
                "validation_season": validation_season,
                **evaluate_margin_predictions(
                    validate["home_margin"].to_numpy(dtype=float), elo_margin.to_numpy()
                ),
            }
        )

    for candidate_id, parameters in enumerate(PARAMETER_CANDIDATES, start=1):
        for validation_season in folds:
            train = selection_work[selection_work["season"].astype(int) < validation_season]
            validate = selection_work[
                selection_work["season"].astype(int).eq(validation_season)
            ]
            if train.empty or validate.empty:
                continue
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(train.reindex(columns=MATCHUP_FEATURES))
            x_validate = imputer.transform(validate.reindex(columns=MATCHUP_FEATURES))
            model = _new_regressor(parameters, random_state)
            model.fit(x_train, train["home_margin"].to_numpy(dtype=float), verbose=False)
            predicted = model.predict(x_validate)
            metrics = evaluate_margin_predictions(
                validate["home_margin"].to_numpy(dtype=float), predicted
            )
            validation_residuals[candidate_id].extend(
                (validate["home_margin"].to_numpy(dtype=float) - predicted).tolist()
            )
            evidence_rows.append(
                {
                    "candidate": candidate_id,
                    "validation_season": validation_season,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    **metrics,
                }
            )

    evidence = pd.DataFrame(evidence_rows)
    if evidence.empty:
        raise ValueError("No valid game-model selection folds were produced")
    summary = (
        evidence.groupby(["candidate", "parameters"], as_index=False)
        .agg(
            folds=("validation_season", "count"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            winner_accuracy=("winner_accuracy", "mean"),
        )
        .sort_values(["mae", "rmse", "winner_accuracy"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    selected_parameters = json.loads(summary.iloc[0]["parameters"])
    selected_candidate = int(summary.iloc[0]["candidate"])
    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(work.reindex(columns=MATCHUP_FEATURES))
    margin_model = _new_regressor(selected_parameters, random_state)
    total_model = _new_regressor(selected_parameters, random_state + 1)
    margin_model.fit(matrix, work["home_margin"].to_numpy(dtype=float), verbose=False)
    total_model.fit(matrix, work["game_total"].to_numpy(dtype=float), verbose=False)
    residuals = np.asarray(validation_residuals[selected_candidate], dtype=float)
    if residuals.size == 0:
        fitted = margin_model.predict(matrix)
        residuals = work["home_margin"].to_numpy(dtype=float) - fitted
    bundle = GameModelBundle(
        margin_model=margin_model,
        total_model=total_model,
        imputer=imputer,
        features=list(MATCHUP_FEATURES),
        residual_mean=float(np.mean(residuals)),
        residual_std=float(np.std(residuals, ddof=1)),
        residual_q10=float(np.quantile(residuals, 0.10)),
        residual_q90=float(np.quantile(residuals, 0.90)),
        selected_parameters=selected_parameters,
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "independent_xgboost.joblib")
    atomic_write_csv(evidence, models_dir / "game_model_fold_evidence.csv")
    xgb_summary = summary.copy()
    xgb_summary.insert(0, "model", "xgboost_candidate_" + xgb_summary["candidate"].astype(str))
    xgb_summary.insert(1, "selected", xgb_summary["candidate"].eq(selected_candidate))
    if baseline_rows:
        baseline = pd.DataFrame(baseline_rows)
        baseline_summary = pd.DataFrame(
            [
                {
                    "model": "elo_margin_baseline",
                    "selected": False,
                    "candidate": 0,
                    "parameters": '{"elo_points_per_point": 25, "home_field_points": 2.5}',
                    "folds": len(baseline),
                    "mae": baseline["mae"].mean(),
                    "rmse": baseline["rmse"].mean(),
                    "winner_accuracy": baseline["winner_accuracy"].mean(),
                }
            ]
        )
        comparison_summary = pd.concat([xgb_summary, baseline_summary], ignore_index=True)
    else:
        comparison_summary = xgb_summary
    comparison_summary = comparison_summary.sort_values(
        ["selected", "mae"], ascending=[False, True]
    ).reset_index(drop=True)
    atomic_write_csv(comparison_summary, models_dir / "game_model_evidence.csv")
    importance = pd.DataFrame(
        {
            "feature": MATCHUP_FEATURES,
            "importance": margin_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    atomic_write_csv(importance, models_dir / "game_model_feature_importance.csv")
    metadata = {
        "algorithm": "XGBoost regression",
        "target": "home_points - away_points",
        "selected_parameters": selected_parameters,
        "residual_mean": bundle.residual_mean,
        "residual_std": bundle.residual_std,
        "validation_seasons": folds,
        "features": MATCHUP_FEATURES,
        "sportsbook_spread_used_as_feature": False,
    }
    (models_dir / "game_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return bundle, comparison_summary


def load_game_model(models_dir: Path) -> GameModelBundle:
    path = models_dir / "independent_xgboost.joblib"
    if not path.exists():
        raise FileNotFoundError("Independent XGBoost model has not been trained yet")
    return joblib.load(path)


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
    "sos_contribution": {"sos_elo_diff"},
    "quality_wins_contribution": {"ranked_wins_diff", "quality_wins_diff"},
    "bad_losses_contribution": {"bad_losses_diff"},
    "recent_form_contribution": {"recent_margin_3_diff"},
}


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
    feature_sums = {team: np.zeros(len(MATCHUP_FEATURES), dtype=float) for team in names}
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
        else np.empty((0, len(MATCHUP_FEATURES) + 1), dtype=float)
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
        by_feature = dict(zip(MATCHUP_FEATURES, average_contributions, strict=True))
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
                "home_spread": consensus_spread,
                "away_spread": -consensus_spread if pd.notna(consensus_spread) else np.nan,
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
    aggregated["market_home_margin"] = -aggregated["consensus_spread"]
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
    output["cover_probability_home"] = bundle.win_probability(
        output["model_edge_home"].fillna(0).to_numpy()
    )
    output.loc[output["market_home_margin"].isna(), "cover_probability_home"] = np.nan
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
        "best_prop_ev_100", "line_fetched_at",
    ]
    return output[keep].sort_values("start_date").reset_index(drop=True)


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
