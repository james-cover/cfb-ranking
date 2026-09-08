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
from xgboost import XGBRegressor

from .evaluation import evaluate_margin_predictions
from .features import MATCHUP_FEATURES
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
    row = {}
    for name in [feature.removesuffix("_diff") for feature in MATCHUP_FEATURES if feature.endswith("_diff")]:
        row[f"{name}_diff"] = float(team_a.get(name, 0.0)) - float(team_b.get(name, 0.0))
    row.update({"neutral_site": 1.0, "home_field": 0.0, "season_progress": 1.0})
    return pd.DataFrame([{name: row.get(name, 0.0) for name in MATCHUP_FEATURES}])


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
    for pair_index, (team_a, team_b) in enumerate(pair_names):
        forward = float(predicted_margins[2 * pair_index])
        reverse = float(predicted_margins[2 * pair_index + 1])
        antisymmetric_margin = (forward - reverse) / 2.0
        margin_sums[team_a] += antisymmetric_margin
        margin_sums[team_b] -= antisymmetric_margin
        comparisons[team_a] += 1
        comparisons[team_b] += 1
    rows = []
    for team in names:
        state = work.loc[team]
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
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["model_rating", "elo"], ascending=[False, False]
    ).reset_index(drop=True)
    output.insert(0, "independent_rank", np.arange(1, len(output) + 1))
    return output


def consensus_current_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["game_id", "market_home_margin", "consensus_spread", "line_fetched_at", "sportsbooks"])
    work = lines.copy()
    work["spread"] = pd.to_numeric(work["spread"], errors="coerce")
    work["fetched_ts"] = pd.to_datetime(work["fetched_at"], errors="coerce", utc=True)
    work = work.dropna(subset=["game_id", "spread", "fetched_ts"])
    # Keep each provider's most recently observed line, then take the median.
    work = work.sort_values("fetched_ts").drop_duplicates(["game_id", "provider"], keep="last")
    aggregated = work.groupby("game_id", as_index=False).agg(
        consensus_spread=("spread", "median"),
        line_fetched_at=("fetched_ts", "max"),
        sportsbooks=("provider", lambda values: ", ".join(sorted(set(map(str, values))))),
    )
    # CFBD's spread is the home-team handicap, so -3 implies a home margin of +3.
    aggregated["market_home_margin"] = -aggregated["consensus_spread"]
    return aggregated


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
    output["model_edge_home"] = output["model_home_margin"] - output["market_home_margin"]
    output["cover_probability_home"] = bundle.win_probability(
        output["model_edge_home"].fillna(0).to_numpy()
    )
    output.loc[output["market_home_margin"].isna(), "cover_probability_home"] = np.nan
    output["model_ats_lean"] = np.where(
        output["model_edge_home"].isna(),
        "No line",
        np.where(output["model_edge_home"] >= 0, output["home_team"], output["away_team"]),
    )
    keep = [
        "game_id", "start_date", "week", "away_team", "home_team", "neutral_site",
        "predicted_away_score", "predicted_home_score", "model_home_margin",
        "margin_low_80", "margin_high_80", "home_win_probability", "consensus_spread",
        "market_home_margin", "model_edge_home", "cover_probability_home", "model_ats_lean",
        "sportsbooks", "line_fetched_at",
    ]
    return output[keep].sort_values("start_date").reset_index(drop=True)
