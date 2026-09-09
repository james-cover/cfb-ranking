"""Interpretable Bayesian margin model using basic opponent-context statistics."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_ats, evaluate_margin_predictions
from .storage import atomic_write_csv
from .validated_training import apply_probability, fit_probability, probability_scores

BASIC_FEATURES = [
    "points_per_game_diff",
    "points_allowed_per_game_diff",
    "rushing_ypg_diff",
    "rushing_ypg_allowed_diff",
    "passing_ypg_diff",
    "passing_ypg_allowed_diff",
    "turnover_margin_per_game_diff",
    "possession_time_pg_diff",
    "sos_elo_diff",
    "home_field",
]


def _fit(frame: pd.DataFrame, features: list[str], target: str):
    numeric = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    usable = [name for name in numeric if numeric[name].nunique() > 1]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(numeric[usable]))
    model = BayesianRidge(compute_score=True)
    model.fit(matrix, frame[target].to_numpy(float))
    return model, imputer, scaler, usable


def _predict(fitted, frame: pd.DataFrame) -> np.ndarray:
    model, imputer, scaler, features = fitted
    numeric = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return np.asarray(model.predict(scaler.transform(imputer.transform(numeric))), float)


def select_and_train_bayesian_model(
    game_features: pd.DataFrame,
    models_dir: Path,
    *,
    validation_seasons: int = 4,
    validation_cutoff_season: int | None = None,
    random_state: int = 42,
):
    del random_state  # BayesianRidge is deterministic.
    from .game_model import GameModelBundle

    work = game_features.dropna(subset=["home_margin", "game_total", "season"]).copy()
    history = work if validation_cutoff_season is None else work[
        work.season < validation_cutoff_season
    ]
    seasons = sorted(history.season.astype(int).unique())
    if len(seasons) < 5:
        raise ValueError("Need five historical seasons for selection, calibration, and testing.")
    calibration_year, test_year = seasons[-2], seasons[-1]
    folds = seasons[1:-2][-validation_seasons:]
    candidates = [
        ("basic_stats_sos", BASIC_FEATURES),
        ("basic_stats_sos_elo", ["elo_diff", *BASIC_FEATURES]),
    ]
    fold_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for candidate, (name, features) in enumerate(candidates):
        metrics_for_candidate = []
        for year in folds:
            train, valid = history[history.season < year], history[history.season == year]
            prediction = _predict(_fit(train, features, "home_margin"), valid)
            metrics = evaluate_margin_predictions(valid.home_margin, prediction)
            if "market_home_margin" in valid:
                metrics.update(evaluate_ats(valid.home_margin, prediction, valid.market_home_margin))
            fold_rows.append({"candidate": candidate, "model": name, "season": year,
                              "scope": "selection_cv", **metrics})
            metrics_for_candidate.append(metrics)
        means = pd.DataFrame(metrics_for_candidate).mean(numeric_only=True).to_dict()
        summaries.append({"candidate": candidate, "model": name,
                          "feature_count": len(features), **means})
    summary = pd.DataFrame(summaries).sort_values(["mae", "feature_count", "candidate"])
    best_mae = float(summary.iloc[0].mae)
    # Differences smaller than 0.05 points are operationally meaningless; prefer
    # the more transparent model inside that predeclared tolerance.
    eligible = summary[summary.mae <= best_mae + 0.05]
    selected = int(eligible.sort_values(["feature_count", "mae"]).iloc[0].candidate)
    selected_name, selected_candidates = candidates[selected]

    cal = history[history.season == calibration_year]
    cal_prediction = _predict(
        _fit(history[history.season < calibration_year], selected_candidates, "home_margin"), cal
    )
    calibration = fit_probability(cal.home_margin, cal_prediction)
    residuals = cal.home_margin.to_numpy(float) - cal_prediction

    test = history[history.season == test_year]
    test_prediction = _predict(
        _fit(history[history.season < test_year], selected_candidates, "home_margin"), test
    )
    probability = apply_probability(test_prediction, calibration)
    metrics = evaluate_margin_predictions(test.home_margin, test_prediction)
    if "market_home_margin" in test:
        metrics.update(evaluate_ats(test.home_margin, test_prediction, test.market_home_margin))
    scores = probability_scores((test.home_margin > 0).astype(int), probability)
    selected_row = {
        "model": selected_name, "selected": True, "scope": "held_out_test",
        "test_season": test_year, "calibration_season": calibration_year,
        "winner_brier": scores["brier"], "winner_log_loss": scores["log_loss"], **metrics,
    }
    summary["selected"] = summary.candidate.eq(selected)
    summary["scope"] = "selection_cv"
    evidence = pd.concat([pd.DataFrame([selected_row]), summary], ignore_index=True)

    margin_model, imputer, scaler, features = _fit(work, selected_candidates, "home_margin")
    total_numeric = scaler.transform(imputer.transform(work.reindex(columns=features)))
    total_model = BayesianRidge(compute_score=True).fit(total_numeric, work.game_total)
    bundle = GameModelBundle(
        margin_model, total_model, imputer, features,
        float(residuals.mean()), float(residuals.std(ddof=1)),
        float(np.quantile(residuals, .1)), float(np.quantile(residuals, .9)),
        {"family": "bayesian_ridge", "selection_tolerance_mae": 0.05},
        probability_calibration=calibration, schema_version=7, scaler=scaler,
        model_family="bayesian_ridge",
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "independent_bayesian.joblib")
    atomic_write_csv(evidence, models_dir / "game_model_evidence.csv")
    atomic_write_csv(pd.DataFrame(fold_rows), models_dir / "game_model_fold_evidence.csv")
    backtest_columns = [name for name in (
        "game_id", "season", "week", "start_date", "home_team", "away_team",
        "home_margin", "market_home_margin",
    ) if name in test]
    backtest = test[backtest_columns].copy()
    backtest["prediction"] = test_prediction
    backtest["probability_home"] = probability
    atomic_write_csv(backtest, models_dir / "game_model_backtest_predictions.csv")
    coefficients = pd.DataFrame({
        "feature": features,
        "coefficient_points_per_standard_deviation": margin_model.coef_,
        "absolute_coefficient": np.abs(margin_model.coef_),
    }).sort_values("absolute_coefficient", ascending=False)
    atomic_write_csv(coefficients, models_dir / "game_model_feature_importance.csv")
    (models_dir / "game_model_metadata.json").write_text(json.dumps({
        "schema_version": 7, "family": "bayesian_ridge", "model": selected_name,
        "features": features, "selection_seasons": [int(year) for year in folds],
        "calibration_season": int(calibration_year), "test_season": int(test_year),
        "selection_tolerance_mae": 0.05,
    }, indent=2), encoding="utf-8")
    return bundle, evidence
