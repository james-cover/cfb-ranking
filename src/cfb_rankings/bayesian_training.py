"""Season-forward selection of opponent-adjusted Bayesian and XGBoost models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .evaluation import evaluate_ats, evaluate_margin_predictions
from .storage import atomic_write_csv
from .validated_training import apply_probability, fit_probability, probability_scores

OPPONENT_ADJUSTED_FEATURES = [
    "bayes_points_off_rating_diff", "bayes_points_def_rating_diff",
    "bayes_rushing_off_rating_diff", "bayes_rushing_def_rating_diff",
    "bayes_passing_off_rating_diff", "bayes_passing_def_rating_diff",
    "opp_adj_points_off_diff", "opp_adj_points_def_diff",
    "opp_adj_rushing_off_diff", "opp_adj_rushing_def_diff",
    "opp_adj_passing_off_diff", "opp_adj_passing_def_diff",
    "turnover_margin_per_game_diff", "possession_time_pg_diff",
    "sos_elo_diff", "home_field",
]


def _regressor(family: str, seed: int):
    if family == "bayesian_ridge":
        return BayesianRidge(compute_score=True)
    return XGBRegressor(
        objective="reg:squarederror", n_estimators=500, learning_rate=0.03,
        max_depth=3, min_child_weight=15, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=10.0, reg_alpha=0.2, random_state=seed, n_jobs=1,
    )


def _fit(frame, features, target, family, seed):
    numeric = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    usable = [name for name in numeric if numeric[name].nunique() > 1]
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(numeric[usable]))
    model = _regressor(family, seed)
    model.fit(matrix, frame[target].to_numpy(float))
    return model, imputer, scaler, usable


def _predict(fitted, frame):
    model, imputer, scaler, features = fitted
    numeric = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return np.asarray(model.predict(scaler.transform(imputer.transform(numeric))), float)


def select_and_train_bayesian_model(
    game_features: pd.DataFrame, models_dir: Path, *, validation_seasons: int = 4,
    validation_cutoff_season: int | None = None, random_state: int = 42,
):
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
        (family, label, features)
        for family in ("bayesian_ridge", "xgboost")
        for label, features in (
            ("bayesian_expectation_residuals", OPPONENT_ADJUSTED_FEATURES),
            ("bayesian_expectation_residuals_elo", ["elo_diff", *OPPONENT_ADJUSTED_FEATURES]),
        )
    ]

    fold_rows, summaries = [], []
    for candidate, (family, label, features) in enumerate(candidates):
        rows = []
        for year in folds:
            train, valid = history[history.season < year], history[history.season == year]
            prediction = _predict(
                _fit(train, features, "home_margin", family, random_state), valid
            )
            metrics = evaluate_margin_predictions(valid.home_margin, prediction)
            if "market_home_margin" in valid:
                metrics.update(evaluate_ats(
                    valid.home_margin, prediction, valid.market_home_margin
                ))
            fold_rows.append({"candidate": candidate, "family": family, "model": label,
                              "season": int(year), "scope": "selection_cv", **metrics})
            rows.append(metrics)
        means = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
        summaries.append({"candidate": candidate, "family": family, "model": label,
                          "feature_count": len(features), **means})

    summary = pd.DataFrame(summaries).sort_values(["mae", "feature_count", "candidate"])
    best_mae = float(summary.iloc[0].mae)
    bayesian_near_best = summary[
        summary.family.eq("bayesian_ridge") & summary.mae.le(best_mae + 0.05)
    ]
    winner = (
        bayesian_near_best.sort_values("mae").iloc[0]
        if not bayesian_near_best.empty
        else summary.iloc[0]
    )
    selected = int(winner.candidate)
    family, label, selected_features = candidates[selected]

    calibration = history[history.season == calibration_year]
    calibration_prediction = _predict(_fit(
        history[history.season < calibration_year], selected_features,
        "home_margin", family, random_state,
    ), calibration)
    probability_fit = fit_probability(calibration.home_margin, calibration_prediction)
    residuals = calibration.home_margin.to_numpy(float) - calibration_prediction

    test = history[history.season == test_year]
    test_prediction = _predict(_fit(
        history[history.season < test_year], selected_features,
        "home_margin", family, random_state,
    ), test)
    probability = apply_probability(test_prediction, probability_fit)
    metrics = evaluate_margin_predictions(test.home_margin, test_prediction)
    if "market_home_margin" in test:
        metrics.update(evaluate_ats(test.home_margin, test_prediction, test.market_home_margin))
    scores = probability_scores((test.home_margin > 0).astype(int), probability)
    selected_row = {
        "model": label, "family": family, "selected": True, "scope": "held_out_test",
        "test_season": int(test_year), "calibration_season": int(calibration_year),
        "winner_brier": scores["brier"], "winner_log_loss": scores["log_loss"], **metrics,
    }
    summary["selected"], summary["scope"] = summary.candidate.eq(selected), "selection_cv"
    evidence = pd.concat([pd.DataFrame([selected_row]), summary], ignore_index=True)

    margin_model, imputer, scaler, features = _fit(
        work, selected_features, "home_margin", family, random_state
    )
    total_matrix = scaler.transform(imputer.transform(work.reindex(columns=features)))
    total_model = _regressor(family, random_state)
    total_model.fit(total_matrix, work.game_total.to_numpy(float))
    bundle = GameModelBundle(
        margin_model, total_model, imputer, features,
        float(residuals.mean()), float(residuals.std(ddof=1)),
        float(np.quantile(residuals, 0.1)), float(np.quantile(residuals, 0.9)),
        {"family": family, "feature_set": label, "selection_tolerance_mae": 0.05},
        probability_calibration=probability_fit, schema_version=9,
        scaler=scaler, model_family=family,
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "independent_model.joblib")
    atomic_write_csv(evidence, models_dir / "game_model_evidence.csv")
    atomic_write_csv(pd.DataFrame(fold_rows), models_dir / "game_model_fold_evidence.csv")

    backtest_columns = [name for name in (
        "game_id", "season", "week", "start_date", "home_team", "away_team",
        "home_margin", "market_home_margin",
    ) if name in test]
    backtest = test[backtest_columns].copy()
    backtest["prediction"], backtest["probability_home"] = test_prediction, probability
    atomic_write_csv(backtest, models_dir / "game_model_backtest_predictions.csv")

    if family == "bayesian_ridge":
        importance = np.abs(margin_model.coef_)
        signed = margin_model.coef_
    else:
        importance = margin_model.feature_importances_
        signed = np.full(len(features), np.nan)
    coefficients = pd.DataFrame({
        "feature": features, "coefficient_points_per_standard_deviation": signed,
        "importance": importance,
    })
    coefficients = coefficients.sort_values("importance", ascending=False)
    atomic_write_csv(coefficients, models_dir / "game_model_feature_importance.csv")
    (models_dir / "game_model_metadata.json").write_text(json.dumps({
        "schema_version": 9, "family": family, "model": label, "features": features,
        "selection_seasons": [int(year) for year in folds],
        "calibration_season": int(calibration_year), "test_season": int(test_year),
        "selection_tolerance_mae": 0.05,
    }, indent=2), encoding="utf-8")
    return bundle, evidence
