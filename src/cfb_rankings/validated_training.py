"""Chronological selection, separate probability calibration, and final-season testing.

No sportsbook inputs enter the independent model. The market-aware model learns
actual margin minus the market margin. All decisions here precede final testing.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from .evaluation import evaluate_ats, evaluate_margin_predictions
from .features import MATCHUP_FEATURES
from .storage import atomic_write_csv

COMPACT_FEATURES = [
    "elo_diff", "win_pct_diff", "points_per_game_diff", "points_allowed_per_game_diff",
    "sos_elo_diff", "quality_adj_margin_diff", "recent_margin_3_diff",
    "offense_ppa_diff", "defense_ppa_diff", "offense_success_rate_diff",
    "defense_success_rate_diff", "yards_per_game_diff", "yards_allowed_per_game_diff",
    "turnover_margin_per_game_diff", "home_field", "season_progress",
]


def probability_scores(actual, probability):
    y, p = np.asarray(actual, float), np.asarray(probability, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6)
    if not len(y):
        return {"brier": np.nan, "log_loss": np.nan}
    return {"brier": float(np.mean((y-p)**2)),
            "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}


def fit_probability(actual, prediction):
    y, x = np.asarray(actual, float), np.asarray(prediction, float)
    valid = np.isfinite(y) & np.isfinite(x) & (y != 0)
    y, x = y[valid], x[valid]
    if len(y) < 50 or len(np.unique(y > 0)) < 2 or np.std(x) < 1e-10:
        return (0.0, 0.0)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(x.reshape(-1, 1), (y > 0).astype(int))
    slope = float(model.coef_[0, 0])
    return (float(model.intercept_[0]), slope) if slope > 0 else (0.0, 0.0)


def apply_probability(prediction, calibration):
    intercept, slope = calibration
    z = intercept + slope*np.asarray(prediction, float)
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_edge_shrinkage(actual, prediction):
    y, x = np.asarray(actual, float), np.asarray(prediction, float)
    if len(y) < 50 or np.std(x) < 1e-10:
        return (0.0, 0.0)
    slope = float(np.clip(np.cov(x, y)[0, 1]/np.var(x, ddof=1), 0, 1))
    if slope <= 0:
        return (0.0, 0.0)
    return (float(np.clip(y.mean()-slope*x.mean(), -2, 2)), slope)


def _matrix(frame, features):
    return frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _fit(frame, target, candidates, parameters, seed):
    numeric = _matrix(frame, list(dict.fromkeys(candidates)))
    features = [name for name in numeric if numeric[name].nunique() > 1]
    if not features:
        raise ValueError("No usable, non-constant pregame features. Rebuild features first.")
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(numeric[features])
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_estimators=400,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        n_jobs=4, random_state=seed, **parameters,
    )
    model.fit(x, frame[target].to_numpy(float), verbose=False)
    return model, imputer, features


def _predict(fitted, frame):
    model, imputer, features = fitted
    return np.asarray(model.predict(imputer.transform(_matrix(frame, features))), float)


def _split(work, cutoff, n_folds):
    history = work if cutoff is None else work[work.season < cutoff]
    seasons = sorted(history.season.astype(int).unique())
    if len(seasons) < 5:
        raise ValueError("Need five historical seasons for tuning, calibration, and final testing.")
    calibration, test = int(seasons[-2]), int(seasons[-1])
    folds = [int(y) for y in seasons[1:-2][-n_folds:]]
    return history, folds, calibration, test


def _backtest(frame, prediction, probability):
    columns = [c for c in ("game_id", "season", "week", "start_date", "home_team",
                           "away_team", "home_margin", "market_home_margin") if c in frame]
    result = frame[columns].copy()
    result["prediction"] = prediction
    result["probability_home"] = probability
    return result


def _coverage(work, features, path):
    numeric = _matrix(work, MATCHUP_FEATURES)
    result = pd.DataFrame({"feature": numeric.columns,
                           "non_null_rows": numeric.notna().sum().to_numpy(),
                           "unique_values": numeric.nunique().to_numpy()})
    result["used_by_independent_model"] = result.feature.isin(features)
    atomic_write_csv(result, path / "feature_coverage.csv")


def select_and_train_game_model(game_features, models_dir: Path, *, validation_seasons=4,
                                validation_cutoff_season=None, random_state=42):
    from .game_model import GameModelBundle

    work = game_features.dropna(subset=["home_margin", "game_total", "season"]).copy()
    history, folds, calibration_year, test_year = _split(
        work, validation_cutoff_season, validation_seasons
    )
    candidates = []
    for label, features in (("compact", COMPACT_FEATURES), ("full", MATCHUP_FEATURES)):
        for depth in (2, 4):
            candidates.append((label, features, {"max_depth": depth, "min_child_weight": 12,
                                                "reg_alpha": 0.5, "reg_lambda": 8.0}))
    evidence, summaries = [], []
    for candidate, (label, features, parameters) in enumerate(candidates):
        fold_metrics = []
        for year in folds:
            train, valid = history[history.season < year], history[history.season == year]
            fitted = _fit(train, "home_margin", features, parameters, random_state)
            prediction = _predict(fitted, valid)
            metrics = evaluate_margin_predictions(valid.home_margin.to_numpy(), prediction)
            if "market_home_margin" in valid:
                metrics.update(evaluate_ats(valid.home_margin, prediction, valid.market_home_margin))
            evidence.append({"candidate": candidate, "feature_set": label, "season": year,
                             "scope": "selection_cv", **metrics})
            fold_metrics.append(metrics)
        means = pd.DataFrame(fold_metrics).mean().to_dict()
        summaries.append({"candidate": candidate, "feature_set": label,
                          "parameters": json.dumps(parameters), **means})
    summary = pd.DataFrame(summaries).sort_values(["mae", "rmse", "candidate"])
    selected = int(summary.iloc[0].candidate)
    label, features, parameters = candidates[selected]

    cal = history[history.season == calibration_year]
    cal_fit = _fit(history[history.season < calibration_year], "home_margin", features,
                   parameters, random_state)
    cal_prediction = _predict(cal_fit, cal)
    probability_calibration = fit_probability(cal.home_margin, cal_prediction)
    residuals = cal.home_margin.to_numpy() - cal_prediction

    test = history[history.season == test_year]
    test_fit = _fit(history[history.season < test_year], "home_margin", features,
                    parameters, random_state)
    prediction = _predict(test_fit, test)
    probability = apply_probability(prediction, probability_calibration)
    metrics = evaluate_margin_predictions(test.home_margin.to_numpy(), prediction)
    if "market_home_margin" in test:
        metrics.update(evaluate_ats(test.home_margin, prediction, test.market_home_margin))
    scores = probability_scores((test.home_margin > 0).astype(int), probability)
    selected_row = {"model": f"xgboost_{label}_{parameters['max_depth']}", "selected": True,
                    "scope": "held_out_test", "test_season": test_year,
                    "calibration_season": calibration_year, "candidate": selected,
                    "feature_set": label, "winner_brier": scores["brier"],
                    "winner_log_loss": scores["log_loss"], **metrics}
    elo = 0.04*test.elo_diff.fillna(0) + 2.5*test.home_field.fillna(0)
    baseline = {"model": "elo_margin_baseline", "selected": False,
                "scope": "held_out_test", "test_season": test_year,
                **evaluate_margin_predictions(test.home_margin, elo)}
    rows = [selected_row, baseline]
    if "market_home_margin" in test and test.market_home_margin.notna().any():
        lined = test.dropna(subset=["market_home_margin"])
        lined_mask = test.market_home_margin.notna().to_numpy()
        rows.append({"model": selected_row["model"], "selected": False,
                     "scope": "held_out_test_lined_games", "test_season": test_year,
                     **evaluate_margin_predictions(lined.home_margin, prediction[lined_mask])})
        rows.append({"model": "market_margin_baseline", "selected": False,
                     "scope": "held_out_test_lined_games", "test_season": test_year,
                     **evaluate_margin_predictions(lined.home_margin, lined.market_home_margin)})
    summary["selected"], summary["scope"] = False, "selection_cv"
    result = pd.concat([pd.DataFrame(rows), summary], ignore_index=True)

    model, imputer, selected_features = _fit(work, "home_margin", features, parameters, random_state)
    # Total is a separate target; never reuse a robust-margin loss or select its
    # hyperparameters by claiming the margin validation also validates totals.
    total = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_estimators=400,
                         learning_rate=0.03, max_depth=2, reg_lambda=10, n_jobs=4,
                         random_state=random_state + 1)
    total.fit(imputer.transform(_matrix(work, selected_features)), work.game_total)
    bundle = GameModelBundle(model, total, imputer, selected_features,
                              float(residuals.mean()), float(residuals.std(ddof=1)),
                              float(np.quantile(residuals, .1)), float(np.quantile(residuals, .9)),
                              parameters, probability_calibration=probability_calibration,
                              schema_version=6)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "independent_xgboost.joblib")
    atomic_write_csv(result, models_dir / "game_model_evidence.csv")
    atomic_write_csv(pd.DataFrame(evidence), models_dir / "game_model_fold_evidence.csv")
    atomic_write_csv(_backtest(test, prediction, probability),
                     models_dir / "game_model_backtest_predictions.csv")
    atomic_write_csv(pd.DataFrame({"feature": selected_features,
                                   "importance": model.feature_importances_}).sort_values(
                                       "importance", ascending=False),
                     models_dir / "game_model_feature_importance.csv")
    _coverage(work, selected_features, models_dir)
    (models_dir / "game_model_metadata.json").write_text(json.dumps({
        "schema_version": 6, "feature_set": label, "features": selected_features,
        "selection_seasons": folds, "calibration_season": calibration_year,
        "test_season": test_year, "parameters": parameters,
        "probability_calibration": probability_calibration,
        "test_is_pristine_future_season": False,
        "totals_validated_for_betting": False,
    }, indent=2), encoding="utf-8")
    return bundle, result


def train_edge_model(game_features, models_dir: Path, *, validation_seasons=4,
                     validation_cutoff_season=None, random_state=99):
    from .game_model import EDGE_FEATURES, EdgeModelBundle, _build_edge_features

    def skip(reason):
        models_dir.mkdir(parents=True, exist_ok=True)
        # An old validated bundle must not survive a newly skipped training run.
        joblib.dump(None, models_dir / "edge_xgboost.joblib")
        atomic_write_csv(pd.DataFrame([{"edge_model": reason, "validated_for_live_ev": False}]),
                         models_dir / "edge_model_evidence.csv")
        return None, {"edge_model": reason}

    required = ["home_margin", "market_home_margin", "season"]
    if any(name not in game_features for name in required):
        return skip("skipped_no_lines")
    work = _build_edge_features(game_features.dropna(subset=required).copy())
    if len(work) < 200:
        return skip("skipped_insufficient_data")
    history, folds, calibration_year, test_year = _split(
        work, validation_cutoff_season, validation_seasons
    )
    work["residual"] = work.home_margin - work.market_home_margin
    history = work.loc[history.index]
    parameters = [
        {"max_depth": 2, "min_child_weight": 30, "reg_alpha": 2, "reg_lambda": 20},
        {"max_depth": 3, "min_child_weight": 40, "reg_alpha": 3, "reg_lambda": 30},
    ]
    rows = []
    for candidate, params in enumerate(parameters):
        for year in folds:
            valid = history[history.season == year]
            fitted = _fit(history[history.season < year], "residual", EDGE_FEATURES,
                           params, random_state)
            prediction = _predict(fitted, valid)
            rows.append({"candidate": candidate, "season": year, "scope": "selection_cv",
                         "rmse": float(np.sqrt(np.mean((valid.residual-prediction)**2)))})
    tuning = pd.DataFrame(rows)
    selected = int(tuning.groupby("candidate").rmse.mean().idxmin())
    params = parameters[selected]
    cal = history[history.season == calibration_year]
    cal_fit = _fit(history[history.season < calibration_year], "residual", EDGE_FEATURES,
                   params, random_state)
    cal_prediction = _predict(cal_fit, cal)
    shrinkage = fit_edge_shrinkage(cal.residual, cal_prediction)
    cal_edge = shrinkage[0] + shrinkage[1]*cal_prediction
    probability_calibration = fit_probability(cal.residual, cal_edge)
    test = history[history.season == test_year]
    test_fit = _fit(history[history.season < test_year], "residual", EDGE_FEATURES,
                    params, random_state)
    raw = _predict(test_fit, test)
    edge = shrinkage[0] + shrinkage[1]*raw
    probability = apply_probability(edge, probability_calibration)
    nonpush = test.residual.ne(0).to_numpy()
    scores = probability_scores((test.residual.to_numpy()[nonpush] > 0), probability[nonpush])
    # Fixed guardrails, not optimized thresholds. Passing is evidence for further
    # paper tracking, not a guarantee of profit or a pristine future-season test.
    chosen = np.maximum(probability, 1-probability) >= .55
    settled = chosen & nonpush
    wins = np.where(probability >= .5, test.residual > 0, test.residual < 0)
    count = int(settled.sum())
    win_count = int(wins[settled].sum())
    from .evaluation import wilson_lower
    low = wilson_lower(win_count, count)
    roi = ((win_count*100/110 - (count-win_count))/int(chosen.sum())
           if chosen.any() else float("nan"))
    passed = bool(nonpush.sum() >= 400 and scores["brier"] < .2475
                  and scores["log_loss"] < np.log(2) and count >= 100
                  and low > .5 and roi > 0 and shrinkage[1] >= .05)
    metrics = {"edge_model": "validation_passed" if passed else "trained_not_validated",
               "test_season": test_year, "calibration_season": calibration_year,
               "test_games": int(nonpush.sum()), "brier": scores["brier"],
               "log_loss": scores["log_loss"], "market_brier": .25,
               "probability_bet_games": count, "probability_bet_roi": roi,
               "probability_bet_wilson_low": low, "shrinkage_slope": shrinkage[1],
               "validated_for_live_ev": passed,
               **evaluate_ats(test.home_margin, test.market_home_margin+edge,
                              test.market_home_margin)}
    model, imputer, features = _fit(work, "residual", EDGE_FEATURES, params, random_state)
    bundle = EdgeModelBundle(model, imputer, features,
                              float(np.std(cal.residual-cal_edge, ddof=1)),
                              shrinkage=shrinkage, probability_calibration=probability_calibration,
                              validated=passed, schema_version=6)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / "edge_xgboost.joblib")
    atomic_write_csv(pd.DataFrame([metrics]), models_dir / "edge_model_evidence.csv")
    atomic_write_csv(tuning, models_dir / "edge_model_fold_evidence.csv")
    backtest = _backtest(test, edge, probability)
    backtest["raw_edge"], backtest["push"] = raw, ~nonpush
    atomic_write_csv(backtest, models_dir / "edge_model_backtest_predictions.csv")
    atomic_write_csv(pd.DataFrame({"feature": features, "importance": model.feature_importances_})
                     .sort_values("importance", ascending=False),
                     models_dir / "edge_model_feature_importance.csv")
    return bundle, metrics
