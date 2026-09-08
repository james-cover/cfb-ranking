from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRanker, XGBRegressor

from .evaluation import evaluate_ap_predictions
from .features import AP_FEATURES
from .storage import atomic_write_csv

LOGGER = logging.getLogger(__name__)


@dataclass
class APModelBundle:
    name: str
    kind: str
    estimator: Any
    imputer: SimpleImputer
    points_calibrator: IsotonicRegression
    features: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.imputer.transform(frame.reindex(columns=self.features))
        return np.asarray(self.estimator.predict(matrix), dtype=float)

    def predict_points(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.points_calibrator.predict(self.predict(frame)), dtype=float)


class PreviousPollBaseline:
    def fit(self, x: np.ndarray, y: np.ndarray, **_: Any) -> PreviousPollBaseline:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        # previous points, rank, and "previously ranked" are columns 1, 0, and 3.
        return x[:, 1] + (40.0 - x[:, 0]) * 0.01 + x[:, 3] * 0.001


def _candidate_factories(random_state: int = 42):
    factories: dict[str, tuple[str, Any]] = {
        "previous_poll_baseline": ("regression", lambda: PreviousPollBaseline()),
        "xgboost_lambdamart": (
            "xgb_ranker",
            lambda: XGBRanker(
                objective="rank:ndcg",
                eval_metric="ndcg@25",
                n_estimators=500,
                learning_rate=0.035,
                max_depth=5,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "xgboost_points_regression": (
            "regression",
            lambda: XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.035,
                max_depth=5,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting_points": (
            "regression",
            lambda: HistGradientBoostingRegressor(
                learning_rate=0.055,
                max_iter=350,
                max_leaf_nodes=24,
                l2_regularization=2.0,
                random_state=random_state,
            ),
        ),
    }
    try:
        from lightgbm import LGBMRanker

        factories["lightgbm_lambdarank"] = (
            "lgbm_ranker",
            lambda: LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=24,
                max_depth=6,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        )
    except ImportError:
        LOGGER.warning("LightGBM is unavailable; its AP candidate will be skipped")
    return factories


def _sorted_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    ordered = frame.sort_values(["poll_group", "team"]).reset_index(drop=True)
    codes, _ = pd.factorize(ordered["poll_group"], sort=False)
    group_sizes = ordered.groupby("poll_group", sort=False).size().tolist()
    return ordered, codes.astype(np.int32), group_sizes


def _fit_candidate(
    name: str,
    kind: str,
    factory: Any,
    train: pd.DataFrame,
) -> APModelBundle:
    ordered, qid, group_sizes = _sorted_groups(train)
    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(ordered.reindex(columns=AP_FEATURES))
    relevance = np.where(
        ordered["actual_ap_rank"].to_numpy() <= 25,
        26 - ordered["actual_ap_rank"].to_numpy(),
        0,
    ).astype(int)
    estimator = factory()
    if kind == "xgb_ranker":
        estimator.fit(matrix, relevance, qid=qid, verbose=False)
    elif kind == "lgbm_ranker":
        estimator.fit(matrix, relevance, group=group_sizes)
    else:
        estimator.fit(matrix, ordered["actual_ap_points"].to_numpy(dtype=float))
    training_scores = np.asarray(estimator.predict(matrix), dtype=float)
    points_calibrator = IsotonicRegression(out_of_bounds="clip")
    points_calibrator.fit(
        training_scores, ordered["actual_ap_points"].to_numpy(dtype=float)
    )
    return APModelBundle(
        name, kind, estimator, imputer, points_calibrator, list(AP_FEATURES)
    )


def select_and_train_ap_model(
    training_frame: pd.DataFrame,
    models_dir: Path,
    *,
    validation_seasons: int = 5,
    validation_cutoff_season: int | None = None,
    random_state: int = 42,
) -> tuple[APModelBundle, pd.DataFrame]:
    if training_frame.empty:
        raise ValueError("AP training frame is empty")
    selection_frame = training_frame
    if validation_cutoff_season is not None:
        selection_frame = training_frame[
            pd.to_numeric(training_frame["season"], errors="coerce").lt(
                validation_cutoff_season
            )
        ]
    seasons = [
        int(value)
        for value in sorted(selection_frame["season"].dropna().astype(int).unique())
    ]
    if len(seasons) < 3:
        raise ValueError("At least three seasons are required for chronological AP validation")
    folds = seasons[-min(validation_seasons, len(seasons) - 1) :]
    factories = _candidate_factories(random_state)
    evidence_rows: list[dict[str, Any]] = []

    for validation_season in folds:
        train = selection_frame[selection_frame["season"].astype(int) < validation_season]
        validate = selection_frame[selection_frame["season"].astype(int).eq(validation_season)]
        if train.empty or validate.empty:
            continue
        for name, (kind, factory) in factories.items():
            bundle = _fit_candidate(name, kind, factory, train)
            scores = bundle.predict(validate)
            metrics = evaluate_ap_predictions(
                validate, scores, bundle.predict_points(validate)
            )
            evidence_rows.append(
                {"model": name, "validation_season": validation_season, **metrics}
            )
            LOGGER.info(
                "AP validation %s on %s: NDCG@25 %.4f",
                name,
                validation_season,
                metrics["ndcg_at_25"],
            )

    evidence = pd.DataFrame(evidence_rows)
    if evidence.empty:
        raise ValueError("No valid AP model-selection folds were produced")
    summary = (
        evidence.groupby("model", as_index=False)
        .agg(
            folds=("validation_season", "count"),
            ndcg_at_25=("ndcg_at_25", "mean"),
            top25_f1=("top25_f1", "mean"),
            rank_mae_actual_top25=("rank_mae_actual_top25", "mean"),
            spearman_actual_top25=("spearman_actual_top25", "mean"),
            ap_points_mae=("ap_points_mae", "mean"),
            selection_score=("selection_score", "mean"),
        )
        .sort_values(
            ["selection_score", "ndcg_at_25", "rank_mae_actual_top25"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    champion_name = str(summary.iloc[0]["model"])
    champion_kind, champion_factory = factories[champion_name]
    champion = _fit_candidate(
        champion_name, champion_kind, champion_factory, training_frame
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, models_dir / "ap_champion.joblib")
    atomic_write_csv(evidence, models_dir / "ap_model_fold_evidence.csv")
    atomic_write_csv(summary, models_dir / "ap_model_evidence.csv")
    metadata = {
        "champion": champion_name,
        "selection_rule": (
            "0.55*NDCG@25 + 0.30*top25_F1 + 0.15*(1-clipped_rank_MAE/25)"
        ),
        "validation_seasons": folds,
        "features": AP_FEATURES,
        "summary": summary.to_dict(orient="records"),
    }
    (models_dir / "ap_model_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    return champion, summary


def load_ap_model(models_dir: Path) -> APModelBundle:
    path = models_dir / "ap_champion.joblib"
    if not path.exists():
        raise FileNotFoundError("AP model has not been trained yet")
    return joblib.load(path)


def predict_ap_poll(bundle: APModelBundle, frame: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    output = frame.copy()
    output["ap_prediction_score"] = bundle.predict(output)
    output["predicted_ap_points"] = bundle.predict_points(output)
    output = output.sort_values(
        ["ap_prediction_score", "previous_ap_points", "elo"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    output["predicted_ap_rank"] = np.arange(1, len(output) + 1)
    columns = [
        "predicted_ap_rank",
        "team",
        "ap_prediction_score",
        "predicted_ap_points",
        "previous_ap_rank",
        "previous_ap_points",
        "wins",
        "losses",
        "elo",
        "target_poll_week",
    ]
    return output.loc[: top_n - 1, columns]
