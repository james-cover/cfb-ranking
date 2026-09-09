from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, ndcg_score


def rank_descending(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def evaluate_ap_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    predicted_points: np.ndarray | None = None,
) -> dict[str, float]:
    work = frame[["poll_group", "actual_ap_rank", "actual_ap_points"]].copy()
    work["score"] = scores
    work["predicted_rank"] = work.groupby("poll_group")["score"].rank(
        method="first", ascending=False
    )
    ndcg_values: list[float] = []
    f1_values: list[float] = []
    rank_errors: list[float] = []
    correlations: list[float] = []
    for _, group in work.groupby("poll_group", sort=False):
        relevance = np.where(group["actual_ap_rank"].to_numpy() <= 25, 26 - group["actual_ap_rank"], 0)
        if relevance.max() > 0:
            ndcg_values.append(
                float(ndcg_score([relevance], [group["score"].to_numpy()], k=25))
            )
        actual_top = (group["actual_ap_rank"] <= 25).astype(int)
        predicted_top = (group["predicted_rank"] <= 25).astype(int)
        f1_values.append(float(f1_score(actual_top, predicted_top, zero_division=0)))
        ranked = group[group["actual_ap_rank"] <= 25]
        if not ranked.empty:
            rank_errors.extend(
                np.abs(ranked["actual_ap_rank"] - ranked["predicted_rank"]).tolist()
            )
            if len(ranked) >= 3:
                correlation = ranked["actual_ap_rank"].corr(
                    ranked["predicted_rank"], method="spearman"
                )
                if not pd.isna(correlation):
                    correlations.append(float(correlation))
    ndcg = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    top25_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    rank_mae = float(np.mean(rank_errors)) if rank_errors else 999.0
    spearman = float(np.mean(correlations)) if correlations else 0.0
    selection_score = 0.55 * ndcg + 0.30 * top25_f1 + 0.15 * max(0.0, 1 - rank_mae / 25)
    points_mae = (
        float(mean_absolute_error(frame["actual_ap_points"], predicted_points))
        if predicted_points is not None
        else float("nan")
    )
    return {
        "ndcg_at_25": ndcg,
        "top25_f1": top25_f1,
        "rank_mae_actual_top25": rank_mae,
        "spearman_actual_top25": spearman,
        "ap_points_mae": points_mae,
        "selection_score": selection_score,
    }


def evaluate_margin_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    non_ties = actual != 0
    winner_accuracy = (
        float(np.mean(np.sign(actual[non_ties]) == np.sign(predicted[non_ties])))
        if non_ties.any()
        else 0.0
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "winner_accuracy": winner_accuracy,
    }


def evaluate_ats(
    actual_home_margin: np.ndarray,
    predicted_home_margin: np.ndarray,
    market_home_margin: np.ndarray,
    thresholds: tuple[int, ...] = (3, 5, 7, 10),
) -> dict[str, float]:
    """ATS accuracy at multiple model-vs-market edge thresholds.

    A bet is placed on the home team when predicted_home_margin > market_home_margin
    by at least `threshold` points, and on the away team when the reverse is true.
    """
    actual = np.asarray(actual_home_margin, dtype=float)
    predicted = np.asarray(predicted_home_margin, dtype=float)
    market = np.asarray(market_home_margin, dtype=float)
    model_edge = predicted - market
    results: dict[str, float] = {}
    for threshold in thresholds:
        has_edge = np.abs(model_edge) >= threshold
        if not has_edge.any():
            results[f"ats_accuracy_{threshold}pt"] = float("nan")
            results[f"ats_games_{threshold}pt"] = 0
            continue
        bet_home = model_edge[has_edge] >= threshold
        actual_edge = actual[has_edge] - market[has_edge]
        covered = np.where(bet_home, actual_edge > 0, actual_edge < 0)
        results[f"ats_accuracy_{threshold}pt"] = float(np.mean(covered))
        results[f"ats_games_{threshold}pt"] = int(has_edge.sum())
    return results
