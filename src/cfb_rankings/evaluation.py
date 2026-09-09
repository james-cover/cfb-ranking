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


def ats_feature_scan(
    game_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """For each feature, compute its correlation with the cover residual
    and ATS accuracy when the feature is above/below median.

    Returns a DataFrame sorted by absolute correlation — the features at
    the top are the ones that actually predict covering the spread.
    """
    work = game_features.dropna(subset=["home_margin", "market_home_margin"]).copy()
    work["cover_residual"] = work["home_margin"] - work["market_home_margin"]
    work["home_covered"] = (work["cover_residual"] > 0).astype(int)

    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        if feature not in work.columns:
            continue
        col = pd.to_numeric(work[feature], errors="coerce")
        valid = col.dropna()
        if len(valid) < 100:
            continue

        corr = float(col.corr(work["cover_residual"]))
        median = float(col.median())

        above = work[col > median]
        below = work[col <= median]
        ats_above = float(above["home_covered"].mean()) if len(above) > 30 else float("nan")
        ats_below = float(below["home_covered"].mean()) if len(below) > 30 else float("nan")

        # Also check top/bottom quartile for stronger effects
        q75 = float(col.quantile(0.75))
        q25 = float(col.quantile(0.25))
        top_q = work[col >= q75]
        bot_q = work[col <= q25]
        ats_top_q = float(top_q["home_covered"].mean()) if len(top_q) > 30 else float("nan")
        ats_bot_q = float(bot_q["home_covered"].mean()) if len(bot_q) > 30 else float("nan")

        rows.append({
            "feature": feature,
            "corr_with_cover": round(corr, 4),
            "abs_corr": round(abs(corr), 4),
            "ats_above_median": round(ats_above, 4) if not np.isnan(ats_above) else None,
            "ats_below_median": round(ats_below, 4) if not np.isnan(ats_below) else None,
            "ats_top_quartile": round(ats_top_q, 4) if not np.isnan(ats_top_q) else None,
            "ats_bottom_quartile": round(ats_bot_q, 4) if not np.isnan(ats_bot_q) else None,
            "n_games": len(valid),
        })

    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
