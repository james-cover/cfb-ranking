from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .ap_model import load_ap_model, predict_ap_poll, select_and_train_ap_model
from .config import Settings
from .features import (
    ap_poll_rows,
    build_ap_training_frame,
    build_sequential_features,
    current_ap_feature_frame,
    fbs_team_names,
)
from .game_model import (
    apply_edge_predictions,
    build_independent_rankings,
    build_season_schedule,
    load_edge_model,
    load_game_model,
    predict_upcoming_games,
    select_and_train_game_model,
    train_edge_model,
)
from .logos import BRANDING_COLUMNS, build_team_branding
from .storage import atomic_write_csv, read_csv, upsert_csv

RAW_FILES = {
    "games": "games.csv",
    "rankings": "rankings.csv",
    "teams": "teams.csv",
    "team_game_stats": "team_game_stats.csv",
    "advanced_game_stats": "advanced_game_stats.csv",
    "lines": "betting_lines.csv",
    "media": "media.csv",
}

REQUIRED_BOX_STATS = {
    "rushingYards": "rushing yards",
    "netPassingYards": "passing yards",
    "possessionTime": "time of possession",
}


def box_score_coverage(frame: pd.DataFrame) -> dict[str, int]:
    """Count usable team-game observations for required, human-readable stats."""
    counts = {name: 0 for name in REQUIRED_BOX_STATS}
    if frame.empty or "stats_json" not in frame:
        return counts
    for value in frame["stats_json"].dropna():
        try:
            stats = json.loads(value) if isinstance(value, str) else {}
        except (json.JSONDecodeError, TypeError):
            continue
        for name in counts:
            if stats.get(name) not in (None, ""):
                counts[name] += 1
    return counts


def require_box_scores(frame: pd.DataFrame) -> dict[str, int]:
    counts = box_score_coverage(frame)
    missing = [label for name, label in REQUIRED_BOX_STATS.items() if counts[name] == 0]
    if missing:
        raise ValueError(
            "Team box-score download is incomplete: no " + ", ".join(missing)
            + ". Run cfb bootstrap again with the repaired downloader; do not train yet."
        )
    return counts


def load_raw(settings: Settings) -> dict[str, pd.DataFrame]:
    return {name: read_csv(settings.raw_dir / filename) for name, filename in RAW_FILES.items()}


def build_features(settings: Settings) -> dict[str, int]:
    raw = load_raw(settings)
    if raw["games"].empty:
        raise FileNotFoundError("data/raw/games.csv is missing; run bootstrap first")
    box_counts = require_box_scores(raw["team_game_stats"])
    game_features, team_week_features = build_sequential_features(
        raw["games"],
        raw["rankings"],
        raw["teams"],
        raw["team_game_stats"],
        raw["advanced_game_stats"],
    )
    ap_training = build_ap_training_frame(
        team_week_features, raw["rankings"], raw["teams"]
    )
    atomic_write_csv(game_features, settings.processed_dir / "game_training_data.csv")
    atomic_write_csv(team_week_features, settings.processed_dir / "team_week_features.csv")
    atomic_write_csv(ap_training, settings.processed_dir / "ap_training_data.csv")
    return {
        "game_training_rows": len(game_features),
        "team_week_rows": len(team_week_features),
        "ap_training_rows": len(ap_training),
        "rushing_stat_rows": box_counts["rushingYards"],
        "passing_stat_rows": box_counts["netPassingYards"],
        "possession_stat_rows": box_counts["possessionTime"],
    }


def train_models(settings: Settings) -> dict[str, object]:
    ap_training = read_csv(settings.processed_dir / "ap_training_data.csv")
    game_training = read_csv(settings.processed_dir / "game_training_data.csv")
    if "feature_schema_version" not in game_training or not game_training["feature_schema_version"].eq(6).all():
        raise ValueError("Pregame features are outdated. Run cfb build-features before cfb train.")
    # Join consensus market lines onto the training data so the validation loop
    # can compute ATS accuracy alongside standard margin metrics.
    raw_lines = read_csv(settings.raw_dir / "betting_lines.csv")
    if not raw_lines.empty and not game_training.empty:
        from .game_model import consensus_current_lines
        lines_lookup = consensus_current_lines(raw_lines)[["game_id", "market_home_margin"]]
        game_training = game_training.merge(lines_lookup, on="game_id", how="left")
    ap_bundle, ap_evidence = select_and_train_ap_model(
        ap_training,
        settings.models_dir,
        validation_cutoff_season=settings.season,
    )
    _game_bundle, game_evidence = select_and_train_game_model(
        game_training,
        settings.models_dir,
        validation_cutoff_season=settings.season,
    )
    # Train the edge model (market-aware, predicts cover residual).
    _edge_bundle, edge_output = train_edge_model(
        game_training,
        settings.models_dir,
        validation_cutoff_season=settings.season,
    )
    selected_row = game_evidence.iloc[0]
    ats_keys = [c for c in game_evidence.columns if c.startswith("ats_")]
    ats_output = {k: (float(selected_row[k]) if not pd.isna(selected_row[k]) else None) for k in ats_keys}
    return {
        "ap_champion": ap_bundle.name,
        "ap_selection_score": float(ap_evidence.iloc[0]["selection_score"]),
        "game_validation_mae": float(selected_row["mae"]),
        "game_winner_accuracy": float(selected_row["winner_accuracy"]),
        "raw_model_ats": ats_output,
        "edge_model": edge_output,
    }


def _current_actual_ap(rankings: pd.DataFrame, season: int) -> pd.DataFrame:
    ap = ap_poll_rows(rankings)
    ap = ap[pd.to_numeric(ap["season"], errors="coerce").eq(season)].copy()
    if ap.empty:
        return pd.DataFrame(columns=["actual_ap_rank", "team", "actual_ap_points"])
    latest = int(pd.to_numeric(ap["poll_week"], errors="coerce").max())
    output = ap[pd.to_numeric(ap["poll_week"], errors="coerce").eq(latest)].copy()
    output = output.rename(
        columns={"rank": "actual_ap_rank", "points": "actual_ap_points"}
    )
    return output[["actual_ap_rank", "team", "actual_ap_points", "poll_week"]].sort_values(
        "actual_ap_rank"
    )


def _tv_outlet_lookup(media: pd.DataFrame) -> pd.DataFrame:
    """Build a game_id → primary TV outlet mapping from raw media data."""
    if media.empty or "game_id" not in media.columns:
        return pd.DataFrame(columns=["game_id", "tv_outlet"])
    tv = media.copy()
    if "media_type" in tv.columns:
        # Prefer TV rows; fall back to web/other if no TV row exists.
        tv["priority"] = tv["media_type"].fillna("").str.lower().map(
            {"tv": 0, "web": 1, "ppv": 2}
        ).fillna(3).astype(int)
    else:
        tv["priority"] = 0
    tv = (
        tv.dropna(subset=["game_id"])
        .sort_values("priority")
        .drop_duplicates("game_id", keep="first")
    )
    return tv[["game_id", "outlet"]].rename(columns={"outlet": "tv_outlet"})


def generate_predictions(settings: Settings) -> dict[str, int]:
    raw = load_raw(settings)
    team_week = read_csv(settings.processed_dir / "team_week_features.csv")
    if team_week.empty:
        raise FileNotFoundError("Processed features are missing; run build-features first")
    if "feature_schema_version" not in team_week or not team_week["feature_schema_version"].eq(6).all():
        raise ValueError("Team features are outdated. Run cfb build-features, then cfb train.")
    ap_bundle = load_ap_model(settings.models_dir)
    game_bundle = load_game_model(settings.models_dir)
    current_ap_features, target_week = current_ap_feature_frame(
        team_week, raw["rankings"], settings.season, raw["teams"]
    )
    predicted_ap = predict_ap_poll(ap_bundle, current_ap_features)

    season_current = current_ap_features.drop_duplicates("team")
    fbs = fbs_team_names(raw["games"], raw["teams"], settings.season)
    independent = build_independent_rankings(game_bundle, season_current, fbs or None)
    all_upcoming = predict_upcoming_games(
        game_bundle,
        raw["games"],
        season_current,
        raw["lines"],
        settings.season,
        next_slate_only=False,
    )
    edge_bundle = load_edge_model(settings.models_dir)
    all_upcoming = apply_edge_predictions(all_upcoming, edge_bundle)
    # Attach broadcast outlet to upcoming games and season schedule.
    tv_lookup = _tv_outlet_lookup(raw.get("media", pd.DataFrame()))
    if not tv_lookup.empty and not all_upcoming.empty:
        all_upcoming = all_upcoming.merge(tv_lookup, on="game_id", how="left")
    if all_upcoming.empty:
        upcoming = all_upcoming.copy()
    else:
        next_week = int(pd.to_numeric(all_upcoming["week"], errors="coerce").min())
        upcoming = all_upcoming[
            pd.to_numeric(all_upcoming["week"], errors="coerce").eq(next_week)
        ].copy()
    season_schedule = build_season_schedule(
        raw["games"], raw["lines"], all_upcoming, settings.season, fbs or None
    )
    if not tv_lookup.empty and not season_schedule.empty:
        season_schedule = season_schedule.merge(tv_lookup, on="game_id", how="left")
    branding = build_team_branding(
        raw["teams"], settings.season, settings.team_logos_dir
    )
    actual = _current_actual_ap(raw["rankings"], settings.season)
    generated_at = datetime.now(UTC).isoformat()
    latest_actual_week = (
        int(pd.to_numeric(actual["poll_week"], errors="coerce").max())
        if not actual.empty
        else 0
    )
    prediction_mode = (
        "pre_release_forecast"
        if target_week > latest_actual_week
        else "nowcast_after_release"
    )
    predicted_ap["generated_at"] = generated_at
    predicted_ap["prediction_mode"] = prediction_mode
    brand_fields = [column for column in BRANDING_COLUMNS if column != "team"]
    if not branding.empty:
        actual = actual.merge(branding, on="team", how="left")
        predicted_ap = predicted_ap.merge(branding, on="team", how="left")
        independent = independent.merge(branding, on="team", how="left")
        def attach_game_branding(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            for side in ("home", "away"):
                side_branding = branding.rename(
                    columns={
                        "team": f"{side}_team",
                        **{field: f"{side}_{field}" for field in brand_fields},
                    }
                )
                frame = frame.merge(side_branding, on=f"{side}_team", how="left")
            return frame

        upcoming = attach_game_branding(upcoming)
        season_schedule = attach_game_branding(season_schedule)
    combined = independent.merge(
        actual[["team", "actual_ap_rank", "actual_ap_points"]], on="team", how="left"
    ).merge(
        predicted_ap[["team", "predicted_ap_rank", "ap_prediction_score"]],
        on="team",
        how="left",
    )
    combined["actual_ap_rank"] = combined["actual_ap_rank"].astype("Int64")
    combined["predicted_ap_rank"] = combined["predicted_ap_rank"].astype("Int64")
    combined["record"] = combined["wins"].astype(str) + "-" + combined["losses"].astype(str)
    combined["generated_at"] = generated_at
    combined["prediction_mode"] = prediction_mode
    combined["predicted_poll_week"] = target_week
    combined = combined.sort_values("independent_rank")

    atomic_write_csv(actual, settings.predictions_dir / "actual_ap_poll.csv")
    atomic_write_csv(predicted_ap, settings.predictions_dir / "predicted_ap_poll.csv")
    atomic_write_csv(independent, settings.predictions_dir / "independent_rankings.csv")
    atomic_write_csv(upcoming, settings.predictions_dir / "upcoming_game_predictions.csv")
    atomic_write_csv(season_schedule, settings.predictions_dir / "season_schedule.csv")
    atomic_write_csv(combined, settings.predictions_dir / "current_rankings.csv")
    upsert_csv(
        predicted_ap,
        settings.predictions_dir / "ap_prediction_history.csv",
        ["generated_at", "team"],
        sort_by=["generated_at", "predicted_ap_rank"],
    )
    if not upcoming.empty:
        upcoming_history = upcoming.assign(generated_at=generated_at)
        upsert_csv(
            upcoming_history,
            settings.predictions_dir / "game_prediction_history.csv",
            ["generated_at", "game_id"],
            sort_by=["generated_at", "start_date"],
        )
    return {
        "actual_ap_teams": len(actual),
        "predicted_ap_teams": len(predicted_ap),
        "independent_teams": len(independent),
        "upcoming_games": len(upcoming),
        "season_games": len(season_schedule),
        "team_logos_cached": int(
            branding["logo_path"].fillna("").ne("").sum()
        ) if not branding.empty else 0,
    }


def create_data_audit(settings: Settings) -> pd.DataFrame:
    raw = load_raw(settings)
    specs = {
        "games": (["game_id"], ["season", "week", "home_team", "away_team"]),
        "rankings": (
            ["season", "season_type", "poll_week", "poll", "team"],
            ["season", "poll_week", "poll", "team", "rank"],
        ),
        "teams": (["season", "team"], ["season", "team"]),
        "team_game_stats": (["game_id", "team"], ["game_id", "team"]),
        "advanced_game_stats": (["game_id", "team"], ["game_id", "team"]),
        "lines": (["game_id", "provider", "fetched_at"], ["game_id", "provider"]),
    }
    optional_datasets = {"team_game_stats", "advanced_game_stats", "lines"}
    rows = []
    for name, frame in raw.items():
        keys, required = specs[name]
        missing_columns = [column for column in required if column not in frame.columns]
        duplicates = (
            int(frame.duplicated([key for key in keys if key in frame.columns]).sum())
            if not frame.empty and all(key in frame.columns for key in keys)
            else 0
        )
        season_values = (
            pd.to_numeric(frame["season"], errors="coerce")
            if "season" in frame.columns
            else pd.Series(dtype=float)
        )
        required_present = [column for column in required if column in frame.columns]
        missing_required_cells = (
            int(frame[required_present].isna().sum().sum()) if required_present else 0
        )
        if frame.empty and name in optional_datasets:
            status = "OPTIONAL_MISSING"
        else:
            status = "PASS" if not missing_columns and duplicates == 0 else "REVIEW"
        rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "min_season": int(season_values.min()) if season_values.notna().any() else np.nan,
                "max_season": int(season_values.max()) if season_values.notna().any() else np.nan,
                "duplicate_keys": duplicates,
                "missing_required_cells": missing_required_cells,
                "missing_columns": ", ".join(missing_columns),
                "status": status,
            }
        )
    audit = pd.DataFrame(rows)
    atomic_write_csv(audit, settings.processed_dir / "data_audit.csv")
    return audit
