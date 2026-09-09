from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cfbd import CFBDAPIError, CFBDClient
from .config import Settings
from .storage import atomic_write_csv, upsert_csv

LOGGER = logging.getLogger(__name__)


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    cleaned = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_games(payload: Iterable[dict[str, Any]], fetched_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in payload:
        rows.append(
            {
                "game_id": _first(game, "id"),
                "season": _first(game, "season"),
                "week": _first(game, "week"),
                "season_type": _first(game, "seasonType", "season_type"),
                "start_date": _first(game, "startDate", "start_date"),
                "completed": bool(_first(game, "completed", default=False)),
                "neutral_site": bool(_first(game, "neutralSite", "neutral_site", default=False)),
                "conference_game": bool(
                    _first(game, "conferenceGame", "conference_game", default=False)
                ),
                "home_team": _first(game, "homeTeam", "home_team"),
                "home_conference": _first(game, "homeConference", "home_conference"),
                "home_classification": _first(
                    game, "homeClassification", "home_classification"
                ),
                "home_points": _first(game, "homePoints", "home_points"),
                "home_pregame_elo": _first(game, "homePregameElo", "home_pregame_elo"),
                "away_team": _first(game, "awayTeam", "away_team"),
                "away_conference": _first(game, "awayConference", "away_conference"),
                "away_classification": _first(
                    game, "awayClassification", "away_classification"
                ),
                "away_points": _first(game, "awayPoints", "away_points"),
                "away_pregame_elo": _first(game, "awayPregameElo", "away_pregame_elo"),
                "venue": _first(game, "venue"),
                "fetched_at": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def normalize_teams(payload: Iterable[dict[str, Any]], season: int, fetched_at: str) -> pd.DataFrame:
    rows = []
    for team in payload:
        rows.append(
            {
                "season": season,
                "team_id": _first(team, "id"),
                "team": _first(team, "school"),
                "mascot": _first(team, "mascot"),
                "abbreviation": _first(team, "abbreviation"),
                "conference": _first(team, "conference"),
                "classification": _first(team, "classification"),
                "color": _first(team, "color"),
                "alternate_color": _first(team, "altColor", "alternateColor"),
                "logos_json": json.dumps(_first(team, "logos", default=[])),
                "fetched_at": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def normalize_rankings(payload: Iterable[dict[str, Any]], fetched_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for poll_week in payload:
        season = _first(poll_week, "season")
        week = _first(poll_week, "week")
        season_type = _first(poll_week, "seasonType", "season_type")
        for poll in _first(poll_week, "polls", default=[]) or []:
            poll_name = _first(poll, "poll")
            for team in _first(poll, "ranks", default=[]) or []:
                rows.append(
                    {
                        "season": season,
                        "season_type": season_type,
                        "poll_week": week,
                        "poll": poll_name,
                        "rank": _first(team, "rank"),
                        "team": _first(team, "school"),
                        "conference": _first(team, "conference"),
                        "first_place_votes": _first(
                            team, "firstPlaceVotes", "first_place_votes", default=0
                        ),
                        "points": _first(team, "points", default=0),
                        "fetched_at": fetched_at,
                    }
                )
    return pd.DataFrame(rows)


def normalize_team_game_stats(
    payload: Iterable[dict[str, Any]], fetched_at: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in payload:
        game_id = _first(game, "id")
        for team in _first(game, "teams", default=[]) or []:
            base = {
                "game_id": game_id,
                "team": _first(team, "school"),
                "conference": _first(team, "conference"),
                "home_away": _first(team, "homeAway", "home_away"),
                "points": _first(team, "points"),
                "fetched_at": fetched_at,
            }
            stats: dict[str, Any] = {}
            for item in _first(team, "stats", default=[]) or []:
                category = str(_first(item, "category", default="unknown"))
                stats[category] = _first(item, "stat")
            base["stats_json"] = json.dumps(stats, sort_keys=True)
            rows.append(base)
    return pd.DataFrame(rows)


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else key
            _flatten(next_prefix, child, output)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value


def normalize_advanced_stats(
    payload: Iterable[dict[str, Any]], fetched_at: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload:
        flattened: dict[str, Any] = {}
        _flatten("", item, flattened)
        row = {
            key.replace("gameId", "game_id"): value
            for key, value in flattened.items()
            if not isinstance(value, (dict, list))
        }
        row["fetched_at"] = fetched_at
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_media(payload: Iterable[dict[str, Any]], fetched_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload:
        media_type = _first(item, "mediaType", "media_type", default="")
        # Keep only TV broadcasts (skip radio, web, PPV, etc.)
        if str(media_type).lower() not in ("tv", "web", "ppv", ""):
            pass  # still include — filtering happens downstream
        rows.append(
            {
                "game_id": _first(item, "id"),
                "season": _first(item, "season"),
                "week": _first(item, "week"),
                "season_type": _first(item, "seasonType", "season_type"),
                "start_date": _first(item, "startDate", "start_date"),
                "home_team": _first(item, "homeTeam", "home_team"),
                "away_team": _first(item, "awayTeam", "away_team"),
                "media_type": media_type,
                "outlet": _first(item, "outlet", default=""),
                "fetched_at": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def normalize_lines(payload: Iterable[dict[str, Any]], fetched_at: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in payload:
        base = {
            "game_id": _first(game, "id"),
            "season": _first(game, "season"),
            "season_type": _first(game, "seasonType", "season_type"),
            "week": _first(game, "week"),
            "start_date": _first(game, "startDate", "start_date"),
            "home_team": _first(game, "homeTeam", "home_team"),
            "away_team": _first(game, "awayTeam", "away_team"),
        }
        for line in _first(game, "lines", default=[]) or []:
            row = {
                **base,
                "provider": _first(line, "provider", default="unknown"),
                # CFBD spread is the home-team handicap: negative means home favorite.
                "spread": _number(_first(line, "spread")),
                "formatted_spread": _first(line, "formattedSpread", "formatted_spread"),
                "spread_open": _number(_first(line, "spreadOpen", "spread_open")),
                # CFBD does not currently guarantee spread-price fields, but retain
                # them when a provider payload supplies them. The prediction layer
                # labels its standard -110 fallback explicitly when these are absent.
                "home_spread_odds": _number(
                    _first(
                        line,
                        "homeSpreadOdds",
                        "home_spread_odds",
                        "homeSpreadPrice",
                        "home_spread_price",
                    )
                ),
                "away_spread_odds": _number(
                    _first(
                        line,
                        "awaySpreadOdds",
                        "away_spread_odds",
                        "awaySpreadPrice",
                        "away_spread_price",
                    )
                ),
                "over_under": _number(_first(line, "overUnder", "over_under")),
                "over_under_open": _number(
                    _first(line, "overUnderOpen", "over_under_open")
                ),
                "home_moneyline": _number(
                    _first(line, "homeMoneyline", "home_moneyline")
                ),
                "away_moneyline": _number(
                    _first(line, "awayMoneyline", "away_moneyline")
                ),
                "fetched_at": fetched_at,
            }
            rows.append(row)
    return pd.DataFrame(rows)


class IngestionPipeline:
    def __init__(self, client: CFBDClient, settings: Settings):
        self.client = client
        self.settings = settings

    def _fetch_and_store(
        self,
        years: Iterable[int],
        fetcher: Callable[[int], list[dict[str, Any]]],
        normalizer: Callable[..., pd.DataFrame],
        output_name: str,
        keys: list[str],
        *,
        optional: bool = False,
        replace_seasons: bool = False,
    ) -> pd.DataFrame:
        years = list(years)
        frames: list[pd.DataFrame] = []
        fetched_at = datetime.now(UTC).isoformat()
        for year in years:
            try:
                payload = fetcher(year)
                if output_name == "teams.csv":
                    frame = normalizer(payload, year, fetched_at)
                else:
                    frame = normalizer(payload, fetched_at)
                frames.append(frame)
                LOGGER.info("Fetched %s rows for %s %s", len(frame), output_name, year)
            except CFBDAPIError:
                if not optional:
                    raise
                LOGGER.warning("Optional endpoint for %s failed in %s", output_name, year)
        new_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        path = self.settings.raw_dir / output_name
        if replace_seasons and not new_rows.empty and "season" in new_rows.columns:
            existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if not existing.empty and "season" in existing.columns:
                existing_seasons = pd.to_numeric(existing["season"], errors="coerce")
                existing = existing[~existing_seasons.isin(years)]
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(keys, keep="last")
            available = [column for column in keys if column in combined.columns]
            if available:
                combined = combined.sort_values(available, kind="stable")
            combined = combined.reset_index(drop=True)
            atomic_write_csv(combined, path)
            return combined
        return upsert_csv(new_rows, path, keys, sort_by=keys)

    def bootstrap(self, start_year: int, end_year: int) -> dict[str, int]:
        years = range(start_year, end_year + 1)
        outputs = {
            "games": self._fetch_and_store(
                years,
                self.client.games,
                normalize_games,
                "games.csv",
                ["game_id"],
                replace_seasons=True,
            ),
            "rankings": self._fetch_and_store(
                years,
                self.client.rankings,
                normalize_rankings,
                "rankings.csv",
                ["season", "season_type", "poll_week", "poll", "team"],
            ),
            "teams": self._fetch_and_store(
                years,
                self.client.fbs_teams,
                normalize_teams,
                "teams.csv",
                ["season", "team"],
            ),
            "team_game_stats": self._fetch_and_store(
                years,
                self.client.team_game_stats,
                normalize_team_game_stats,
                "team_game_stats.csv",
                ["game_id", "team"],
                optional=True,
            ),
            "advanced_game_stats": self._fetch_and_store(
                years,
                self.client.advanced_game_stats,
                normalize_advanced_stats,
                "advanced_game_stats.csv",
                ["game_id", "team"],
                optional=True,
            ),
        }
        # Lines are snapshots. Repeated refreshes retain observations by retrieval time.
        line_frames = []
        fetched_at = datetime.now(UTC).isoformat()
        for year in years:
            try:
                line_frames.append(normalize_lines(self.client.lines(year), fetched_at))
            except CFBDAPIError:
                LOGGER.warning("Optional betting-lines endpoint failed in %s", year)
        new_lines = pd.concat(line_frames, ignore_index=True) if line_frames else pd.DataFrame()
        line_path = self.settings.raw_dir / "betting_lines.csv"
        if not new_lines.empty:
            existing = pd.read_csv(line_path) if line_path.exists() else pd.DataFrame()
            lines = pd.concat([existing, new_lines], ignore_index=True)
            line_keys = ["game_id", "provider", "fetched_at"]
            lines = lines.drop_duplicates(line_keys, keep="last").sort_values(line_keys)
            atomic_write_csv(lines, line_path)
        else:
            lines = pd.read_csv(line_path) if line_path.exists() else pd.DataFrame()
        outputs["betting_lines"] = lines

        # Game media / broadcast info (TV channel, outlet).
        outputs["media"] = self._fetch_and_store(
            years,
            self.client.media,
            normalize_media,
            "media.csv",
            ["game_id", "media_type"],
            optional=True,
            replace_seasons=True,
        )
        return {name: len(frame) for name, frame in outputs.items()}

    def refresh_current(self, season: int | None = None) -> dict[str, int]:
        year = season or self.settings.season
        return self.bootstrap(year, year)


def audit_raw_data(raw_dir: Path) -> pd.DataFrame:
    required = {
        "games.csv": ["game_id", "season", "week", "home_team", "away_team"],
        "rankings.csv": ["season", "poll_week", "poll", "rank", "team"],
        "teams.csv": ["season", "team"],
    }
    rows = []
    for filename, columns in required.items():
        path = raw_dir / filename
        if not path.exists():
            rows.append({"file": filename, "status": "missing", "rows": 0, "issue": "not fetched"})
            continue
        frame = pd.read_csv(path)
        missing_columns = [column for column in columns if column not in frame.columns]
        null_key_rows = int(frame[columns].isna().any(axis=1).sum()) if not missing_columns else None
        status = "ok" if not missing_columns and not null_key_rows else "warning"
        issue = ""
        if missing_columns:
            issue = f"missing columns: {', '.join(missing_columns)}"
        elif null_key_rows:
            issue = f"{null_key_rows} rows have null required values"
        rows.append({"file": filename, "status": status, "rows": len(frame), "issue": issue})
    for filename in ("team_game_stats.csv", "advanced_game_stats.csv", "betting_lines.csv"):
        path = raw_dir / filename
        rows.append(
            {
                "file": filename,
                "status": "optional" if not path.exists() else "ok",
                "rows": len(pd.read_csv(path)) if path.exists() else 0,
                "issue": "endpoint unavailable or not fetched" if not path.exists() else "",
            }
        )
    return pd.DataFrame(rows)
