from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .config import Settings, load_settings


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    # Pandas' JSON encoder consistently converts NaN/NaT to JSON null.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


@dataclass
class CsvCache:
    """Cache parsed CSV records until the file's timestamp or size changes."""

    path: Path
    _signature: tuple[int, int] | None = None
    _records: list[dict[str, Any]] = field(default_factory=list)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            self._signature = None
            self._records = []
            return []
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature != self._signature:
            self._records = _records(pd.read_csv(self.path))
            self._signature = signature
        return self._records


def _payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    generated = next(
        (row.get("generated_at") for row in records if row.get("generated_at")),
        None,
    )
    return {"data": records, "count": len(records), "generated_at": generated}


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or load_settings()
    rankings = CsvCache(settings.predictions_dir / "current_rankings.csv")
    games = CsvCache(settings.predictions_dir / "upcoming_game_predictions.csv")
    schedule = CsvCache(settings.predictions_dir / "season_schedule.csv")

    async def health(_request: Request) -> JSONResponse:
        ranking_rows = rankings.read()
        game_rows = games.read()
        return JSONResponse(
            {
                "status": "ok",
                "rankings_ready": bool(ranking_rows),
                "games_ready": bool(game_rows),
                "ranking_count": len(ranking_rows),
                "game_count": len(game_rows),
                "schedule_count": len(schedule.read()),
            },
            headers={"Cache-Control": "no-store"},
        )

    async def ranking_data(_request: Request) -> JSONResponse:
        return JSONResponse(
            _payload(rankings.read()),
            headers={"Cache-Control": "no-cache"},
        )

    async def game_data(_request: Request) -> JSONResponse:
        return JSONResponse(
            _payload(games.read()),
            headers={"Cache-Control": "no-cache"},
        )

    async def schedule_data(_request: Request) -> JSONResponse:
        return JSONResponse(
            _payload(schedule.read()),
            headers={"Cache-Control": "no-cache"},
        )

    async def team_data(request: Request) -> JSONResponse:
        requested = request.path_params["team"].casefold()
        team = next(
            (row for row in rankings.read() if str(row.get("team", "")).casefold() == requested),
            None,
        )
        if team is None:
            return JSONResponse({"detail": "Team not found"}, status_code=404)
        team_name = str(team["team"])
        team_games = [
            row
            for row in schedule.read()
            if row.get("home_team") == team_name or row.get("away_team") == team_name
        ]
        return JSONResponse(
            {"team": team, "games": team_games},
            headers={"Cache-Control": "no-cache"},
        )

    routes = [
        Route("/api/health", health),
        Route("/api/rankings", ranking_data),
        Route("/api/games", game_data),
        Route("/api/schedule", schedule_data),
        Route("/api/teams/{team:str}", team_data),
        Mount(
            "/team_logos",
            StaticFiles(directory=settings.team_logos_dir, check_dir=False),
            name="team_logos",
        ),
        Mount(
            "/",
            StaticFiles(directory=settings.frontend_dir, html=True, check_dir=False),
            name="frontend",
        ),
    ]
    return Starlette(debug=False, routes=routes)


app = create_app()
