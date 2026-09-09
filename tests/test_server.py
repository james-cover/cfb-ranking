from __future__ import annotations

import asyncio
import json

import pandas as pd

from cfb_rankings.config import Settings
from cfb_rankings.server import CsvCache, create_app


async def _get(app, path: str) -> tuple[int, bytes]:
    messages = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
            "root_path": "",
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], body


def test_csv_cache_reloads_only_when_file_changes(tmp_path):
    path = tmp_path / "rankings.csv"
    pd.DataFrame([{"team": "Nebraska", "independent_rank": 1}]).to_csv(
        path, index=False
    )
    cache = CsvCache(path)

    first = cache.read()
    second = cache.read()
    assert first is second
    assert first[0]["team"] == "Nebraska"

    pd.DataFrame(
        [
            {"team": "Nebraska", "independent_rank": 1},
            {"team": "Iowa", "independent_rank": 2},
        ]
    ).to_csv(path, index=False)

    refreshed = cache.read()
    assert len(refreshed) == 2
    assert refreshed is not first


def test_app_exposes_api_before_static_frontend(tmp_path):
    settings = Settings(project_root=tmp_path, current_season=2026)
    settings.ensure_directories()
    settings.frontend_dir.mkdir(parents=True, exist_ok=True)
    (settings.frontend_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")

    app = create_app(settings)
    paths = [getattr(route, "path", "") for route in app.routes]

    assert paths[:5] == [
        "/api/health",
        "/api/rankings",
        "/api/games",
        "/api/schedule",
        "/api/teams/{team:str}",
    ]
    assert paths[-1] == ""

    status, body = asyncio.run(_get(app, "/api/health"))
    payload = json.loads(body)
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["rankings_ready"] is False


def test_rankings_endpoint_returns_prediction_csv(tmp_path):
    settings = Settings(project_root=tmp_path, current_season=2026)
    settings.ensure_directories()
    settings.frontend_dir.mkdir(parents=True, exist_ok=True)
    (settings.frontend_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    pd.DataFrame(
        [{"team": "Nebraska", "independent_rank": 1, "generated_at": "2026-09-08"}]
    ).to_csv(settings.predictions_dir / "current_rankings.csv", index=False)

    status, body = asyncio.run(_get(create_app(settings), "/api/rankings"))
    payload = json.loads(body)

    assert status == 200
    assert payload["count"] == 1
    assert payload["generated_at"] == "2026-09-08"
    assert payload["data"][0]["team"] == "Nebraska"
