from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class CFBDAPIError(RuntimeError):
    pass


@dataclass
class CFBDClient:
    api_key: str
    base_url: str = "https://api.collegefootballdata.com"
    timeout_seconds: int = 60
    retries: int = 4

    def __post_init__(self) -> None:
        if not self.api_key or self.api_key == "replace_me":
            raise ValueError("CFBD_API_KEY is missing. Copy .env.example to .env and add your key.")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "cfb-ranking-model/0.1",
            }
        )

    def get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        clean_params = {key: value for key, value in params.items() if value is not None}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, params=clean_params, timeout=self.timeout_seconds)
                if response.status_code == 429 or response.status_code >= 500:
                    delay = min(2**attempt, 8)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise CFBDAPIError(f"Expected a list from {path}, received {type(payload).__name__}")
                return payload
            except (requests.RequestException, ValueError, CFBDAPIError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise CFBDAPIError(f"CFBD request failed for {path}: {last_error}")

    def games(self, year: int) -> list[dict[str, Any]]:
        return self.get("games", year=year, division="fbs")

    def rankings(self, year: int) -> list[dict[str, Any]]:
        return self.get("rankings", year=year)

    def team_game_stats(self, year: int) -> list[dict[str, Any]]:
        return self.get("games/teams", year=year, classification="fbs")

    def advanced_game_stats(self, year: int) -> list[dict[str, Any]]:
        return self.get("stats/game/advanced", year=year)

    def lines(self, year: int) -> list[dict[str, Any]]:
        return self.get("lines", year=year)

    def fbs_teams(self, year: int) -> list[dict[str, Any]]:
        return self.get("teams/fbs", year=year)

