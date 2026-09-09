from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


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
                if 400 <= response.status_code < 500:
                    detail = response.text.strip().replace("\n", " ")[:500]
                    raise CFBDAPIError(
                        f"HTTP {response.status_code} for {path} with {clean_params}: {detail}"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise CFBDAPIError(f"Expected a list from {path}, received {type(payload).__name__}")
                return payload
            except CFBDAPIError:
                # Retrying a permanent validation/authentication error only hides
                # the useful response and wastes several seconds.
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise CFBDAPIError(f"CFBD request failed for {path}: {last_error}")

    def games(self, year: int) -> list[dict[str, Any]]:
        # CFBD API v5 renamed the filter from ``division`` to ``classification``.
        # Sending the legacy name is silently ignored and returns every division.
        return self.get("games", year=year, classification="fbs")

    def rankings(self, year: int) -> list[dict[str, Any]]:
        return self.get("rankings", year=year)

    def team_game_stats(self, year: int) -> list[dict[str, Any]]:
        # The current endpoint requires week, team, or conference when year is used.
        # Fetching by week avoids hard-coding a conference list and keeps requests bounded.
        payload: list[dict[str, Any]] = []
        failures: list[str] = []
        # Week 0 is used in modern seasons. Some historical seasons reject week
        # values that do not exist; those failures must not erase valid weeks.
        requests_to_make = [
            *(('regular', week) for week in range(17)),
            *(('postseason', week) for week in range(1, 6)),
        ]
        for season_type, week in requests_to_make:
            try:
                payload.extend(
                    self.get(
                        "games/teams",
                        year=year,
                        week=week,
                        seasonType=season_type,
                        classification="fbs",
                    )
                )
            except CFBDAPIError as exc:
                failures.append(f"{season_type} week {week}: {exc}")
                LOGGER.info("Skipping unavailable %s team stats", failures[-1])
        if not payload:
            detail = failures[0] if failures else "all successful requests returned zero rows"
            raise CFBDAPIError(f"No team box scores returned for {year}; first failure: {detail}")
        if failures:
            LOGGER.warning(
                "Retained %d team-stat games for %d despite %d unavailable week requests",
                len(payload), year, len(failures),
            )
        return payload

    def advanced_game_stats(self, year: int) -> list[dict[str, Any]]:
        return self.get("stats/game/advanced", year=year)

    def lines(self, year: int) -> list[dict[str, Any]]:
        return self.get("lines", year=year)

    def fbs_teams(self, year: int) -> list[dict[str, Any]]:
        return self.get("teams/fbs", year=year)

    def media(self, year: int) -> list[dict[str, Any]]:
        return self.get("games/media", year=year)
