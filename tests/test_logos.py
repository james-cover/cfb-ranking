from __future__ import annotations

import json
from typing import ClassVar

import pandas as pd

from cfb_rankings.logos import build_team_branding


class FakeResponse:
    headers: ClassVar[dict[str, str]] = {
        "Content-Type": "image/png",
        "Content-Length": "8",
    }

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield b"fake-png"


class FakeSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls = 0

    def get(self, url: str, *, timeout: int, stream: bool) -> FakeResponse:
        assert url == "https://example.test/158.png"
        assert timeout == 30
        assert stream is True
        self.calls += 1
        return FakeResponse()


def test_branding_downloads_once_and_returns_frontend_path(tmp_path):
    teams = pd.DataFrame(
        [
            {
                "season": 2026,
                "team_id": 158,
                "team": "Nebraska",
                "color": "#E41C38",
                "alternate_color": "#FFFFFF",
                "logos_json": json.dumps(["https://example.test/158.png"]),
            }
        ]
    )
    session = FakeSession()

    first = build_team_branding(teams, 2026, tmp_path, session=session)
    second = build_team_branding(teams, 2026, tmp_path, session=session)

    assert first.iloc[0]["logo_path"] == "team_logos/158.png"
    assert first.iloc[0]["logo_url"] == "https://example.test/158.png"
    assert first.iloc[0]["team_color"] == "#E41C38"
    assert first.iloc[0]["alt_color"] == "#FFFFFF"
    assert second.iloc[0]["logo_path"] == "team_logos/158.png"
    assert (tmp_path / "158.png").read_bytes() == b"fake-png"
    assert session.calls == 1
