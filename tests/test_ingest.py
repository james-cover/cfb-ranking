import pytest

from cfb_rankings.cfbd import CFBDAPIError, CFBDClient
from cfb_rankings.ingest import (
    normalize_games,
    normalize_lines,
    normalize_rankings,
    normalize_team_game_stats,
)
from cfb_rankings.pipeline import box_score_coverage, require_box_scores


def test_normalizers_flatten_core_payloads():
    fetched_at = "2026-09-08T15:00:00+00:00"
    games = normalize_games(
        [
            {
                "id": 1,
                "season": 2026,
                "week": 2,
                "seasonType": "regular",
                "startDate": "2026-09-12T18:00:00Z",
                "completed": False,
                "neutralSite": False,
                "homeTeam": "Nebraska",
                "awayTeam": "Iowa",
            }
        ],
        fetched_at,
    )
    assert games.loc[0, "home_team"] == "Nebraska"
    assert not bool(games.loc[0, "completed"])

    rankings = normalize_rankings(
        [
            {
                "season": 2026,
                "seasonType": "regular",
                "week": 2,
                "polls": [
                    {
                        "poll": "AP Top 25",
                        "ranks": [{"rank": 20, "school": "Nebraska", "points": 220}],
                    }
                ],
            }
        ],
        fetched_at,
    )
    assert rankings.loc[0, "points"] == 220
    assert rankings.loc[0, "first_place_votes"] == 0

    lines = normalize_lines(
        [
            {
                "id": 1,
                "season": 2026,
                "week": 2,
                "homeTeam": "Nebraska",
                "awayTeam": "Iowa",
                "lines": [
                    {
                        "provider": "Book A",
                        "spread": -3.5,
                        "homeSpreadOdds": -105,
                        "awaySpreadOdds": -115,
                    }
                ],
            }
        ],
        fetched_at,
    )
    assert lines.loc[0, "spread"] == -3.5
    assert lines.loc[0, "home_spread_odds"] == -105
    assert lines.loc[0, "away_spread_odds"] == -115


def test_client_uses_current_classification_filter(monkeypatch):
    client = CFBDClient("test-key")
    calls = []

    def fake_get(path, **params):
        calls.append((path, params))
        return ([{"id": 1, "teams": []}]
                if path == "games/teams" and params.get("week") == 1 else [])

    monkeypatch.setattr(client, "get", fake_get)
    client.games(2026)
    client.team_game_stats(2026)

    assert calls[0] == ("games", {"year": 2026, "classification": "fbs"})
    assert len(calls) == 23
    assert calls[1][1]["week"] == 0
    assert all(call[1]["classification"] == "fbs" for call in calls)
    assert all("week" in call[1] for call in calls[1:])


def test_team_stats_keep_successful_weeks_when_one_week_fails(monkeypatch):
    client = CFBDClient("test-key")

    def fake_get(path, **params):
        if params["week"] == 16:
            raise CFBDAPIError("invalid week")
        return [{"id": params["week"], "teams": []}] if params["week"] == 1 else []

    monkeypatch.setattr(client, "get", fake_get)
    result = client.team_game_stats(2026)
    assert len(result) == 2  # regular and postseason week 1 both survived


def test_box_score_normalization_and_required_coverage():
    frame = normalize_team_game_stats(
        [{"id": 1, "teams": [{"school": "Alpha", "stats": [
            {"category": "rushingYards", "stat": "187"},
            {"category": "netPassingYards", "stat": "301"},
            {"category": "possessionTime", "stat": "31:42"},
        ]}]}],
        "2026-09-09T00:00:00Z",
    )
    assert box_score_coverage(frame) == {
        "rushingYards": 1, "netPassingYards": 1, "possessionTime": 1,
    }
    assert require_box_scores(frame)["possessionTime"] == 1
    with pytest.raises(ValueError, match="no time of possession"):
        require_box_scores(frame.assign(stats_json='{"rushingYards":"1","netPassingYards":"2"}'))
