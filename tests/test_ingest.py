from cfb_rankings.ingest import normalize_games, normalize_lines, normalize_rankings
from cfb_rankings.cfbd import CFBDClient


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
        return []

    monkeypatch.setattr(client, "get", fake_get)
    client.games(2026)
    client.team_game_stats(2026)

    assert calls[0] == ("games", {"year": 2026, "classification": "fbs"})
    assert len(calls) == 22
    assert all(call[1]["classification"] == "fbs" for call in calls)
    assert all("week" in call[1] for call in calls[1:])
