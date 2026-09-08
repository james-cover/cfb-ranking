from cfb_rankings.ingest import normalize_games, normalize_lines, normalize_rankings


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
                "lines": [{"provider": "Book A", "spread": -3.5}],
            }
        ],
        fetched_at,
    )
    assert lines.loc[0, "spread"] == -3.5

