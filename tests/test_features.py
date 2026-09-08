import pandas as pd

from cfb_rankings.features import build_ap_training_frame, build_sequential_features


def sample_games():
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "season": 2024,
                "week": 1,
                "season_type": "regular",
                "start_date": "2024-08-31T18:00:00Z",
                "completed": True,
                "neutral_site": False,
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_points": 31,
                "away_points": 10,
            },
            {
                "game_id": 2,
                "season": 2024,
                "week": 2,
                "season_type": "regular",
                "start_date": "2024-09-07T18:00:00Z",
                "completed": True,
                "neutral_site": False,
                "home_team": "Beta",
                "away_team": "Alpha",
                "home_points": 20,
                "away_points": 24,
            },
        ]
    )


def sample_rankings():
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "season_type": "regular",
                "poll_week": 1,
                "poll": "AP Top 25",
                "team": "Alpha",
                "rank": 20,
                "points": 100,
                "first_place_votes": 0,
            },
            {
                "season": 2024,
                "season_type": "regular",
                "poll_week": 2,
                "poll": "AP Top 25",
                "team": "Alpha",
                "rank": 12,
                "points": 400,
                "first_place_votes": 0,
            },
            {
                "season": 2024,
                "season_type": "regular",
                "poll_week": 3,
                "poll": "AP Top 25",
                "team": "Alpha",
                "rank": 10,
                "points": 500,
                "first_place_votes": 0,
            },
        ]
    )


def test_game_rows_use_only_pregame_state():
    game_features, weekly = build_sequential_features(sample_games(), sample_rankings())
    first = game_features.iloc[0]
    second = game_features.iloc[1]
    assert first["home_games"] == 0
    assert first["away_games"] == 0
    assert second["away_games"] == 1
    assert second["away_avg_margin"] == 21
    assert len(weekly) == 4


def test_ap_target_uses_previous_week_features_and_poll():
    _, weekly = build_sequential_features(sample_games(), sample_rankings())
    teams = pd.DataFrame(
        [
            {"season": 2024, "team": "Alpha"},
            {"season": 2024, "team": "Beta"},
        ]
    )
    ap_frame = build_ap_training_frame(weekly, sample_rankings(), teams)
    alpha_week_2 = ap_frame[
        ap_frame["team"].eq("Alpha") & ap_frame["target_poll_week"].eq(2)
    ].iloc[0]
    assert alpha_week_2["games"] == 1
    assert alpha_week_2["previous_ap_rank"] == 20
    assert alpha_week_2["actual_ap_rank"] == 12

