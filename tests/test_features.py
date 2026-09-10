import json

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
    # Model differentials use a four-game neutral prior, so one result is not
    # treated as a stable full-season average.
    assert round(second["avg_margin_diff"], 1) == -8.4
    assert len(weekly) == 4


def test_current_cfbd_box_scores_join_and_enter_next_games_features():
    stats = pd.DataFrame([
        {"game_id": 1, "team": "Alpha", "stats_json": json.dumps({
            "rushingYards": "210", "netPassingYards": "300",
            "possessionTime": "34:00", "totalYards": "510", "turnovers": "0",
        })},
        {"game_id": 1, "team": "Beta", "stats_json": json.dumps({
            "rushingYards": "90", "netPassingYards": "140",
            "possessionTime": "26:00", "totalYards": "230", "turnovers": "2",
        })},
        {"game_id": 2, "team": "Alpha", "stats_json": json.dumps({
            "rushingYards": "180", "netPassingYards": "250", "possessionTime": "31:00",
        })},
        {"game_id": 2, "team": "Beta", "stats_json": json.dumps({
            "rushingYards": "130", "netPassingYards": "200", "possessionTime": "29:00",
        })},
    ])
    game_features, _ = build_sequential_features(
        sample_games(), sample_rankings(), team_game_stats=stats
    )
    assert game_features["box_score_available"].all()
    first = game_features.iloc[0]
    second = game_features.iloc[1]
    assert first["opp_adj_passing_off_diff"] == 0
    assert first["opp_adj_rushing_off_diff"] == 0
    assert second["rushing_ypg_diff"] != 0
    assert second["passing_ypg_diff"] != 0
    assert second["possession_time_pg_diff"] != 0
    assert second["opp_adj_passing_off_diff"] != 0
    assert second["opp_adj_rushing_off_diff"] != 0


def test_lower_division_elo_does_not_carry_into_a_new_season():
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "season": 2024,
                "week": 1,
                "season_type": "regular",
                "start_date": "2024-08-31T18:00:00Z",
                "completed": True,
                "neutral_site": False,
                "home_team": "FCS Power",
                "away_team": "Alpha",
                "home_points": 35,
                "away_points": 10,
            },
            {
                "game_id": 2,
                "season": 2025,
                "week": 1,
                "season_type": "regular",
                "start_date": "2025-08-31T18:00:00Z",
                "completed": True,
                "neutral_site": False,
                "home_team": "FCS Power",
                "away_team": "Alpha",
                "home_points": 21,
                "away_points": 20,
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {"season": 2024, "team": "Alpha"},
            {"season": 2025, "team": "Alpha"},
        ]
    )
    game_features, weekly = build_sequential_features(games, teams=teams)
    assert game_features.empty  # FBS/FCS games update state but are not training targets.
    # Alpha's pregame opponent strength remains the FCS baseline both years.
    alpha = weekly[weekly.team.eq("Alpha")]
    assert alpha["last_week_opponent_elo"].eq(1350.0).all()


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
