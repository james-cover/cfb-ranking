import warnings

import numpy as np
import pandas as pd
import pytest

from cfb_rankings.evaluation import ats_feature_scan, evaluate_ats
from cfb_rankings.features import (
    MATCHUP_FEATURES,
    TeamState,
    fbs_team_names,
    model_adjusted_snapshot,
)
from cfb_rankings.game_model import (
    apply_edge_predictions,
    build_independent_rankings,
    load_edge_model,
    load_game_model,
    predict_upcoming_games,
    select_and_train_game_model,
    train_edge_model,
)
from cfb_rankings.validated_training import _split, fit_edge_shrinkage


def test_pushes_are_not_losses_and_zero_edge_is_not_a_pick():
    result = evaluate_ats([3, 7, 0, 9], [4, 4, 4, 3], [3, 3, 3, 3])
    assert result['ats_games_0pt'] == 2
    assert result['ats_pushes_0pt'] == 1
    assert result['ats_accuracy_0pt'] == .5
    assert result['ats_roi_0pt'] == pytest.approx((100/110 - 1)/3)


def test_constant_features_do_not_trigger_correlation_warnings():
    frame = pd.DataFrame({'home_margin': np.tile([0, 3, 7], 100),
                          'market_home_margin': 3, 'constant': 0,
                          'variable': np.arange(300)})
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        scan = ats_feature_scan(frame, ['constant', 'variable'])
    assert scan.feature.tolist() == ['variable']
    assert 'corr_with_cover_residual' in scan


def test_unavailable_stats_are_not_synthetic_signals():
    early, late = TeamState('A').snapshot(), TeamState('B').snapshot()
    early.update(games=1, yards_per_game=0)
    late.update(games=10, yards_per_game=0)
    assert model_adjusted_snapshot(early)['yards_per_game'] == 375
    assert model_adjusted_snapshot(late)['yards_per_game'] == 375
    late['prior_points_per_game'] = np.nan
    assert np.isfinite(model_adjusted_snapshot(late)['points_per_game'])


def test_historical_classification_overrides_current_membership():
    games = pd.DataFrame([{'season': 2024, 'home_team': 'Promoted', 'away_team': 'A',
                           'home_classification': 'fcs', 'away_classification': 'fbs'}])
    teams = pd.DataFrame([{'season': 2024, 'team': 'Promoted'}, {'season': 2024, 'team': 'A'}])
    assert fbs_team_names(games, teams, 2024) == {'A'}


def test_chronological_split_and_nonpositive_edge_shrinkage():
    _, folds, cal, test = _split(pd.DataFrame({'season': range(2014, 2027)}), 2026, 4)
    assert folds == [2020, 2021, 2022, 2023]
    assert (cal, test) == (2024, 2025)
    assert fit_edge_shrinkage(-np.arange(100), np.arange(100)) == (0., 0.)


def synthetic_history():
    rng = np.random.default_rng(73)
    n = 6*100
    frame = pd.DataFrame({name: rng.normal(size=n) for name in MATCHUP_FEATURES})
    frame['season'] = np.repeat(np.arange(2019, 2025), 100)
    frame['week'] = np.tile(np.arange(100) % 14 + 1, 6)
    frame['game_id'] = np.arange(n)
    frame['home_team'], frame['away_team'] = 'A', 'B'
    frame['home_field'] = rng.integers(0, 2, n)
    frame['elo_diff'] *= 200
    frame['turnover_margin_per_game_diff'] = 0  # must be excluded from fitted features
    frame['home_margin'] = .04*frame.elo_diff + 3*frame.home_field + rng.normal(0, 8, n)
    frame['game_total'] = 50 + rng.normal(0, 5, n)
    frame['market_home_margin'] = .04*frame.elo_diff + 3*frame.home_field
    return frame


def test_real_xgboost_train_save_rank_predict_and_live_edge_inputs(tmp_path):
    history = synthetic_history()
    bundle, evidence = select_and_train_game_model(history, tmp_path, validation_seasons=1)
    assert evidence.iloc[0]['scope'] == 'held_out_test'
    assert evidence.iloc[0]['test_season'] == 2024
    assert evidence.iloc[0]['calibration_season'] == 2023
    assert 'market_home_margin' not in bundle.features
    assert 'turnover_margin_per_game_diff' not in bundle.features
    restored = load_game_model(tmp_path)
    np.testing.assert_allclose(bundle.predict_margin(history.head()), restored.predict_margin(history.head()))
    contributions = restored.predict_margin_contributions(history.head())
    np.testing.assert_allclose(contributions.sum(axis=1), restored.predict_margin(history.head()), atol=1e-4)
    edge, metrics = train_edge_model(history, tmp_path, validation_seasons=1)
    assert load_edge_model(tmp_path).schema_version == 6
    assert not edge.validated  # test season has fewer than 400 decisive games
    assert metrics['edge_model'] == 'trained_not_validated'
    states = pd.DataFrame([TeamState('A', elo=1650).snapshot(), TeamState('B', elo=1450).snapshot()])
    games = pd.DataFrame([{'game_id': 700, 'season': 2025, 'week': 2,
                           'start_date': '2025-09-07T12:00:00Z', 'completed': False,
                           'neutral_site': False, 'home_team': 'A', 'away_team': 'B'}])
    lines = pd.DataFrame([{'game_id': 700, 'spread': -3.5, 'provider': 'Example',
                           'fetched_at': '2025-09-06T12:00:00Z', 'home_moneyline': -140,
                           'away_moneyline': 120, 'home_spread_odds': -110, 'away_spread_odds': -110}])
    upcoming = predict_upcoming_games(restored, games, states, lines, 2025)
    assert set(MATCHUP_FEATURES) <= set(upcoming.columns)
    assert upcoming.iloc[0].elo_diff == 200
    assert apply_edge_predictions(upcoming, edge).home_spread_ev_100.isna().all()
    independent = build_independent_rankings(restored, states, {'A', 'B'})
    assert len(independent) == 2
    assert independent.model_rating.sum() == pytest.approx(0)

    class CheckingEdge:
        validated = True

        def predict_edge(self, frame):
            assert frame.iloc[0].elo_diff == 200
            assert set(MATCHUP_FEATURES) <= set(frame.columns)
            return np.full(len(frame), 4.)

        def cover_probability(self, edge):
            return np.full(len(edge), .90)

    updated = apply_edge_predictions(upcoming, CheckingEdge())
    assert updated.iloc[0].best_prop_type == 'spread'
    assert updated.iloc[0].best_prop_team == 'A'
    assert updated.iloc[0].betting_model_home_margin == 7.5
    assert updated.iloc[0].model_home_margin == upcoming.iloc[0].model_home_margin
    assert not updated.columns.duplicated().any()
    integer_line = upcoming.copy()
    integer_line['home_spread'] = -3
    assert apply_edge_predictions(integer_line, CheckingEdge()).home_spread_ev_100.isna().all()
