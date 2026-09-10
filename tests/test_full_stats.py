import numpy as np
import pandas as pd
import pytest

from cfb_rankings.bayesian_training import _fit, _predict, candidate_grid
from cfb_rankings.features import build_sequential_features, matchup_features_from_snapshots
from cfb_rankings.full_stats import METRICS, FullStatProfiles, observations, ratio
from cfb_rankings.game_model import GameModelBundle


def test_parse_rates_time_and_no_fake_missing_values():
    values = observations({"netPassingYards": 300, "rushingYards": 120,
                           "completionAttempts": "20-30", "rushingAttempts": 24,
                           "possessionTime": "32:30", "thirdDownEff": "6-12",
                           "totalPenaltiesYards": "5-45", "turnovers": 0},
                          {"offense_ppa": .2, "offense_success_rate": .45}, 30, 20,
                          {"turnovers": 2})
    assert values["yards"] == 420
    assert values["yards_per_pass"] == 10
    assert values["yards_per_rush"] == 5
    assert values["possession"] == 32.5
    assert values["third_down"] == .5
    assert values["penalty_yards"] == 45
    assert values["turnover_margin"] == 2
    assert ratio("0-0") is None
    assert ratio("50%") == .5
    missing = observations({}, {}, 7, 3, {})
    assert missing["possession"] is None
    assert missing["ppa"] is None
    assert missing["yards"] is None


def test_missing_stat_does_not_dilute_average_and_week_is_frozen():
    bank = FullStatProfiles(["A", "B"])
    initial = bank.snapshot("A")
    bank.record("A", "B", {"points": 40, "passing": 300})
    pd.testing.assert_series_equal(pd.Series(bank.snapshot("A")), pd.Series(initial))
    assert bank.predict("A", "B", "passing") == 215
    bank.finish_week()
    bank.record("A", "B", {"points": 30, "passing": None})
    bank.finish_week()
    state = bank.snapshot("A")
    assert state["full_passing_raw_for_games"] == 1
    assert state["full_passing_raw_for_observed"] == 300
    assert state["full_points_raw_for_games"] == 2
    assert state["full_points_raw_for_observed"] == 35
    assert state["full_passing_off_residual_observed"] == 85


def test_expectation_depends_on_both_sides_and_residual_sign():
    bank = FullStatProfiles(["A", "B", "Strong", "Weak"])
    bank.defense["Strong", "passing"] = 60
    bank.defense["Weak", "passing"] = -60
    bank.off["A", "passing"] = 30
    assert bank.predict("A", "Strong", "passing") == 185
    assert bank.predict("A", "Weak", "passing") == 305
    bank.record("A", "Strong", {"passing": 300})
    bank.record("B", "Weak", {"passing": 300})
    assert bank.audit[0]["residual"] > bank.audit[1]["residual"]
    bank.finish_week()
    assert bank.snapshot("Strong")["full_passing_def_residual_observed"] == -115


def test_every_requested_stat_has_both_sides_and_real_counts():
    bank = FullStatProfiles(["A", "B"])
    bank.record("A", "B", {m: center + sd for m, (center, sd, _) in METRICS.items()})
    bank.finish_week()
    for metric in METRICS:
        assert bank.snapshot("A")[f"full_{metric}_raw_for_games"] == 1
        assert bank.snapshot("B")[f"full_{metric}_raw_against_games"] == 1


def test_grid_is_bounded_and_contains_actual_parameter_combinations():
    quick, expanded = candidate_grid(), candidate_grid(True)
    assert len(quick) == 16
    assert len(expanded) == 88
    assert {p[3]["max_depth"] for p in expanded if p[0] == "xgboost" and p[3]} == {2, 4}
    assert all("market_home_margin" not in features for _, _, features, _ in expanded)


@pytest.mark.parametrize("family", ["bayesian_ridge", "xgboost", "hist_gradient_boosting"])
def test_families_fit_imputer_and_scaler_on_training_only(family):
    train = pd.DataFrame({"x": np.arange(50.), "missing": np.nan, "target": np.arange(50.) * 2})
    fit = _fit(train, ["x", "missing"], "target", family, 42)
    assert fit[3] == ["x"]
    assert fit[2].mean_[0] == 24.5
    assert np.isfinite(_predict(fit, pd.DataFrame({"x": [1000., np.nan]}))).all()


@pytest.mark.parametrize("family", ["bayesian_ridge", "xgboost", "hist_gradient_boosting"])
def test_all_family_contributions_reconstruct_prediction(family):
    frame = pd.DataFrame({"x": np.arange(40.), "z": np.sin(np.arange(40.)), "target": np.arange(40.)})
    model, imputer, scaler, features = _fit(frame, ["x", "z"], "target", family, 42)
    bundle = GameModelBundle(model, model, imputer, features, 0., 1., -1., 1., {},
                             scaler=scaler, model_family=family)
    np.testing.assert_allclose(bundle.predict_margin_contributions(frame).sum(axis=1),
                               bundle.predict_margin(frame), atol=1e-4)


def test_full_feature_pipeline_pregame_and_live_parity():
    from test_features import sample_games, sample_rankings

    games = sample_games()
    features, weekly = build_sequential_features(games, sample_rankings())
    changed = games.copy()
    changed.loc[changed.game_id == 2, "home_points"] = 99
    altered, _ = build_sequential_features(changed, sample_rankings())
    full_names = [n for n in features if n.startswith("full_")]
    pd.testing.assert_frame_equal(features[full_names], altered[full_names])
    state = weekly[weekly.game_week == 1].set_index("team")
    live = matchup_features_from_snapshots(state.loc["Beta"], state.loc["Alpha"], neutral_site=False, week=2)
    for name in full_names:
        assert live[name] == pytest.approx(features.loc[features.game_id == 2, name].iloc[0])
    audit = pd.DataFrame(features.attrs["full_stat_audit"])
    assert set(audit.stat) == {"points", "margin"}
