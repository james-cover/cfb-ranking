import pandas as pd

from cfb_rankings.game_model import (
    _moneyline_expected_value,
    consensus_current_lines,
)


def test_consensus_uses_latest_line_per_provider_then_median():
    lines = pd.DataFrame(
        [
            {"game_id": 1, "provider": "A", "spread": -2, "fetched_at": "2026-09-08T10:00:00Z"},
            {"game_id": 1, "provider": "A", "spread": -4, "fetched_at": "2026-09-08T11:00:00Z"},
            {"game_id": 1, "provider": "B", "spread": -3, "fetched_at": "2026-09-08T11:00:00Z"},
        ]
    )
    consensus = consensus_current_lines(lines).iloc[0]
    assert consensus["consensus_spread"] == -3.5
    assert consensus["market_home_margin"] in (3, 4)
    assert consensus["market_home_margin"] == -consensus["home_spread"]
    assert consensus["sportsbooks"] == "A, B"


def test_moneyline_aggregation_uses_best_current_price_and_calculates_ev():
    lines = pd.DataFrame(
        [
            {
                "game_id": 1,
                "provider": "A",
                "spread": -3,
                "home_moneyline": -150,
                "away_moneyline": 130,
                "fetched_at": "2026-09-08T11:00:00Z",
            },
            {
                "game_id": 1,
                "provider": "B",
                "spread": -3.5,
                "home_moneyline": -140,
                "away_moneyline": 135,
                "fetched_at": "2026-09-08T11:00:00Z",
            },
        ]
    )
    current = consensus_current_lines(lines).iloc[0]
    assert current["home_moneyline"] == -140
    assert current["away_moneyline"] == 135
    assert current["home_moneyline_sportsbook"] == "B"
    probability = pd.Series([0.65])
    expected_value = _moneyline_expected_value(probability, pd.Series([-140.0]))
    assert round(expected_value.iloc[0], 2) == 11.43


def test_spread_rows_include_both_sides_and_label_standard_price_fallback():
    lines = pd.DataFrame(
        [
            {
                "game_id": 7,
                "provider": "A",
                "spread": -5.5,
                "fetched_at": "2026-09-08T11:00:00Z",
            }
        ]
    )
    current = consensus_current_lines(lines).iloc[0]
    assert current["home_spread"] == -5.5
    assert current["away_spread"] == 5.5
    assert current["home_spread_odds"] == -110
    assert current["away_spread_odds"] == -110
    assert bool(current["spread_odds_estimated"])
    assert current["spread_odds_sportsbook"] == "Standard -110 estimate"
