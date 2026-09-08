import pandas as pd

from cfb_rankings.game_model import consensus_current_lines


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
    assert consensus["market_home_margin"] == 3.5
    assert consensus["sportsbooks"] == "A, B"

