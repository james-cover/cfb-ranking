import pandas as pd

from cfb_rankings.storage import read_csv, upsert_csv


def test_upsert_replaces_matching_key(tmp_path):
    path = tmp_path / "games.csv"
    upsert_csv(pd.DataFrame([{"id": 1, "score": 7}]), path, ["id"])
    upsert_csv(pd.DataFrame([{"id": 1, "score": 14}]), path, ["id"])
    result = read_csv(path)
    assert len(result) == 1
    assert result.loc[0, "score"] == 14
