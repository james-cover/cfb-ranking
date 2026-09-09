# Apply the v0.6.1 box-score repair

Extract this patch into the existing `cfb-ranking` directory. Keep the virtual
environment active. No dependency installation is required.

```bash
unzip -o ~/Downloads/cfb-box-scores-v0.6.1-patch.zip -d .
cfb box-scores --start-year 2014 --end-year 2026
cfb build-features
```

Do not run `cfb train` yet. First confirm `build-features` reports nonzero values
for `rushing_stat_rows`, `passing_stat_rows`, and `possession_stat_rows`. Then
send that output back for review before we build and compare the Bayesian model.

The box-score command only requests `/games/teams`; it does not redownload the
other API feeds. It updates `data/raw/team_game_stats.csv` while retaining any
previously downloaded game/team rows. Invalid or unavailable weeks are skipped
individually. If every request fails, the warning now includes the first actual
API response so authentication, subscription, or parameter problems are visible.
