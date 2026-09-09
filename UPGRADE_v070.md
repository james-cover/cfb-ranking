# Apply the v0.7.0 Bayesian + box-score join repair

Extract the patch into the existing `cfb-ranking` directory with the virtual
environment active. No package installation or API bootstrap is required.

```bash
unzip -o ~/Downloads/cfb-bayesian-v0.7.0-patch.zip -d .
cfb box-scores --start-year 2014 --end-year 2026
cfb build-features
cfb train
cfb predict
cfb serve
```

The box-score download must be repeated. CFBD's current response calls the team
field `team`; v0.6.1 looked only for the older `school` field. That caused both
teams in a game to share a null upsert key and left all derived passing, rushing,
and possession features constant. v0.7.0 reads either schema and removes those
invalid legacy rows.

Verify these checkpoints before training:

- `cfb box-scores` should return substantially more than the previous 10,468
  rows because most games now retain both named teams.
- `cfb build-features` should finish and report a nonzero
  `box_score_game_rows`. It now stops with an explicit error if passing,
  rushing, or possession features are constant.

The independent margin/ranking model is now Bayesian ridge regression. Its
candidate inputs are intentionally limited to points scored/allowed, rushing
and passing yards gained/allowed, turnover margin, average possession time,
Elo strength of schedule, home field, and optionally team Elo. Chronological
season validation chooses between the with-Elo and without-Elo variants. The
separate betting-edge model remains XGBoost and still suppresses live EV unless
its held-out validation thresholds pass.

After training, review:

- `models/game_model_evidence.csv`
- `models/game_model_fold_evidence.csv`
- `models/game_model_feature_importance.csv`
- `models/game_model_backtest_predictions.csv`

The patch passed 22 tests, including a regression test that confirms current
CFBD team names join to games and affect the following game's pregame passing,
rushing, and possession features.
