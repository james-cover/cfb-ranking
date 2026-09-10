# Full-stat opponent adjustment and model search — v0.10.0

This is a source-code patch for your working v0.9 project, not a replacement
project. It does not contain API keys, downloaded data, trained models, or frontend
files. It makes no API requests and introduces no new package requirements beyond
your existing scikit-learn/XGBoost installation (threadpoolctl comes with sklearn).

## Install and get new results on your Mac

Stop the running web server with Control-C. Download the patch zip. In the terminal
already open in your `cfb-ranking` directory, with `(.venv)` active:

```bash
cp -R src src-backup-before-v0100
cp -R models models-backup-before-v0100
unzip -o ~/Downloads/cfb-full-stats-v0.10.0-patch.zip -d .
cfb build-features
cfb train --tune
cfb predict
cfb serve
```

Use unused backup directory names if those backups already exist. The backup
commands preserve your previous source and fitted models. Run each command after
the preceding command succeeds. `unzip -o` replaces ONLY the files listed in this
patch; your frontend, `.env`, and raw CSVs stay in place. Do not replace the entire
project or entire `src` folder using Finder's Replace-folder option.

Your existing editable install should pick up the source immediately. No `pip`
command, historical bootstrap, or box-score download is needed. If the downloaded
zip has a different filename/location, adjust the unzip path. If `--tune` is not
recognized, run `python -m src.cfb_rankings.cli train --tune` from this directory;
that tests the patched source rather than a different installed command.

`cfb train` without `--tune` is a shorter 16-candidate comparison. `--tune` runs
88 candidates, normally across four season-forward folds (352 selection fits,
plus calibration, benchmark, and final fitting). Runtime depends on your Mac and
dataset. Each independent-model fit prints its candidate, parameters and season.
It is CPU-only, with one numerical worker to limit CPU contention.
Progress CSVs are saved after each fold. Interrupting preserves those diagnostics,
but restarting the search recomputes it; checkpoint-resume is not implemented.

## What is added

All of your requested categories are represented on both sides of the matchup:

| Metric | Unit / treatment |
| --- | --- |
| Points scored / allowed | Points per observed game |
| Scoring margin | Team points minus opponent points, per game |
| Passing gained / allowed | Net passing yards per observed game |
| Rushing gained / allowed | Rushing yards per observed game |
| Total yards gained / allowed | Yards per observed game |
| Yards per pass; yards per rush | Mean of observed game-level rates; not divided by games again |
| Turnover margin | Opponent turnovers minus own turnovers per observed game |
| Time of possession | Minutes per observed game, parsed from each team's own value |
| Third-down percentage | Mean of observed game conversion rates, on a 0–1 scale |
| First downs | First downs per observed game |
| Penalty yards | Penalty yards per observed game |
| Offensive / defensive PPA | Mean of observed game-level PPA |
| Offensive / defensive success rate | Mean of observed game-level rates, on a 0–1 scale |

Rates are equal-game averages, not season attempt-weighted rates. Defense/against
means the opponent's corresponding production. For penalties, turnovers and
possession, these are opponent effects, not necessarily literal defensive skill.
Coefficient signs are learned; a higher penalty rating is not called better play.
Missing values never add a zero or increment that stat's denominator. A zero-
attempt third-down rate is missing, not 0%. No opponent possession time is fabricated
as `60 - own TOP`. Missing advanced offense can fall back to the opponent's reported
defense, without double-counting the same observation.

Each metric has six profile fields: stabilized raw for/against, fitted offense/
defense effects, and offense/defense pregame residuals. The adjusted candidates use
the four adjusted fields, not raw means. Matchup inputs are home-minus-away
differences plus home field; a separate candidate also includes Elo and SOS.

## What opponent adjustment actually does

Before the week's games, each metric has an expectation:

`expected(team versus opponent) = metric center + fitted team effect - fitted opponent effect`

After results are available, the prediction error updates both teams' effects
with a Gaussian-filter approximation and shrinkage. The archived residual is
`actual - pregame expected`; the opponent's suppression residual reverses that sign.
It is NOT actual minus the opponent's raw season average, and it does not multiply
yards by an arbitrary Elo ratio. All expectations for a week are frozen before
that week's updates. Profiles carry into the next season only for eligible FBS
teams, with regression toward the initial center. Current lower-division results
can inform opponents, but lower-division teams are not ranked as FBS teams.

Important limitations: this is a diagonal approximate Bayesian filter, not an
exact joint posterior. Metric centers, noise/prior scales and the .65 carryover
are explicit assumptions in `full_stats.py`, not fitted on future data. The current
grid tunes the second-stage margin predictor, not these expectation-filter priors.
The expectation stage does not yet model injuries, rosters, pace or venue-specific
stat effects. The final margin model does include home field.

Raw/residual features use a four-observation prior to stabilize tiny samples.
`*_observed` columns retain the unshrunk averages, and `*_games` columns retain
the real denominators. Fitted rating effects already have Bayesian shrinkage.
Model scaling and median imputation are fitted separately inside each training fold.

## Search and evaluation

The search compares Bayesian Ridge, XGBoost and histogram gradient boosting across
the full adjusted feature set, one without the redundant margin/total-yards
groups, and full adjusted plus Elo/SOS. Raw-stat and previous-core feature sets
and an Elo/home-field baseline are also evaluated. The full-feature model is not
automatically forced to win. Unavailable and constant training columns are dropped;
check coverage before interpreting which stats actually participated.

- XGBoost grid: depth 2/4, learning rate .03/.08, minimum child weight 10/30,
  and L2 regularization 5/20. Other settings, including 400 trees, remain fixed.
- Histogram boosting: depth 2/4, learning rate .03/.08, L2 .1/10; 250 iterations.
  Automatic random early-stopping splits are disabled.
- Bayesian Ridge: coefficient-precision prior shape (`lambda_1`) 1e-6/.001/1.
  Bayesian Ridge has no branch depth or boosting learning rate.

With 2014–2025 history and current season 2026, candidates train on seasons before
each of 2020–2023, then predict that season. Selection uses mean fold margin MAE,
preferring Bayesian Ridge if within .05 points of the best. 2024 calibrates win
probabilities; 2025 provides a development benchmark. It has already been inspected
repeatedly in this project and is NOT an untouched test. Future games must confirm
any apparent improvement. Final live fitting uses all completed available games.

ATS hit rate and ROI are diagnostics, not the selection objective. Market lines
are excluded from the independent model's inputs. The separate market-aware edge
model and its validation gate remain in place; tuning does not unlock unvalidated
live EV claims. Historical market snapshots may differ from prices you could have
bet at a particular time. Totals remain unvalidated for betting and are not the
target of this search. The AP model workflow is unchanged.

## Files to inspect / send back

After building features:

- `data/processed/full_stat_coverage.csv`: real observation counts by season and
  metric, including zero-count missing metrics.
- `data/processed/stat_expectation_audit.csv`: team, opponent, actual stat, pregame
  expected stat and residual for every usable observation.

After training:

- `models/game_model_evidence.csv`: champion benchmark and all candidate summaries.
- `models/game_model_fold_evidence.csv`: each candidate's season results/parameters;
  includes market-vs-model MAE on the same lined games.
- `models/feature_coverage.csv`: available/constant columns and which the champion uses.
- `models/game_model_feature_importance.csv`: signed standardized coefficients for
  Bayesian Ridge, gain for XGBoost, or training-set permutation importance for
  histogram boosting. These methods are NOT directly comparable or causal effects.
- `models/game_model_metadata.json`: selected features, parameters and season splits.
- `models/game_model_backtest_predictions.csv`: individual benchmark predictions.
- `models/game_model_search_progress.csv`: completed selection folds, also available
  during the run.

Frontend files are untouched. Full-stat contributions feed the existing efficiency
group. Bayesian contributions are additive coefficients; XGBoost uses TreeSHAP;
histogram boosting uses an order-dependent additive path explanation, NOT SHAP.
The ranking output includes `contribution_method` to make that distinction explicit.

The implementation has synthetic regression tests; the repaired full dataset is
on your Mac, so this patch is not evidence of improved real-game accuracy yet.

Implementation references: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[Bayesian Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html),
[histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
