# CFB model review — September 9, 2026

## Bottom line

XGBoost is still a reasonable approach. The immediate problems are data handling,
training-to-live consistency, and how betting evidence is measured—not a single
stat whose weight should be manually reduced. This patch repairs those paths and
adds a stricter experiment. It does **not** establish improved football accuracy.

I reviewed the latest CFB source, your ATS log, and all three March Madness
scripts. The latest archives do not include the football training dataset or
`MM_ML.xlsx`, so neither a real-data retraining comparison nor the reported 85%
March Madness performance can be reproduced here.

## Confirmed problems and changes

| Finding | Change |
| --- | --- |
| `predict_upcoming_games` discarded team-stat differential columns before the live edge model ran. The downstream code recreated some inputs with defaults and imputed the rest. | Preserve the full pregame input vector; assert required edge inputs exist; regression-test the actual handoff. |
| The pipeline updated spread probabilities/EV after choosing `best_prop_*`, leaving the displayed best play stale. | Recompute best plays after all probability updates. |
| Pushes were counted as ATS losses; the scan label implied binary cover correlation although it calculated margin residual correlation. | Exclude pushes from settled accuracy, record them separately, and label the residual correlation accurately. |
| Constant features caused NumPy correlation warnings. | Skip constant columns in the scan and remove unusable columns using each training fold only. |
| Missing stat families could become artificial season-progress signals through default values. | Use observed stat-family counts when shrinking inputs. With no observations and no prior, the value stays at its baseline. |
| Historical FCS strength could contaminate an FBS experiment when roster membership was unreliable. | Prefer historical game classifications; train margin models on FBS-vs-FBS targets; retain FBS/FCS games for résumé updates; carry only prior-season FBS profiles. |
| Two-game priors and stronger offseason retention made September results more volatile. Opponent-weight multiplication and division canceled out after one game. | Restore four-game shrinkage and 0.65 offseason Elo retention; carry stabilized previous-season stat profiles; use an additive opponent adjustment and cap recent/quality margins at 35. These are design choices to test, not proven optimal constants. |
| A median handicap could be paired with a sportsbook price quoted for a different handicap. | Keep the median as context, but pair spread prices and model comparisons with an actual provider's handicap. |
| Spread confidence came from a normal-error conversion without independent probability calibration. | Fit a separate market-residual model and chronological calibration; keep its margin separate from the independent model. |

Your latest source already included the API classification-filter and weekly
box-score-fetch changes. Those are not new fixes in this patch. A historical
classification cannot be reconstructed reliably if both game classifications and
season-specific roster data are absent; the fallback is the supplied roster.

The earlier NDSU/Kennesaw outputs are clues, not proof of one causal feature.
Feature importance measures model usage; it is not a fixed scoring weight or a
causal explanation. Team-level TreeSHAP contributions remain available.

## What transfers from March Madness

The scripts use prior seasons for training, a later season for evaluation,
training-only medians, target-column exclusions, and comparatively shallow trees
(depth 2 in the round-of-64 script). Those are useful patterns. Shallow trees and
regularization control complexity; their exact values still need testing.
[XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).

Do not transfer tournament advancement accuracy directly to ATS accuracy: the
targets, eligible teams, sample sizes, and baseline difficulty differ. For weekly
football predictions, every stat must be available **before that matchup**, not
at the end of the season.

The MM files also have audit limitations: `SHEET_NAME` is defined but not passed
to `read_excel`, missing labels are filled with zero, and some printed labels are
left over from the Final Four task. Those do not prove leakage or invalidate your
result, but the workbook and saved predictions are needed to check it properly.

## The new experiment

The AP model family is unchanged. The independent model is XGBoost margin
regression without market lines; rankings average antisymmetrized neutral-field
matchups. Winner probabilities are calibrated from the predicted margins.

For a 2026 run with complete 2014–2025 history:

1. Select compact-vs-full feature sets and depth 2-vs-4 margin candidates using
   expanding-season validation ending in 2020, 2021, 2022, and 2023. Select by MAE.
2. Predict 2024 using training through 2023; use these predictions for probability
   calibration and residual intervals.
3. Fit through 2024 and evaluate 2025 without changing the chosen settings.
4. Refit production estimators on all available completed games, keeping the
   calibration parameters fixed. No uncompleted game targets enter fitting.

Probability calibration uses observations not used to fit the corresponding
base estimator. A separate calibration set matters because good discrimination
does not itself imply accurate probabilities.
[Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

The edge model predicts **actual home margin minus market home margin**. Two
regularized shallow candidates are selected by chronological residual RMSE.
The separate calibration season estimates edge shrinkage and cover probabilities.
Each model's evidence is saved separately, with per-game held-out predictions.
The independent-vs-market comparison includes a matching lined-game subset.

2025 is held out from selection **in this run**, not a pristine unseen season
across the project's history. Repeatedly tuning after looking at it would defeat
that separation. Future timestamped, prospective predictions are the next test.

## Reading the ATS log and EV

The largest reported absolute residual correlation is only 0.0214. That is a weak
one-feature linear association, not a 2.14% betting advantage. XGBoost can learn
interactions, but this table does not demonstrate that profitable ones exist.
Searching many features and thresholds on the same games can highlight chance
patterns. Do not select features solely by the scan.

Spread EV is released only if a fixed held-out gate passes: at least 400 decisive
games, Brier below 0.2475, log loss below log(2), at least 100 decisive selections
at a fixed 55% probability cutoff, positive assumed -110 ROI, Wilson lower bound
above 50%, and nontrivial calibration shrinkage. These are conservative screening
rules, not a statistical guarantee or an optimized betting strategy. The Wilson
bound does not account for clustering by team/week or repeated experimentation.

If the gate fails, the trained model remains inspectable but live spread EV is
blank. Integer-spread dollar EV is also withheld because pushes are not modeled
probabilistically. Moneyline EV remains an **exploratory** calculation from
calibrated independent winner probabilities; passing ATS checks does not validate
moneyline profitability. Missing prices are not real offers; the existing -110
fallback remains explicitly marked as an estimate.

## Remaining priorities

- Rebuild and train on your actual data before judging whether rankings improved.
- Verify coverage by season. Family-level availability checks do not substitute
  for auditing missing individual box-score fields or changed API definitions.
- Historical latest-line records are not authenticated pregame quote snapshots.
  Fix a decision time, preserve provider/handicap/price and timestamp, and evaluate
  against the information actually available then. Current-line refreshes alone
  cannot recreate that history.
- Compare live-calibrated probabilities against market-implied probabilities and
  track calibration, ROI, and uncertainty by season, week, spread size, and book.
- Test additional pregame inputs—opponent-adjusted efficiency, returning/QB
  experience, rest/travel and roster availability—one group at a time. Preserve
  publication timestamps and retain only improvements that survive later seasons.
- The total/score model is still descriptive and has not been validated for totals
  betting. A strong margin model does not automatically validate projected totals.
- Keep paper-tracking while gathering prospective evidence; neither accuracy nor
  high displayed EV establishes a reliable profit opportunity.
