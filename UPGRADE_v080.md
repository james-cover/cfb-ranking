# Apply the v0.8.0 opponent-adjustment patch

No API download or dependency installation is required. Extract the patch into
the existing project, then rebuild the features and models:

```bash
unzip -o ~/Downloads/cfb-opponent-adjusted-v0.8.0-patch.zip -d .
cfb build-features
cfb train
cfb predict
cfb serve
```

For every game, v0.8.0 calculates pregame-only residuals for:

- points scored versus that opponent's expected points allowed;
- points prevented versus that opponent's expected scoring;
- rushing yards gained/allowed versus that opponent's expectation;
- passing yards gained/allowed versus that opponent's expectation.

The four-game priors stabilize early-season expectations. Turnover margin,
average possession time, Elo SOS, home field, and optionally team Elo remain
readable supporting inputs. Market lines are excluded.

Bayesian Ridge and XGBoost now compete on the identical corrected feature set
over complete future-season folds. The output identifies both
`independent_model` and `independent_feature_set`. The final reported test season
does not participate in selection.

The patch passed 22 tests and static analysis, including assertions that the
first game has no prior-performance residual and that its box score changes the
following game's opponent-adjusted passing and rushing inputs.
