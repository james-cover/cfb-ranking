# Apply the v0.9.0 Bayesian expectation patch

No API download or dependency installation is required.

```bash
unzip -o ~/Downloads/cfb-bayesian-expectations-v0.9.0-patch.zip -d .
cfb build-features
cfb train
cfb predict
cfb serve
```

This version replaces opponent rolling-average subtraction with three online
Bayesian matchup models: points, rushing yards, and passing yards. Each model
maintains separate partially pooled offense and defense effects for every team:

```text
expected stat = FBS baseline + team offense − opponent defense
```

After a game, the observed result updates the team's offense and the opponent's
defense according to their posterior uncertainty. Previous-season ratings carry
forward at 65%, while Elo supplies the remaining preseason prior. The generated
training CSV now includes the exact pregame expectations for auditing.

The final independent model receives both the latent offense/defense ratings and
the actual-minus-expected residual history. It still excludes market spreads.
Bayesian Ridge and XGBoost compete across complete future seasons. The 0.05-MAE
preference applies only when choosing Bayesian Ridge over XGBoost; it no longer
discards the better Elo feature set merely because it has one additional input.

The patch passed 23 tests and static analysis.
