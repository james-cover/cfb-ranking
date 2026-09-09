# Apply the v0.6.0 model patch

This is an update to your uploaded project, not a new installation. It contains
source code, tests, the existing HTML with a small bet-slip compatibility fix,
and this review. No data, credentials, trained models, cached logos, environment,
or dependency bundles are included. Your layout and logo styling are unchanged.

Stop the local server and make a backup of your project first. With the existing
virtual environment active, run these commands from your **cfb-ranking** directory:

```bash
unzip -o ~/Downloads/cfb-model-v0.6.0-patch.zip -d .
cfb build-features
cfb train
cfb predict
cfb serve
```

Adjust the ZIP path if your browser saved it elsewhere. Extract into the existing
project directory; do not replace that entire directory in Finder. The ZIP updates
only the included files. Your editable installation reads the updated source;
no pip installation or new bootstrap is required.

`cfb go` now runs build-features, train, and predict without invoking pip.
Prediction may still download any missing team logos using your existing logic.

Feature and model schema checks intentionally reject old model artifacts until
the rebuild and training steps finish. Training overwrites derived CSV/model
outputs as usual. Keep your backup if you want to compare against the old run.

## What to send back

After training, send the terminal output plus these files from `models/`:

- `game_model_evidence.csv`
- `game_model_backtest_predictions.csv`
- `edge_model_evidence.csv`
- `edge_model_backtest_predictions.csv`
- `feature_coverage.csv`

Also send `data/predictions/current_rankings.csv` if the new rankings still look
wrong. Your configured predictions directory may differ; use the existing one.
Never include your API key or environment file.

`trained_not_validated` is an honest result, not a crash: the ATS model was fitted
but failed the release guardrails. Independent rankings and winner predictions
remain available. Spread EV stays blank; price selections and payouts still work.

## Local verification

The patch passed 19 Python tests, including actual XGBoost training, calibration,
serialization, neutral-field rankings, TreeSHAP additivity, live feature delivery,
push handling, and the EV release gate. The shipped JavaScript parsed and passed
bet-slip checks for missing probabilities and same-game probability suppression.
The CLI help command and lint checks passed. These are correctness checks, not
football accuracy results. Tests ran on Linux/Python 3.12, not your Mac/Python 3.14.

To rerun tests if your environment already has pytest and Node installed:

```bash
python -m pytest tests -q
node tests/test_frontend_slip.cjs
```
