# College Football Ranking Lab

A live, CSV-backed system that produces:

1. The current official AP Top 25.
2. A machine-learning forecast of the next AP Top 25.
3. An independent Bayesian ranking of FBS teams.
4. Predicted margins and scores for upcoming games, compared with current consensus spreads.
5. Locally cached team logos and official team colors for the included HTML frontend.
6. Full-season team schedules, selectable spread and moneyline props, model EV, and a local bet-slip calculator.

The sportsbook spread is **never an input** to the independent model. It is joined only after
prediction so the model-versus-market comparison remains honest.

## Production frontend/API connection

The backend serves the HTML and JSON API from the same process. Start it with:

```powershell
cfb-rankings serve
```

Then open `http://127.0.0.1:8000`. The frontend automatically loads the latest backend output;
CSV importing remains available only as a fallback. Available endpoints are:

- `GET /api/health`
- `GET /api/rankings`
- `GET /api/games`
- `GET /api/schedule`
- `GET /api/teams/{team}`
- `GET /team_logos/{filename}`

CSV responses are cached in memory until the source file's modification time or size changes.
Because prediction files are written atomically, a live refresh cannot expose a partial CSV.
For deployment, run this service behind the site's HTTPS reverse proxy rather than exposing the
Uvicorn port directly.

## How the models work

### AP poll forecast

The training unit is one team in one historical AP poll week. Every row contains only
information available before that poll. The pipeline performs rolling-season validation of:

- XGBoost LambdaMART (`rank:ndcg`)
- LightGBM LambdaRank
- XGBoost AP-points regression
- Histogram gradient-boosted AP-points regression
- The prior poll as a persistence baseline

The champion is selected from out-of-time evidence using NDCG@25, top-25 membership F1,
rank error, and rank correlation. Results are written to `models/ap_model_evidence.csv`; the
dashboard exposes them rather than silently hard-coding a favored algorithm.

### Independent ranking and game forecast

The independent model is Bayesian ridge regression trained to predict the home team's final
scoring margin from pregame information. Candidate basic-stat feature sets are selected using
complete future seasons as validation sets, with a simpler model preferred when validation MAE
is within 0.05 points.

The independent model deliberately uses readable inputs: points scored and allowed, rushing
yards gained and allowed, passing yards gained and allowed, turnover margin, average time of
possession, Elo strength of schedule, and home field. A second candidate may also include team
Elo; season-forward validation decides whether it earns its place. The sportsbook line is never
an input. The separate betting-edge model remains XGBoost and is not shown as validated unless
its held-out evidence clears the safety thresholds.

To rank teams, the trained model predicts every FBS-versus-FBS matchup on a neutral field. A
team's independent rating is its average predicted margin across those opponents. This makes
the ranking a direct model output, not a manually weighted polynomial score.

Early-season rate statistics use a four-game neutral prior before entering the model. The raw
record and displayed statistics remain unchanged, but one blowout cannot masquerade as a
stable full-season average. Elo persists only for teams that were FBS in the prior season;
new FBS members do not inherit a lower-division rating. Independent-ranking exports include
linear contribution groups so each team's detail page shows what raised or lowered its rank.

## Leakage policy

The project deliberately prevents these common backtesting errors:

- A game uses team state from **before kickoff**, never end-of-season averages.
- A poll forecast uses games through the preceding football week.
- Ranked wins use the poll available when the game was played.
- The betting spread is excluded from independent-model features.
- Model selection validates on later, completely unseen seasons.
- Betting lines are stored with retrieval timestamps rather than overwritten.

## Locked-down Windows setup (recommended)

The Windows portable bundle includes all Python 3.14 dependency wheels. It does not require
administrator rights, a compiler, editable installation, or internet access. From PowerShell:

```powershell
.\setup-offline.ps1
.\cfb.ps1 bootstrap --start-year 2014
.\cfb.ps1 audit
.\cfb.ps1 build-features
.\cfb.ps1 train
.\cfb.ps1 predict
.\cfb.ps1 dashboard
```

Dependencies are placed under `%LOCALAPPDATA%\CFBRankingRuntime\py314`, outside OneDrive and
outside the system Python installation. You do not activate a virtual environment.

## Standard Python setup

From PowerShell in this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

Put the CollegeFootballData key in `.env`:

```dotenv
CFBD_API_KEY=your_real_key_here
CFB_CURRENT_SEASON=2026
CFB_TIMEZONE=America/Chicago
```

The `.env` file is ignored by Git and must not be emailed or committed.

## First historical build

```powershell
cfb-rankings bootstrap --start-year 2014
cfb-rankings audit
cfb-rankings build-features
cfb-rankings train
cfb-rankings predict
cfb-rankings dashboard
```

The historical bootstrap may take several minutes. Optional advanced-stat or line endpoints
are allowed to fail cleanly if the API key's plan does not include them; the core games,
rankings, and team data are required.

## Weekly/live refresh

After the initial model training:

```powershell
.\scripts\update_current.ps1
```

This refreshes the current season, audits the CSVs, rebuilds pregame features, and regenerates
rankings and upcoming-game predictions. It does not retrain on every refresh. To deliberately
reselect and retrain both models:

```powershell
.\scripts\retrain.ps1
```

Run the refresh after the final relevant game and before the AP poll is released to create a
true next-poll prediction. Running it after release creates a nowcast that can be compared with
the newly observed poll.

## Outputs

All tables are ordinary CSV files:

```text
data/
├── raw/
│   ├── games.csv
│   ├── rankings.csv
│   ├── teams.csv
│   ├── team_game_stats.csv
│   ├── advanced_game_stats.csv
│   └── betting_lines.csv
├── processed/
│   ├── data_audit.csv
│   ├── game_training_data.csv
│   ├── team_week_features.csv
│   └── ap_training_data.csv
└── predictions/
    ├── actual_ap_poll.csv
    ├── predicted_ap_poll.csv
    ├── independent_rankings.csv
    ├── upcoming_game_predictions.csv
    ├── season_schedule.csv
    ├── current_rankings.csv
    ├── ap_prediction_history.csv
    └── game_prediction_history.csv
frontend/
├── index.html
└── team_logos/
    └── <team_id>.png
```

The two history files are append-only prediction snapshots. Each AP snapshot is marked
`pre_release_forecast` or `nowcast_after_release`, preventing a post-release refresh from being
mistaken for a genuine forecast.

Trained model binaries and validation evidence are placed in `models/`.

## Team logos and colors

The CFBD teams response already supplies each team's logo URLs, primary color, and alternate
color. During `predict`, the backend downloads the current-season logos once into
`frontend/team_logos/` and reuses the cached files on later refreshes. Logo download failures
are logged but never block rankings or game predictions.

The ranking CSVs include `team_id`, `logo_url`, `logo_path`, `team_color`, and `alt_color`.
Upcoming-game predictions include the same fields with `home_` and `away_` prefixes. Open
`frontend/index.html` directly; it uses the local image first, the remote URL as a fallback,
and team initials if neither image is available.

### Repair or refresh team box scores only

If `team_game_stats.csv` is empty, fetch just the missing box-score feed without
redownloading rankings, lines, or the schedule:

```bash
cfb box-scores --start-year 2014 --end-year 2026
cfb build-features
```

The command preserves successful weeks when CFBD rejects an unavailable week.
`build-features` will stop instead of silently training unless rushing yards,
passing yards, and possession time are present. Its output reports the usable
row count for each required category.

## Spread/moneyline EV and bet slip

For each game, the backend keeps every provider's most recent quote, uses the median spread,
and selects the best available home and away moneyline. Expected value per $100 is calculated
for all four selectable outcomes: both teams' spreads and both teams' moneylines. The Highest EV
view places those prices directly beside each team and sorts the upcoming slate by the strongest
available model EV.

CFBD supplies spread points but usually does not supply spread juice. When exact spread prices
are absent, the backend uses a clearly marked standard `-110` assumption. Exact provider spread
prices are retained and used automatically if `homeSpreadOdds` and `awaySpreadOdds` are present.

The frontend bet slip can mix spread or moneyline parlay legs and single bets. It calculates combined American odds,
total wager, return if the picks hit, and model-estimated EV. Slip data stays in browser local
storage and never places a wager. Parlay probability multiplies the leg probabilities and is
explicitly labeled as an independence approximation because correlated outcomes can invalidate it.

## Spread conventions

The model stores `model_home_margin` as:

```text
predicted home score - predicted away score
```

Positive means the home team is favored. CFBD's numeric sportsbook spread is a home-team
handicap, so a spread of `-3.5` becomes a market-implied home margin of `+3.5`.

```text
model_edge_home = model_home_margin - market_home_margin
```

Positive edge favors the home team against the line; negative edge favors the away team. Line
timestamps and contributing providers appear beside every comparison. These estimates are
uncertain and are intended for model evaluation, not as guaranteed betting outcomes.

## Tests

```powershell
pytest
ruff check src tests
```
