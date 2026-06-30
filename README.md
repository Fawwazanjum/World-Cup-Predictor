# World Cup 2026 Match Predictor

Predicts outcomes (home win / draw / away win) for international football matches
and simulates the remaining FIFA World Cup 2026 bracket.

## Project structure

```
data/
  raw/         <- put downloaded CSVs here (see "Data needed" below)
  processed/   <- generated feature tables (created by the pipeline)
models/        <- saved trained models
src/
  config.py     <- paths & constants
  data_loader.py<- read raw CSVs into DataFrames
  features.py   <- recent form, head-to-head, ranking gap, venue features
  train.py      <- logistic regression / XGBoost training + evaluation
  simulate.py   <- Monte Carlo bracket simulation
notebooks/     <- exploratory analysis (optional)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data needed (place in `data/raw/`)

1. `results.csv` — historical international match results (date, home_team,
   away_team, scores, tournament, neutral venue flag).
2. `fifa_ranking.csv` — current FIFA ranking snapshot (June 2026), rank/points
   for the 48 World Cup 2026 teams only.
3. `wc2026_fixtures.csv` — current/remaining World Cup 2026 fixtures and bracket
   state, since the tournament is in progress and no static dataset covers it.

See the assistant's message for exact source recommendations and expected columns.
