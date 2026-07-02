# FIFA World Cup 2026 Match Predictor

A machine-learning pipeline that predicts the outcome of FIFA World Cup 2026
matches and simulates the remaining knockout bracket using Monte Carlo methods.

Built as a self-directed project by a second-year mathematics student at UCL.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Data Sources](#data-sources)
5. [Methodology](#methodology)
   - [Feature Engineering](#feature-engineering)
   - [Elo Rating System](#elo-rating-system)
   - [Model Training](#model-training)
   - [Monte Carlo Simulation](#monte-carlo-simulation)
6. [Key Design Decisions & Reflections](#key-design-decisions--reflections)
7. [Current Predictions](#current-predictions)
8. [Updating Results](#updating-results)
9. [Limitations](#limitations)

---

## Project Overview

The goal is to answer: *given the current state of the 2026 World Cup, what is
each remaining team's probability of winning the tournament?*

The pipeline works in four stages:

1. **Feature engineering** — extract meaningful signals from 35 years of
   international football results: recent form, head-to-head records, Elo
   ratings, FIFA rankings, and venue type.
2. **Training** — fit a classifier on ~18,000 competitive historical matches
   (1990–2022) to learn the relationship between those features and match
   outcomes (home win / draw / away win).
3. **Simulation** — for each remaining fixture, the model produces match
   probabilities. A Monte Carlo loop runs 10,000 full bracket simulations,
   sampling outcomes match by match and propagating winners through the
   bracket. Each team's championship probability is the fraction of
   simulations they won.
4. **Live updating** — as real results come in, `results.csv` and
   `wc2026_fixtures.csv` are updated and the simulation is re-run, producing
   revised probabilities that reflect the current tournament state.

---

## Project Structure

```
World Cup Predictor/
├── data/
│   ├── raw/
│   │   ├── results.csv            # 49,000+ international match results (1872–present)
│   │   ├── fifa_ranking.csv       # Current FIFA ranking snapshot (June 2026, 48 teams)
│   │   └── wc2026_fixtures.csv    # Full WC2026 bracket: R32 → Final, with results filled in
│   └── processed/
│       └── wc2026_predictions.csv # Latest simulation output — probability table per team
├── models/
│   ├── best_model.pkl             # Winning model (LR or XGB) with scaler bundled
│   ├── logistic_regression.pkl    # Logistic regression + StandardScaler
│   └── xgboost.pkl                # XGBoost classifier
├── src/
│   ├── config.py                  # All paths and constants in one place
│   ├── data_loader.py             # Load CSVs, split played/pending, filter competitive
│   ├── elo.py                     # Elo rating computation (chronological, parameterised)
│   ├── features.py                # Full feature engineering pipeline
│   ├── train.py                   # Train, evaluate and save classifiers
│   └── simulate.py                # Monte Carlo bracket simulation
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

To train models from scratch:
```bash
python -m src.train
```

To run the simulation:
```bash
python -m src.simulate
```

---

## Data Sources

### `results.csv`
Historical international match results from 1872 to the present day, sourced
from the [martj42/international_results](https://github.com/martj42/international_results)
dataset. Columns: `date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral`. WC2026 results are appended as they are
played, maintaining a live dataset.

### `fifa_ranking.csv`
A single point-in-time snapshot of the official FIFA Men's World Ranking as of
11 June 2026, covering all 48 WC2026 qualified teams. Scraped from
[whereig.com](https://www.whereig.com/football/fifa-world-rankings.html) which
mirrors the official FIFA data. Columns: `rank, team, points`.

**Note:** This snapshot is used as a supplementary feature only. The primary
strength-of-team signal comes from the Elo system (see below), which is
historically accurate unlike this static snapshot.

### `wc2026_fixtures.csv`
The complete WC2026 knockout bracket from Round of 32 through the Final (32
rows). Columns: `match_no, stage, date, home_team, away_team, home_score,
away_score, pens_home, pens_away, winner, venue, neutral`. Results are filled
in as they happen; TBD bracket slots use `Winner M73`-style placeholders that
`simulate.py` resolves dynamically.

---

## Methodology

### Feature Engineering

`src/features.py` builds a feature vector for every historical match, using
only information available *before* that match was played (no data leakage).

| Feature | Description |
|---|---|
| `home_form` | Home team's average points-per-game over their last 5 competitive matches, computed with a shift(1) rolling window so the current match is excluded |
| `away_form` | Same for the away team |
| `h2h_home_win_rate` | Historical win rate for the home team in all prior meetings between these two teams within the last 20 years |
| `h2h_draw_rate` | Historical draw rate for the same matchup |
| `h2h_matches_played` | Number of prior meetings used in the H2H calculation |
| `elo_diff` | Difference in Elo ratings (home minus away) at the time of the match |
| `rank_gap` | FIFA ranking difference (away rank minus home rank) from the June 2026 snapshot |
| `points_gap` | FIFA points difference from the same snapshot |
| `ranking_available` | Binary flag: 1 if both teams are in our 48-team ranking snapshot |
| `is_neutral` | Whether the match is played at a neutral venue |

**Imputation:** Missing values (e.g. ranking features for non-WC teams, H2H
for first-ever meetings) are filled with neutral defaults: 0 for ranking gaps,
dataset base rates for H2H, and 1.5 (midpoint of the 0–3 points scale) for
form on a team's very first-ever match.

### Elo Rating System

`src/elo.py` implements a standard Elo system for international football.

**How it works:** Each team has a rating R. Before a match, the expected
outcome for the home team is:

```
E_home = 1 / (1 + 10^((R_away - R_home_adjusted) / 400))
```

where `R_home_adjusted = R_home + 100` for non-neutral fixtures (home
advantage). After the result, both ratings update:

```
R_new = R_old + K × (actual_score - expected_score)
```

where `actual_score` is 1 for a win, 0.5 for a draw, 0 for a loss.

**K-factors by tournament:**
| Tournament type | K |
|---|---|
| Friendly | 20 |
| Competitive (qualifiers, regional cups, Nations Leagues) | 40 |
| Major tournament (World Cup, Euros, Copa América, AFCON, etc.) | 60 |

**Why Elo over FIFA rankings:** FIFA rankings are published quarterly and use a
proprietary formula. More importantly, we only have a single June 2026
snapshot — applying it to historical training matches is anachronistic
(Argentina's 2026 ranking tells us nothing useful about their 2005 form).
Elo is computed chronologically from match results: a 2005 training row gets
Argentina's Elo as it stood in 2005, built from their actual results up to
that point. Draws are correctly rewarded by opponent quality — Cape Verde
drawing Spain earns far more Elo points than drawing Saudi Arabia.

**The split-window approach:** Using a 35-year Elo window for *training*
means the model learns the true relationship between Elo differences and match
outcomes across a large, well-converged dataset (accuracy 60.3%). Using a
2020+ Elo window for *simulation predictions* means current team ratings
reflect today's squad quality rather than being inflated by historical
dominance from past generations (Spain's Euro 2024 win counts; their 2010
World Cup win does not).

This separation — learn the pattern from 35 years, but estimate current
strength from 6 years — meaningfully reduces prediction polarisation while
keeping the model well-calibrated.

### Model Training

`src/train.py` trains two classifiers on competitive international matches
from 1990 to 2022 (18,163 rows after filtering friendlies):

- **Logistic Regression** (multinomial, L2 regularisation, lbfgs solver) with
  StandardScaler. Performs well when Elo difference is available because Elo
  is approximately linear in the log-odds of winning.
- **XGBoost** (400 trees, max depth 4, learning rate 0.05) without scaling,
  as tree-based models are scale-invariant.

**Train/test split:** strictly time-based at 2022-01-01. Matches from 2022
onwards form the test set (~3,400 rows). A random split would allow the model
to "see the future" (train on a 2024 match to predict a 2020 match), which
would inflate test accuracy and give a false sense of performance.

**Why competitive matches only:** Friendly matches are played with rotated
squads at low stakes — their outcomes carry far less signal about true team
quality. Removing them (18,388 matches) improved accuracy by ~0.8% and
meaningfully improved probability calibration.

**Results:**
| Model | Accuracy | Log-loss |
|---|---|---|
| Logistic Regression | 60.3% | 0.879 |
| XGBoost | 60.2% | 0.882 |

Log-loss is the primary selection criterion: it penalises overconfident wrong
predictions, which matters here because the simulation uses raw probabilities,
not binary predictions. Logistic Regression wins and is saved as
`best_model.pkl`.

**Context:** 60% accuracy on 3-class football prediction (home win/draw/away
win) is competitive with published academic models (typically 55–65%).
A naive baseline of always predicting "home win" scores ~49%.

**Known weakness:** draws are almost never predicted (recall ~1%). This is
a structural problem with discriminative classifiers on imbalanced outcome
distributions. In practice, the simulation still produces draws in group-stage
matches because it samples from the full probability distribution — the model
assigns low but non-zero draw probability to most matches, and with 10,000
simulation runs these accumulate realistically.

### Monte Carlo Simulation

`src/simulate.py` simulates the remaining tournament bracket 10,000 times.

**Setup (run once):**
1. Load `wc2026_fixtures.csv` to identify completed matches (known winners)
   and pending fixtures.
2. Compute current Elo ratings for all 48 WC teams using the 2020+ window.
3. Compute each team's current form (rolling average over their last 5
   competitive matches).
4. Compute head-to-head statistics for all 48×47 team pairs using a
   vectorised groupby operation.

**Per simulation run:**
1. Initialise the bracket with all known real results.
2. For each pending R32 match: call the model to get P(home win), P(draw),
   P(away win). Since knockouts have no draws, normalise to a 2-way contest:
   `P(home win adjusted) = P(home win) / (P(home win) + P(away win))`.
   Sample a winner using this adjusted probability.
3. Propagate winners forward through R16, QF, SF, and Final using the fixed
   bracket wiring derived from the official FIFA schedule.
4. Record the champion.

**Output:** Each team's probability of reaching R16, QF, SF, Final, and
winning the tournament is the fraction of 10,000 simulations in which they
achieved that stage.

---

## Key Design Decisions & Reflections

**Why not a Poisson goals model?** The standard approach in serious football
analytics is to model goals scored by each team as independent Poisson random
variables, then derive win/draw/loss probabilities from the joint distribution.
This naturally handles draw calibration and produces better-calibrated
probabilities. However, it requires restructuring the entire pipeline around
goals rather than outcomes. It represents the most impactful remaining
improvement if this project is extended.

**Why keep FIFA rankings alongside Elo?** The ranking features (rank_gap,
points_gap) are anachronistic for training (see above) but valid for the
current WC2026 predictions, where both teams always have a June 2026 ranking.
XGBoost's tree structure is robust to noisy features — it learns to down-weight
them in training while still potentially extracting value at prediction time.
The `ranking_available` flag tells the model explicitly when to trust these.

**The polarisation problem:** An early version using a 35-year Elo window
produced Argentina at 37.4% and Spain at 26.2% — the two teams together
accounting for 63.6% of simulated championships. This was too polarised
relative to betting markets. The solution (split-window Elo) reduced this to
a more realistic spread, with the top three teams (Spain 25.1%, Argentina
20.2%, France 18.6%) sharing roughly 64% but across three teams rather than
two. Residual polarisation reflects genuine structural quality differences
between teams that Elo captures correctly.

---

## Current Predictions

As of July 2, 2026 (Round of 32 — 10/16 complete):

| Rank | Team | P(R16) | P(QF) | P(SF) | P(Final) | P(Champion) |
|---|---|---|---|---|---|---|
| 1 | Spain | 89.5% | 69.4% | 56.7% | 39.2% | 25.1% → **23.4%** |
| 2 | France | — | 89.0% | 65.9% | 37.2% | **20.4%** |
| 3 | Argentina | 94.7% | 81.1% | 55.0% | 32.3% | **19.3%** |
| 4 | Mexico | — | 52.5% | 40.0% | 19.9% | **8.9%** |
| 5 | Morocco | — | 85.7% | 36.4% | 17.2% | **8.5%** |
| 6 | England | — | 47.5% | 34.8% | 17.1% | **7.7%** |
| 7 | Brazil | — | 63.4% | 22.9% | 8.6% | **3.6%** |
| 8 | Colombia | 89.7% | 47.9% | 14.3% | 6.0% | **2.0%** |
| 9 | Portugal | 69.0% | 20.1% | 11.7% | 5.2% | **1.9%** |
| 10 | Belgium | — | 70.8% | 20.5% | 6.3% | **1.7%** |

(— indicates team is already through to the Round of 16)

Full predictions saved to `data/processed/wc2026_predictions.csv`.

---

## Updating Results

As new results come in, update the two data files and re-run the simulation:

**1. Add real results to `results.csv`:**
Append rows in the existing format:
```
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-07-05,Mexico,England,1,2,FIFA World Cup,Arlington,United States,True
```

**2. Update `wc2026_fixtures.csv`:**
Fill in `home_score`, `away_score`, and `winner` for each completed match.
For penalty shootouts, record the 90-minute score in the score columns and
the advancing team in `winner`; use `pens_home` / `pens_away` for the
shootout score.

**3. Re-run the simulation:**
```bash
python -m src.simulate
```

`simulate.py` reads the bracket state dynamically from `wc2026_fixtures.csv`
each time it runs — no code changes are needed as the tournament progresses.

---

## Limitations

- **Draw calibration:** the model rarely predicts draws (~1% recall). This
  does not affect knockout simulation (no draws there) but means group-stage
  draw probabilities are slightly underestimated.
- **No player-level signal:** squad injuries, individual player form (Haaland,
  Mbappé), and managerial changes are invisible to the model. This partly
  explains gaps versus betting markets (e.g. Norway underrated at 0.7%).
- **Static rankings:** the FIFA ranking snapshot is contemporaneous for WC2026
  predictions but anachronistic for historical training rows — largely
  mitigated by the Elo system which replaces its role.
- **Home crowd effect:** all WC matches are tagged as neutral venue, but the
  USA/Canada/Mexico co-hosting creates real crowd advantages that the model
  cannot distinguish.
- **Cold-start Elo:** all teams begin at 1500 in 2020 under the short Elo
  window. Teams with less international activity since 2020 have ratings closer
  to 1500 than their true strength would suggest.
