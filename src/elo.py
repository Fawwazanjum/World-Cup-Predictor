"""Team Elo rating computation for international football (1990-present).

Elo is computed chronologically over ALL matches (including friendlies) from
1990 onwards so ratings stay current between competitive fixtures. Friendlies
use a lower K-factor (20) so they shift ratings gently; major tournaments use
a higher K-factor (60) to reflect higher-stakes, higher-quality matches.

The pre-match Elo of both teams is recorded for every match so it can be used
as a historically-valid training feature — unlike the current FIFA ranking
snapshot which is anachronistic when applied to historical matches.
"""

import numpy as np
import pandas as pd

from src import config

DEFAULT_ELO = 1500   # starting rating for any team not yet seen
HOME_ADVANTAGE = 100 # Elo points added when team plays at home (non-neutral)

# K determines how much a single result shifts ratings.
# Higher K = faster adaptation but more volatile.
_MAJOR_TOURNAMENTS = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América", "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
    "CONCACAF Championship",
    "Olympic Games",
}


def _k_factor(tournament: str) -> float:
    if tournament == "Friendly":
        return 20
    if tournament in _MAJOR_TOURNAMENTS:
        return 60
    return 40


def compute_elo(
    df: pd.DataFrame,
    start_date: str = config.TRAINING_START_DATE,
) -> tuple[pd.DataFrame, dict]:
    """Compute Elo ratings for every match from start_date onwards.

    Called with two different start dates for two different purposes:
    - Training  (start_date = TRAINING_START_DATE = 1990): builds Elo across
      35 years so the model learns the relationship between Elo gap and
      match outcome from the full historical record. Keeps accuracy at ~60%.
    - Simulation (start_date = ELO_START_DATE = 2020): uses only recent
      results so the *current* ratings reflect today's squad quality rather
      than historical dominance from past generations. Reduces polarisation.

    Returns
    -------
    elo_df : DataFrame with columns (date, home_team, away_team,
             elo_home, elo_away, elo_diff) — one row per match,
             all values are PRE-MATCH so they are valid training features.
    final_ratings : dict {team: elo} reflecting the state after all matches,
             used by simulate.py to look up current team strength.
    """
    subset = (
        df[df["date"] >= start_date]
        .sort_values(["date", "home_team"])
        .reset_index(drop=True)
    )

    ratings: dict[str, float] = {}
    records = []

    for _, row in subset.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_h = ratings.get(home, DEFAULT_ELO)
        r_a = ratings.get(away, DEFAULT_ELO)

        # Apply home advantage to the expected-score calculation only;
        # the stored rating itself never carries the venue bonus.
        r_h_adj = r_h if row["neutral"] else r_h + HOME_ADVANTAGE

        e_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h_adj) / 400.0))
        e_a = 1.0 - e_h

        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            s_h, s_a = 1.0, 0.0
        elif hs == as_:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        k = _k_factor(row["tournament"])

        # Record PRE-match ratings
        records.append({
            "date":       row["date"],
            "home_team":  home,
            "away_team":  away,
            "elo_home":   r_h,
            "elo_away":   r_a,
            "elo_diff":   r_h - r_a,
        })

        # Update ratings
        ratings[home] = r_h + k * (s_h - e_h)
        ratings[away] = r_a + k * (s_a - e_a)

    elo_df = pd.DataFrame(records)
    return elo_df, ratings
