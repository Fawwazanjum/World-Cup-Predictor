"""Load and clean raw historical match / ranking CSVs into tidy DataFrames."""

import pandas as pd

from src import config


def load_results() -> pd.DataFrame:
    """Load historical international match results.

    Expects columns similar to the martj42/international_results dataset:
    date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral.
    """
    df = pd.read_csv(config.RESULTS_CSV, parse_dates=["date"])
    return df


def split_played_and_pending(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split results into matches with a known score and matches still to be played.

    pandas reads the literal string "NA" in home_score/away_score as NaN by
    default, so a played match is simply one where both scores are non-null.
    The "pending" half is currently the 18 unplayed WC2026 group games — these
    are what simulate.py draws random outcomes for, not what train.py trains on.
    """
    has_score = df["home_score"].notna() & df["away_score"].notna()
    played = df[has_score].copy()
    pending = df[~has_score].copy()
    return played, pending


def load_rankings() -> pd.DataFrame:
    """Load the current FIFA ranking snapshot (June 2026) for the 48 WC2026 teams.

    Columns: rank, team, points. This is a single point-in-time snapshot, not
    a per-match historical lookup, so features.py applies it uniformly across
    all training rows rather than joining on match date.
    """
    df = pd.read_csv(config.RANKINGS_CSV)
    return df


def load_fixtures() -> pd.DataFrame:
    """Load the remaining World Cup 2026 fixtures / current bracket state.

    Expects columns: date, stage, home_team, away_team, venue, neutral.
    For TBD bracket slots (e.g. "Winner R16-3"), keep as a placeholder string;
    simulate.py resolves these as earlier rounds are simulated.
    """
    df = pd.read_csv(config.FIXTURES_CSV, parse_dates=["date"])
    return df
