"""Simulate remaining World Cup 2026 fixtures through the knockout bracket."""

import pandas as pd

from src import config


def predict_match_probabilities(model, home_team: str, away_team: str, neutral: bool) -> dict:
    """Return {home_win, draw, away_win} probabilities for a single fixture."""
    raise NotImplementedError


def simulate_knockout_bracket(model, fixtures: pd.DataFrame, n_simulations: int = 10_000) -> pd.DataFrame:
    """Monte Carlo simulate the bracket forward, resolving TBD slots round by round.

    Returns a DataFrame of each team's probability of reaching each stage
    (Round of 16, QF, SF, Final, Champion).
    """
    raise NotImplementedError


if __name__ == "__main__":
    pass
