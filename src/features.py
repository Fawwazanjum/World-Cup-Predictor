"""Feature engineering: recent form, head-to-head, ranking gap, venue type."""

import numpy as np
import pandas as pd

from src import config


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 3-class target column: 0 = away win, 1 = draw, 2 = home win."""
    df = df.copy()
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [2, 1],
        default=0,
    )
    return df


def _long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape one row per match into two rows, one per team's perspective.

    Each row carries: match_idx (df's row position), team, side (home/away),
    date, and points earned (3 win / 1 draw / 0 loss). This shape is what
    makes a per-team rolling average straightforward in pandas.
    """
    home = pd.DataFrame({
        "match_idx": np.arange(len(df)),
        "team": df["home_team"].values,
        "side": "home",
        "date": df["date"].values,
        "points": np.select(
            [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
            [3, 1],
            default=0,
        ),
    })
    away = pd.DataFrame({
        "match_idx": np.arange(len(df)),
        "team": df["away_team"].values,
        "side": "away",
        "date": df["date"].values,
        "points": np.select(
            [df["away_score"] > df["home_score"], df["home_score"] == df["away_score"]],
            [3, 1],
            default=0,
        ),
    })
    return pd.concat([home, away], ignore_index=True)


def add_recent_form(df: pd.DataFrame, window: int = config.FORM_WINDOW_MATCHES) -> pd.DataFrame:
    """Add rolling points-per-game (last N matches) for home_team and away_team.

    Only matches strictly before the current one are used (shift(1) before the
    rolling average), so a team's form going into a match never includes the
    result of that match itself.
    """
    df = df.reset_index(drop=True).copy()
    long = _long_format(df).sort_values(["team", "date", "match_idx"])
    long["form"] = (
        long.groupby("team")["points"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    home_form = long.loc[long["side"] == "home", ["match_idx", "form"]].rename(columns={"form": "home_form"})
    away_form = long.loc[long["side"] == "away", ["match_idx", "form"]].rename(columns={"form": "away_form"})
    df = df.merge(home_form, left_index=True, right_on="match_idx").drop(columns="match_idx")
    df = df.merge(away_form, left_index=True, right_on="match_idx").drop(columns="match_idx")
    return df


def add_head_to_head(df: pd.DataFrame, lookback_years: int = config.H2H_LOOKBACK_YEARS) -> pd.DataFrame:
    """Add historical head-to-head win/draw/loss rate between the two teams.

    For each match, looks only at *prior* meetings (within lookback_years)
    between the same two teams, regardless of which side was home in those
    earlier meetings, and computes the current home_team's win/draw rate.
    """
    df = df.reset_index(drop=True).copy()
    pair = [tuple(sorted([h, a])) for h, a in zip(df["home_team"], df["away_team"])]
    df["pair"] = pair

    h2h_home_win_rate = np.full(len(df), np.nan)
    h2h_draw_rate = np.full(len(df), np.nan)
    h2h_matches_played = np.zeros(len(df), dtype=int)

    for _, group in df.sort_values("date").groupby("pair"):
        history = []  # list of (date, winner) where winner is a team name or "draw"
        for pos, row in group.iterrows():
            cutoff = row["date"] - pd.DateOffset(years=lookback_years)
            relevant = [w for d, w in history if d >= cutoff]
            if relevant:
                n = len(relevant)
                h2h_matches_played[pos] = n
                h2h_home_win_rate[pos] = sum(w == row["home_team"] for w in relevant) / n
                h2h_draw_rate[pos] = sum(w == "draw" for w in relevant) / n
            if row["home_score"] > row["away_score"]:
                winner = row["home_team"]
            elif row["home_score"] < row["away_score"]:
                winner = row["away_team"]
            else:
                winner = "draw"
            history.append((row["date"], winner))

    df["h2h_home_win_rate"] = h2h_home_win_rate
    df["h2h_draw_rate"] = h2h_draw_rate
    df["h2h_matches_played"] = h2h_matches_played
    df = df.drop(columns=["pair"])
    return df


def add_rankings(df: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Join the current FIFA rank + points for home_team/away_team, add rank gap.

    rankings is a single current snapshot (not historical), so this is a static
    join on team name rather than a nearest-prior-date lookup. Teams outside the
    48 WC2026 list won't have a match; historical training rows involving other
    teams will get NaN ranking features.
    """
    df = df.copy()
    home_rank = rankings.rename(columns={"team": "home_team", "rank": "home_rank", "points": "home_points"})
    away_rank = rankings.rename(columns={"team": "away_team", "rank": "away_rank", "points": "away_points"})
    df = df.merge(home_rank, on="home_team", how="left")
    df = df.merge(away_rank, on="away_team", how="left")
    df["rank_gap"] = df["away_rank"] - df["home_rank"]
    df["points_gap"] = df["home_points"] - df["away_points"]
    return df


def add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_home_advantage / is_neutral flags derived from the neutral column."""
    df = df.copy()
    df["is_neutral"] = df["neutral"].astype(bool)
    df["is_home_advantage"] = ~df["is_neutral"]
    return df


def build_feature_set(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline and return a model-ready DataFrame."""
    df = matches.sort_values("date").reset_index(drop=True)
    df = add_target(df)
    df = add_recent_form(df)
    df = add_head_to_head(df)
    df = add_rankings(df, rankings)
    df = add_venue_features(df)
    return df
