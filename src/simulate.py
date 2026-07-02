"""Monte Carlo simulation of the remaining WC2026 knockout bracket."""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from src import config, data_loader
from src.features import FEATURE_COLUMNS

N_SIMULATIONS = 10_000

WC_TEAMS = [
    'Mexico','South Africa','South Korea','Czech Republic',
    'Switzerland','Canada','Bosnia and Herzegovina','Qatar',
    'Brazil','Morocco','Scotland','Haiti',
    'United States','Australia','Paraguay','Turkey',
    'Germany','Ivory Coast','Ecuador','Curaçao',
    'Netherlands','Japan','Sweden','Tunisia',
    'Egypt','Iran','Belgium','New Zealand',
    'Spain','Uruguay','Cape Verde','Saudi Arabia',
    'France','Norway','Senegal','Iraq',
    'Argentina','Austria','Algeria','Jordan',
    'Colombia','Portugal','DR Congo','Uzbekistan',
    'England','Ghana','Croatia','Panama',
]

def _load_bracket_state() -> tuple[dict, set, pd.DataFrame]:
    """Read wc2026_fixtures.csv and return current bracket state.

    Returns
    -------
    known_winners : {match_no: winner_team} for every completed match
    eliminated    : set of teams knocked out at any stage so far
    pending_r32   : DataFrame of R32 fixtures not yet played
    """
    fx = data_loader.load_fixtures()

    known_winners = {}
    for _, row in fx.iterrows():
        if pd.notna(row['winner']) and str(row['winner']).strip():
            known_winners[int(row['match_no'])] = row['winner']

    # Find all teams that appeared in a completed match but did not win it
    eliminated = set()
    r32 = fx[fx['stage'] == 'R32']
    for _, row in r32.iterrows():
        if pd.notna(row['winner']) and str(row['winner']).strip():
            loser = row['away_team'] if row['winner'] == row['home_team'] else row['home_team']
            eliminated.add(loser)

    # Group stage eliminations — every team not in any fixture (already out)
    fixture_teams = set(fx['home_team']).union(set(fx['away_team']))
    for team in WC_TEAMS:
        if team not in fixture_teams:
            eliminated.add(team)

    pending_r32 = fx[
        (fx['stage'] == 'R32') &
        (fx['winner'].isna() | (fx['winner'].astype(str).str.strip() == ''))
    ].copy()

    return known_winners, eliminated, pending_r32

# Full bracket wiring: (output_match_no, input_match_a, input_match_b)
R16_BRACKET = [
    (94, 73, 75),
    (89, 79, 80),
    (90, 81, 82),
    (95, 74, 77),
    (96, 76, 78),
    (91, 83, 84),
    (92, 85, 87),
    (93, 86, 88),
]
QF_BRACKET = [
    (97,  89, 90),
    (98,  91, 92),
    (99,  93, 94),
    (100, 95, 96),
]
SF_BRACKET = [
    (101, 97,  98),
    (102, 99, 100),
]
FINAL_MATCH = (104, 101, 102)
THIRD_PLACE  = (103, 101, 102)  # losers of SFs


def _load_model():
    with open(config.MODELS_DIR / "best_model.pkl", "rb") as f:
        return pickle.load(f)  # dict with keys: model, scaler (None for XGB), type


def _compute_team_forms(competitive: pd.DataFrame, window: int = config.FORM_WINDOW_MATCHES) -> dict:
    """Return {team: avg_points_per_game} over their last `window` competitive matches."""
    rows = []
    for _, r in competitive.iterrows():
        hs, as_ = r['home_score'], r['away_score']
        h_pts = 3 if hs > as_ else (1 if hs == as_ else 0)
        a_pts = 3 if as_ > hs else (1 if hs == as_ else 0)
        rows.append({'team': r['home_team'], 'date': r['date'], 'pts': h_pts})
        rows.append({'team': r['away_team'],  'date': r['date'], 'pts': a_pts})

    long = pd.DataFrame(rows).sort_values(['team', 'date'])
    forms = {}
    for team, grp in long.groupby('team'):
        last = grp['pts'].values[-window:]
        forms[team] = float(last.mean()) if len(last) > 0 else 1.5
    return forms


def _compute_h2h(competitive: pd.DataFrame, wc_teams: list, lookback_years: int = config.H2H_LOOKBACK_YEARS) -> dict:
    """Return {(home_team, away_team): (win_rate, draw_rate, n)} for all WC team pairs.

    Vectorised: filter the DataFrame once, tag every match with a canonical
    pair key, then groupby that key to aggregate — rather than re-filtering
    per pair which is O(n_pairs × n_rows).
    """
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
    wc_set = set(wc_teams)

    rel = competitive[
        (competitive['date'] >= cutoff) &
        competitive['home_team'].isin(wc_set) &
        competitive['away_team'].isin(wc_set)
    ].copy()

    if rel.empty:
        return {(t1, t2): (np.nan, np.nan, 0) for t1 in wc_teams for t2 in wc_teams if t1 != t2}

    # Canonical pair key so we can groupby unordered pairs in one pass
    rel['t1'] = rel[['home_team', 'away_team']].min(axis=1)
    rel['t2'] = rel[['home_team', 'away_team']].max(axis=1)

    # From t1's perspective: did t1 win / draw?
    rel['t1_is_home'] = rel['home_team'] == rel['t1']
    rel['t1_win'] = np.where(
        rel['t1_is_home'],
        rel['home_score'] > rel['away_score'],
        rel['away_score'] > rel['home_score'],
    )
    rel['draw'] = rel['home_score'] == rel['away_score']

    agg = rel.groupby(['t1', 't2']).agg(
        n=('draw', 'count'),
        t1_wins=('t1_win', 'sum'),
        draws=('draw', 'sum'),
    ).reset_index()

    # Build lookup for both orderings: (t1 vs t2) and (t2 vs t1)
    h2h: dict = {}
    for _, row in agg.iterrows():
        t1, t2 = row['t1'], row['t2']
        n = int(row['n'])
        t1_win_rate = row['t1_wins'] / n
        draw_rate   = row['draws']   / n
        t2_win_rate = (n - row['t1_wins'] - row['draws']) / n
        h2h[(t1, t2)] = (t1_win_rate, draw_rate, n)   # t1 as "home"
        h2h[(t2, t1)] = (t2_win_rate, draw_rate, n)   # t2 as "home"

    # Fill missing pairs with NaN so _make_features knows to use base rates
    for t1 in wc_teams:
        for t2 in wc_teams:
            if t1 != t2 and (t1, t2) not in h2h:
                h2h[(t1, t2)] = (np.nan, np.nan, 0)

    return h2h


def _make_features(
    home: str, away: str,
    forms: dict, h2h: dict, rank_lookup: dict, elo_ratings: dict,
    base_h2h_win: float, base_draw: float,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for a matchup."""
    wr, dr, n = h2h.get((home, away), (np.nan, np.nan, 0))
    wr = base_h2h_win if np.isnan(wr) else wr
    dr = base_draw    if np.isnan(dr) else dr

    h_rank, h_pts = rank_lookup.get(home, (None, None))
    a_rank, a_pts = rank_lookup.get(away, (None, None))

    has_rank = int(h_rank is not None and a_rank is not None)
    rank_gap   = (a_rank - h_rank) if has_rank else 0.0
    points_gap = (h_pts  - a_pts)  if has_rank else 0.0

    from src.elo import DEFAULT_ELO
    elo_h = elo_ratings.get(home, DEFAULT_ELO)
    elo_a = elo_ratings.get(away, DEFAULT_ELO)

    return pd.DataFrame([{
        'home_form':          forms.get(home, 1.5),
        'away_form':          forms.get(away, 1.5),
        'h2h_home_win_rate':  wr,
        'h2h_draw_rate':      dr,
        'h2h_matches_played': n,
        'elo_diff':           elo_h - elo_a,
        'rank_gap':           rank_gap,
        'points_gap':         points_gap,
        'ranking_available':  has_rank,
        'is_neutral':         True,
    }])[FEATURE_COLUMNS]


def _knockout_winner(
    home: str, away: str, model, scaler,
    forms, h2h, rank_lookup, elo_ratings,
    base_h2h_win, base_draw, rng,
) -> str:
    """Predict and sample the winner of a knockout match (no draws allowed).

    The model gives P(home win), P(draw), P(away win). In knockout football
    draws don't exist (extra time / penalties decide it), so we renormalise
    to a 2-way contest: P(home win adjusted) = P(home win) / (P(home win) + P(away win)).
    Mathematically this is just conditioning on the event that the match is decisive.
    """
    feat = _make_features(home, away, forms, h2h, rank_lookup, elo_ratings, base_h2h_win, base_draw)
    X = scaler.transform(feat) if scaler is not None else feat
    probs = model.predict_proba(X)[0]  # [p_away, p_draw, p_home] (class order 0,1,2)
    p_away, _, p_home = probs
    total = p_home + p_away
    p_home_adj = p_home / total if total > 0 else 0.5
    return home if rng.random() < p_home_adj else away


def simulate_once(
    model, scaler, pending_r32: pd.DataFrame,
    forms, h2h, rank_lookup, elo_ratings,
    base_h2h_win, base_draw, rng,
    known_winners: dict = None,
) -> dict:
    """Run one full simulation of the remaining bracket.

    Returns a dict {match_no: winner_team} for every knockout match.
    """
    winners = dict(known_winners or {})

    # ── Remaining Round of 32 ────────────────────────────────────────────────
    for _, match in pending_r32.iterrows():
        w = _knockout_winner(
            match['home_team'], match['away_team'],
            model, scaler, forms, h2h, rank_lookup, elo_ratings, base_h2h_win, base_draw, rng,
        )
        winners[int(match['match_no'])] = w

    # ── Round of 16, QF, SF, Final ───────────────────────────────────────────
    for (out_no, in_a, in_b) in R16_BRACKET + QF_BRACKET + SF_BRACKET + [FINAL_MATCH]:
        home = winners[in_a]
        away = winners[in_b]
        winners[out_no] = _knockout_winner(
            home, away, model, scaler, forms, h2h, rank_lookup, elo_ratings, base_h2h_win, base_draw, rng,
        )

    return winners


def run_simulation(n: int = N_SIMULATIONS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Run n Monte Carlo simulations and return a probability table per team.

    For each team we compute:
      - P(reach R16)   = probability of winning their R32 match
      - P(reach QF)    = probability of winning their R16 match
      - P(reach SF)    = probability of winning their QF match
      - P(reach Final) = probability of winning their SF match
      - P(Champion)    = probability of winning the Final
    """
    rng        = np.random.default_rng(seed)
    model_pkg  = _load_model()
    model      = model_pkg["model"]
    scaler     = model_pkg["scaler"]

    known_winners, eliminated, pending_r32 = _load_bracket_state()
    print(f"R32 complete: {16 - len(pending_r32)}/16  |  Pending: {len(pending_r32)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    from src.elo import compute_elo
    df = data_loader.load_results()
    played, _ = data_loader.split_played_and_pending(df)
    competitive = data_loader.filter_competitive(played)

    rankings = data_loader.load_rankings()
    rank_lookup = {
        row['team']: (row['rank'], row['points'])
        for _, row in rankings.iterrows()
    }

    # Short window for current ratings — only 2020+ results so ratings reflect
    # today's squad quality rather than historical dominance from past generations.
    print("Computing Elo ratings...")
    _, elo_ratings = compute_elo(played, start_date=config.ELO_START_DATE)

    # ── Precompute features ───────────────────────────────────────────────────
    forms = _compute_team_forms(competitive)
    h2h   = _compute_h2h(competitive, WC_TEAMS)

    # Global base rates used when no H2H history exists
    wc_matches = competitive[competitive['tournament'] == 'FIFA World Cup']
    if len(wc_matches) > 0:
        base_h2h_win = (wc_matches['home_score'] > wc_matches['away_score']).mean()
        base_draw    = (wc_matches['home_score'] == wc_matches['away_score']).mean()
    else:
        base_h2h_win, base_draw = 0.49, 0.23

    print(f"Precomputation done. Running {n:,} simulations...")

    # ── Counters ──────────────────────────────────────────────────────────────
    stage_counts = defaultdict(lambda: defaultdict(int))

    # Teams that already won R32 are guaranteed to reach R16
    r32_winners_so_far = {v for k, v in known_winners.items()
                          if k in {m[0] for bracket in [R16_BRACKET] for m in bracket
                                   for _ in [None]} or
                          k <= 88}
    # Simpler: any team that is a winner in an R32 match
    for match_no, winner in known_winners.items():
        if match_no <= 88:   # R32 match numbers are 73-88
            stage_counts[winner]['R16'] = n

    pending_r32_nos = set(pending_r32['match_no'].astype(int))

    for _ in range(n):
        sim = simulate_once(
            model, scaler, pending_r32, forms, h2h, rank_lookup, elo_ratings,
            base_h2h_win, base_draw, rng, known_winners,
        )
        # Count R16 entries only for teams that won a *pending* R32 match
        for r32_no in pending_r32_nos:
            if r32_no in sim:
                stage_counts[sim[r32_no]]['R16'] += 1

        for out_no, in_a, in_b in R16_BRACKET:
            stage_counts[sim[out_no]]['QF'] += 1

        for out_no, in_a, in_b in QF_BRACKET:
            stage_counts[sim[out_no]]['SF'] += 1

        for out_no, in_a, in_b in SF_BRACKET:
            stage_counts[sim[out_no]]['Final'] += 1

        stage_counts[sim[FINAL_MATCH[0]]]['Champion'] += 1

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for team in WC_TEAMS:
        if team in eliminated:
            rows.append({'team': team, 'P(R16)': 0, 'P(QF)': 0,
                         'P(SF)': 0, 'P(Final)': 0, 'P(Champion)': 0})
        else:
            sc = stage_counts[team]
            rows.append({
                'team':         team,
                'P(R16)':       round(sc['R16']      / n, 4),
                'P(QF)':        round(sc['QF']       / n, 4),
                'P(SF)':        round(sc['SF']       / n, 4),
                'P(Final)':     round(sc['Final']    / n, 4),
                'P(Champion)':  round(sc['Champion'] / n, 4),
            })

    results = pd.DataFrame(rows).sort_values('P(Champion)', ascending=False)
    results = results.reset_index(drop=True)
    results.index += 1
    return results


if __name__ == "__main__":
    results = run_simulation()
    pd.set_option('display.float_format', '{:.1%}'.format)
    pd.set_option('display.max_rows', 50)
    print("\n=== WC 2026 — CHAMPIONSHIP PROBABILITIES ===\n")
    print(results.to_string())
    results.to_csv(config.PROCESSED_DIR / "wc2026_predictions.csv", index=True)
    print("\nSaved to data/processed/wc2026_predictions.csv")
