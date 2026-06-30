from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Expected raw source files (place downloaded CSVs here)
RESULTS_CSV = RAW_DIR / "results.csv"          # historical match results
RANKINGS_CSV = RAW_DIR / "fifa_ranking.csv"     # current FIFA ranking snapshot (June 2026), 48 WC2026 teams only
FIXTURES_CSV = RAW_DIR / "wc2026_fixtures.csv"  # remaining WC 2026 fixtures / bracket state

# Feature engineering windows
FORM_WINDOW_MATCHES = 5   # number of recent matches used for "form" features
H2H_LOOKBACK_YEARS = 20   # how far back to look for head-to-head history

RANDOM_STATE = 42
