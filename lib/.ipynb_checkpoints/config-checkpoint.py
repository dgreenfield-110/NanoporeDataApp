from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
DB_PATH  = DATA_DIR / "catalog.duckdb"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
