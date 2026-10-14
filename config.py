"""Central configuration: all decisions in one place (R1)."""
from pathlib import Path

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

for p in [PROCESSED, FIGURES, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

TEST_MONTHS = 2
PURCHASE_VALUE = "log"
MIN_CUSTOMERS_PER_PRODUCT = 10
RANDOM_SEED = 42

METHOD = "svd"
K_VALUES = [10, 20, 50, 100, 200]
K_CHOSEN = 50
TOP_N_RECS = 10
N_CLUSTERS = 4
