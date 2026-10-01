"""Central configuration. Every value is taken from the Capstone notebook
(Capstone_fixed_project.ipynb); section numbers are given where relevant."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

RAW_DATA_FILE = DATA_RAW / "robin_dataset.csv"            # original Kaggle file (place here if you have it)
CLEAN_DATA_FILE = DATA_PROCESSED / "clean_holdings.csv"   # cleaned rows (rebuilt from dashboard_scores.csv)
MODEL_FILE = MODELS_DIR / "risk_model.joblib"
PORTABLE_MODEL_FILE = MODELS_DIR / "model_export.json"
BEST_PARAMS_FILE = ARTIFACTS_DIR / "best_params.json"
LIVE_METRICS_FILE = ARTIFACTS_DIR / "live_metrics.json"

# Notebook sections 1, 4, 6
RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15     # chronological split; test = remaining 15 percent
TUNE_STEP = 4                         # every 4th training row is used for the tuning search
WINDOW = 7                            # rolling window / momentum lag
MIN_HISTORY = 16                      # observations required (conservative; see FINDINGS.md)

FEATURES = ["prev_popularity_change", "prev_growth_rate", "prev_rolling_avg", "prev_momentum",
            "prev_users_holding", "down_count_7", "growth_vol_7"]

# Target (notebook section 3): risk = 1 if users_holding DECREASES versus the previous observation.
LABEL_MAP = {0: "Low risk", 1: "High risk"}
HIGH_LABEL, LOW_LABEL = LABEL_MAP[1], LABEL_MAP[0]
DEFAULT_MODEL_NAME = "Gradient Boosting"   # the model saved by notebook section 12

# Data-quality flags (notebook cell 33)
EXTREME_JUMP_PCT = 10.0          # |change| > 1000 percent between consecutive readings
EXTREME_JUMP_MIN_COUNT = 20
ZERO_SHARE_MAX = 0.25
MERGE_ERROR_NAME = "robin_dataset"   # rows mislabelled by the original merge (notebook section 2)
REQUIRED_COLUMNS = ["timestamp", "users_holding", "name"]
