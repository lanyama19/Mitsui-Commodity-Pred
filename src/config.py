from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
TARGET_PAIRS_PATH = DATA_DIR / "target_pairs.csv"
LAGGED_TEST_LABELS_DIR = DATA_DIR / "lagged_test_labels"

OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
