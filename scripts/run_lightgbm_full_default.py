"""Convenience launcher for full LightGBM training without recency weighting."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "scripts" / "train_lightgbm_full.py"),
        "--param-config",
        "configs/lightgbm_lag_params.json",
        "--train-end",
        "1960",
        "--feature-version",
        "v2",
        "--artifact-root",
        "lightgbm_full",
        "--run-prefix",
        "tunedfull",
        "--weight-scheme",
        "none",
        "--lags",
        "1",
        "2",
        "3",
        "4",
        "--bagging-freq",
        "5",
        "--summary-path",
        "artifacts/lightgbm_full_summary_1960.json",
    ]
    print("[run_lightgbm_full_default] Executing:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
