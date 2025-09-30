"""Build NAV-based feature panels using the v2 pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pipeline_v2 import (  # noqa: E402
    build_full_feature_set_v2,
    save_features_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct feature panels with NAV-derived technical indicators",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Subsample size for GP feature generation",
    )
    parser.add_argument(
        "--n-gp-components",
        type=int,
        default=15,
        help="Number of GP components to retain",
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=5,
        help="Number of PCA components to retain",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist features under artifacts/features_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_frames, artifacts = build_full_feature_set_v2(
        sample_size=args.sample_size,
        n_gp_components=args.n_gp_components,
        n_pca_components=args.n_pca_components,
    )
    print(f"[Features-v2] Built feature panels: {list(feature_frames.keys())}")
    if hasattr(artifacts.pca, "loadings"):
        print(f"[Features-v2] PCA loadings shape: {artifacts.pca.loadings.shape}")
    if args.save:
        save_features_v2(feature_frames)
        print('[Features-v2] Saved feature panels to features_v2 directory')


if __name__ == "__main__":
    main()
