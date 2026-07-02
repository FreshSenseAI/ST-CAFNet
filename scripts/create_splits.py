from __future__ import annotations

import argparse
from pathlib import Path

from stcafnet.data import load_manifest, split_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test, folds = split_manifest(
        load_manifest(args.manifest), args.test_fraction, args.folds, args.seed
    )
    test.to_csv(output / "test.csv", index=False)
    for index, (train, validation) in enumerate(folds):
        train.to_csv(output / f"fold_{index}_train.csv", index=False)
        validation.to_csv(output / f"fold_{index}_validation.csv", index=False)


if __name__ == "__main__":
    main()

