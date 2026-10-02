"""Download the exact aligned archive, defaulting to the pinned Kaggle mirror."""

import argparse
from pathlib import Path

from mosei.download import AUTHORS_URL, KAGGLE_URL, download_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["kaggle", "authors"], default="kaggle")
    parser.add_argument("--output", type=Path, default=Path("data/raw/aligned_50.pkl"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        result = download_file(
            KAGGLE_URL if args.source == "kaggle" else AUTHORS_URL,
            args.output,
            workers=args.workers,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Verified dataset: {result}")


if __name__ == "__main__":
    main()
