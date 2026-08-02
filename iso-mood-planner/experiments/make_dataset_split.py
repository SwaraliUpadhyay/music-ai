"""
Run once (or whenever combined_song_pool.csv changes) to produce a
stratified 80/20 dev/test split of the song pool.

- data/song_pool_dev.csv  -- use for ALL exploratory work: choosing
  max_jump/match_fraction defaults, arm-grid bounds, eyeballing stress
  tests, debugging.
- data/song_pool_test.csv -- touch only for final reported metrics
  (see evaluate_on_test_pool.py). Once you've looked at results computed
  on it, treat it as spent for that round of design decisions -- don't
  iterate against it, or it stops being a real held-out set.
"""

import pandas as pd
from iso_planner.data_utils import stratified_train_test_split, stratum_coverage_report

SRC = "data/combined_song_pool.csv"
DEV_OUT = "data/song_pool_dev.csv"
TEST_OUT = "data/song_pool_test.csv"


def main():
    pool = pd.read_csv(SRC)
    train_pool, test_pool = stratified_train_test_split(pool, test_size=0.2, seed=42)

    train_pool.to_csv(DEV_OUT, index=False)
    test_pool.to_csv(TEST_OUT, index=False)

    print(f"Full pool:  {len(pool)} songs")
    print(f"Dev pool:   {len(train_pool)} songs -> {DEV_OUT}")
    print(f"Test pool:  {len(test_pool)} songs -> {TEST_OUT}")

    print("\nStratum coverage, full pool:")
    print(stratum_coverage_report(pool))
    print("\nStratum coverage, dev pool:")
    print(stratum_coverage_report(train_pool))
    print("\nStratum coverage, test pool:")
    print(stratum_coverage_report(test_pool))


if __name__ == "__main__":
    main()
