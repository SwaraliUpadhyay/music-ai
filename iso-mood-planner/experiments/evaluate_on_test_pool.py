"""
Final quantitative evaluation of the planner -- run ONLY on the held-out
test pool (data/song_pool_test.csv), never the dev pool. These are the
numbers that belong in the paper's results section, since they weren't
used to pick max_jump/match_fraction defaults or anything else.

Run experiments/make_dataset_split.py first if song_pool_test.csv doesn't
exist yet.
"""

import os
import numpy as np
import pandas as pd

from iso_planner.trajectory_planner import plan_playlist

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
TEST_POOL_PATH = "data/song_pool_test.csv"

# Same stress-test scenarios used during development, run here against
# held-out songs only.
SCENARIOS = [
    ("sad_to_calm",        (0.20, 0.30), (0.75, 0.35)),
    ("stressed_to_relaxed", (0.25, 0.80), (0.65, 0.30)),
    ("angry_to_calm",       (0.20, 0.85), (0.70, 0.35)),
    ("bored_to_energized",  (0.45, 0.20), (0.70, 0.80)),
    ("already_near_target", (0.60, 0.55), (0.65, 0.60)),
    ("extreme_corner_swing", (0.05, 0.05), (0.95, 0.95)),
]


def evaluate_scenario(pool, name, start, target, n_songs=10):
    playlist = plan_playlist(pool, start, target, n_songs=n_songs)

    final_v, final_a = playlist.iloc[-1][["valence", "arousal"]]
    final_dist = float(np.hypot(final_v - target[0], final_a - target[1]))

    va = playlist[["valence", "arousal"]].to_numpy()
    steps = np.hypot(np.diff(va[:, 0]), np.diff(va[:, 1]))
    mean_jump, max_jump_realized = float(steps.mean()), float(steps.max())

    n_unique_sources = playlist["source"].nunique()
    n_unique_songs = playlist["song_id"].nunique()

    return {
        "scenario": name,
        "n_songs": n_songs,
        "final_distance_to_target": round(final_dist, 4),
        "mean_step_jump": round(mean_jump, 4),
        "max_step_jump": round(max_jump_realized, 4),
        "unique_songs": n_unique_songs,
        "unique_sources_used": n_unique_sources,
    }


def main():
    if not os.path.exists(TEST_POOL_PATH):
        raise FileNotFoundError(
            f"{TEST_POOL_PATH} not found -- run `python -m experiments.make_dataset_split` first."
        )
    test_pool = pd.read_csv(TEST_POOL_PATH)
    print(f"Evaluating on held-out test pool only: {len(test_pool)} songs "
          f"(never used for tuning defaults)\n")

    rows = [evaluate_scenario(test_pool, name, start, target) for name, start, target in SCENARIOS]
    results = pd.DataFrame(rows)
    results.to_csv(f"{OUT_DIR}/test_pool_evaluation.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
