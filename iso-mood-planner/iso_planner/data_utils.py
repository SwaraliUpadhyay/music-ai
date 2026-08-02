"""
Stratified train/test split for the song pool.

Why stratified rather than a plain random split: a naive random split can
accidentally leave one mood region (e.g. low-valence/high-arousal, "angry")
under-represented or entirely absent from the test slice, which would make
the planner look worse than it is -- through a bad test set, not a bad
algorithm. Stratifying by (source dataset, valence-arousal quadrant) keeps
both splits similarly shaped across the mood space and across DEAM/PMEmo/
Emotify, so a held-out evaluation reflects the algorithm's real behavior.

Usage:
    train_pool, test_pool = stratified_train_test_split(pool, test_size=0.2)

Use `train_pool` ("dev pool") for everything exploratory -- picking
max_jump/match_fraction defaults, the bandit's arm-grid bounds, eyeballing
stress-test behavior. Only run final reported metrics on `test_pool`, and
don't let its content influence any further design decisions once you've
looked at it -- that defeats the purpose of holding it out.
"""

import numpy as np
import pandas as pd


def _assign_quadrant(row, midpoint: float = 0.5) -> str:
    v_q = "vHigh" if row["valence"] >= midpoint else "vLow"
    a_q = "aHigh" if row["arousal"] >= midpoint else "aLow"
    return f"{v_q}_{a_q}"


def stratified_train_test_split(pool: pd.DataFrame, test_size: float = 0.2,
                                 seed: int = 42, midpoint: float = 0.5):
    """
    Returns (train_pool, test_pool), both with the original columns
    (no stratum column leaked into the output), split so that each
    (source, valence-arousal quadrant) stratum contributes roughly
    `test_size` fraction of its rows to the test split.

    Strata with very few rows (<2) are kept entirely in train (a single
    row can't be meaningfully split, and forcing it into test would just
    remove rare coverage from the pool the algorithm actually runs on
    during development).
    """
    pool = pool.copy().reset_index(drop=True)
    strata = pool["source"].astype(str) + "_" + pool.apply(_assign_quadrant, axis=1, midpoint=midpoint)
    rng = np.random.default_rng(seed)

    test_idx = []
    for stratum, group in pool.groupby(strata):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        if len(idx) < 2:
            continue
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test])

    test_mask = pool.index.isin(set(test_idx))
    test_pool = pool.loc[test_mask].reset_index(drop=True)
    train_pool = pool.loc[~test_mask].reset_index(drop=True)
    return train_pool, test_pool


def stratum_coverage_report(pool: pd.DataFrame, midpoint: float = 0.5) -> pd.DataFrame:
    """Row counts per (source, VA-quadrant) stratum -- use to sanity-check
    that train/test splits have proportionally similar coverage."""
    strata = pool["source"].astype(str) + "_" + pool.apply(_assign_quadrant, axis=1, midpoint=midpoint)
    return strata.value_counts().sort_index().rename("count").to_frame()
