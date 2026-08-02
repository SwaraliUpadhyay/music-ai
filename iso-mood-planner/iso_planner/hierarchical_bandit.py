"""
Contextual Sharing Across Users (cold-start acceleration)
-----------------------------------------------------------
Each UserBandit currently starts from the same flat prior (mean=0, var=1)
for every arm, regardless of what's been learned from other users. That
means every new user pays the full cost of exploring a 27-arm grid from
scratch, even if hundreds of other users with a similar mood-shift task
have already told us roughly which arms tend to work.

This module adds one level of sharing: a population-level posterior,
bucketed by mood_distance = ||target - start|| (how big a mood swing the
session is asking for), which is used only to set a new user's *prior*
at bandit-creation time. Individual per-user learning then proceeds
exactly as before (bandit.py is unchanged in its update logic) -- once a
user has their own data, their individual likelihood naturally dominates
the (now more informative) prior via the same conjugate update.

Why bucket by mood_distance specifically: the stress tests on the base
planner already showed that large mood swings behave differently (the
matching phase eats into a limited step budget) -- so it's plausible
that "what pace works" partly depends on how far the session needs to
travel, not just on idiosyncratic user taste. Bucketing lets the shared
prior capture that, while still leaving room for individual deviation.

This is deliberately a simple two-level empirical-Bayes hierarchy, not a
full contextual bandit with a learned reward model (e.g. LinUCB) -- that
richer approach is noted as future work once real (non-simulated) data
exists to justify the added complexity.
"""

from __future__ import annotations
import json
import math
import os
from typing import Optional

import numpy as np

from .bandit import Arm, ArmStats, UserBandit, BanditStore, build_arm_grid


# ---------------------------------------------------------------------------
# Context: mood-distance bucketing
# ---------------------------------------------------------------------------

def mood_distance(start: tuple, target: tuple) -> float:
    (v0, a0), (vT, aT) = start, target
    return math.sqrt((vT - v0) ** 2 + (aT - a0) ** 2)


def bucketize_mood_distance(d: float, thresholds=(0.4, 0.8)) -> str:
    """
    d in [0, sqrt(2)] ~= [0, 1.41]. Default thresholds split roughly into
    thirds: 'small' (<0.4), 'medium' (0.4-0.8), 'large' (>0.8).
    """
    lo, hi = thresholds
    if d < lo:
        return "small"
    elif d < hi:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Global (population-level) context prior
# ---------------------------------------------------------------------------

class GlobalContextPrior:
    """
    Pools reward data across ALL users, keyed by (context_bucket, arm), using
    the same Normal-Normal conjugate machinery as UserBandit -- but here the
    "noise variance" represents *between-user* variability (how much
    individual users deviate from the population mean for that bucket/arm),
    which should generally be set larger than an individual user's own
    within-user noise_var, since users differ from each other more than a
    single user's repeated feedback differs from itself.
    """

    def __init__(self,
                 arms: Optional[list[Arm]] = None,
                 buckets=("small", "medium", "large"),
                 hyper_prior_mean: float = 0.0,
                 hyper_prior_var: float = 1.0,
                 between_user_noise_var: float = 0.35):
        self.arms = arms if arms is not None else build_arm_grid()
        self.buckets = buckets
        self.hyper_prior_mean = hyper_prior_mean
        self.hyper_prior_var = hyper_prior_var
        self.between_user_noise_var = between_user_noise_var
        self.stats: dict[tuple, ArmStats] = {
            (bucket, a.as_tuple()): ArmStats() for bucket in buckets for a in self.arms
        }

    def _posterior(self, bucket: str, arm_key: tuple) -> tuple[float, float]:
        s = self.stats[(bucket, arm_key)]
        post_var = 1.0 / (1.0 / self.hyper_prior_var + s.n / self.between_user_noise_var)
        post_mean = post_var * (self.hyper_prior_mean / self.hyper_prior_var
                                 + s.n * s.mean / self.between_user_noise_var)
        return post_mean, post_var

    def get_prior(self, bucket: str) -> dict:
        """{arm_tuple: (mean, var)} to seed a brand-new user's UserBandit."""
        return {a.as_tuple(): self._posterior(bucket, a.as_tuple()) for a in self.arms}

    def update(self, bucket: str, arm: Arm, reward: float):
        s = self.stats[(bucket, arm.as_tuple())]
        s.n += 1
        s.sum_reward += reward

    def n_users_contributed(self, bucket: str) -> int:
        """Rough proxy for how 'mature' a bucket's pooled data is (total pulls)."""
        return sum(self.stats[(bucket, a.as_tuple())].n for a in self.arms)

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "buckets": list(self.buckets),
            "hyper_prior_mean": self.hyper_prior_mean,
            "hyper_prior_var": self.hyper_prior_var,
            "between_user_noise_var": self.between_user_noise_var,
            "arms": [{"match_fraction": a.match_fraction, "max_jump": a.max_jump, "n_songs": a.n_songs}
                     for a in self.arms],
            "stats": {f"{bucket}|{','.join(map(str, key))}": asdict(v)
                      for (bucket, key), v in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalContextPrior":
        arms = [Arm(**a) for a in d["arms"]]
        gp = cls(arms=arms, buckets=tuple(d["buckets"]), hyper_prior_mean=d["hyper_prior_mean"],
                  hyper_prior_var=d["hyper_prior_var"], between_user_noise_var=d["between_user_noise_var"])
        for k_str, v in d["stats"].items():
            bucket, key_str = k_str.split("|")
            parts = key_str.split(",")
            key = (float(parts[0]), float(parts[1]), int(parts[2]))
            gp.stats[(bucket, key)] = ArmStats(**v)
        return gp

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GlobalContextPrior":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Manager tying per-user bandits + the global context prior together
# ---------------------------------------------------------------------------

class HierarchicalBanditManager:
    def __init__(self, base_dir: str, arms: Optional[list[Arm]] = None,
                 buckets=("small", "medium", "large"), bucket_fn=None):
        """
        buckets: the full set of valid bucket keys, pre-populated in the
                 pooled GlobalContextPrior. Defaults to the 3 mood-distance
                 buckets; pass a larger composite set (e.g.
                 "<distance>|<mood_category>|<time_of_day>" strings) to test
                 richer, higher-cardinality bucketing.
        bucket_fn: optional callable (start, target) -> bucket_key,
                   overriding the default mood-distance-only bucketing.
        """
        self.arms = arms if arms is not None else build_arm_grid()
        self.bucket_fn = bucket_fn or (lambda start, target, **kwargs: bucketize_mood_distance(mood_distance(start, target)))
        self.user_store = BanditStore(base_dir=os.path.join(base_dir, "users"))
        self._global_path = os.path.join(base_dir, "global_context_prior.json")
        if os.path.exists(self._global_path):
            self.global_prior = GlobalContextPrior.load(self._global_path)
        else:
            self.global_prior = GlobalContextPrior(arms=self.arms, buckets=buckets)

    def get_bandit_for_session(self, user_id: str, start: tuple, target: tuple, **context_kwargs):
        """
        Returns (bandit, bucket, is_new_user). New users are warm-started
        from the current global context prior for their mood-distance
        bucket; returning users keep their own accumulated posterior
        (which already incorporates whatever prior they were warm-started
        with, plus their own sessions since).

        context_kwargs: forwarded to bucket_fn (e.g. hour_of_day=...) for
        custom bucket_fns that use more than just (start, target).
        """
        bucket = self.bucket_fn(start, target, **context_kwargs)
        user_path = self.user_store._path(user_id)
        is_new_user = not os.path.exists(user_path)

        if is_new_user:
            arm_priors = self.global_prior.get_prior(bucket)
            bandit = UserBandit(user_id, arms=self.arms, arm_priors=arm_priors)
        else:
            bandit = self.user_store.get(user_id)

        return bandit, bucket, is_new_user

    def close_session(self, bandit: UserBandit, bucket: str, arm: Arm, reward: float):
        bandit.update(arm, reward)
        self.global_prior.update(bucket, arm, reward)
        self.user_store.save(bandit)
        self.global_prior.save(self._global_path)
