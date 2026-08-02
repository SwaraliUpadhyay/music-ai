"""
Linear Contextual Sharing (richer context features)
------------------------------------------------------
hierarchical_bandit.py shares data across users via discrete buckets
(mood_distance only: small/medium/large). That's fine for one context
dimension, but doesn't scale: adding mood_category (4 values) and
time_of_day (4 values) the same way gives 3*4*4 = 48 cells, each of which
now needs its OWN population of users before its pooled posterior means
anything -- fragmenting exactly the data that made sharing help in the
first place.

This module replaces discrete bucketing with a linear model of each arm's
expected reward as a function of a continuous context feature vector:

    E[reward | arm k, context x] = x^T w_k

fit online via Bayesian linear (ridge) regression per arm -- ridge
regression's confidence width sqrt(x^T A_k^-1 x) is exactly LinUCB's
uncertainty term; here it's used as a Gaussian std to SAMPLE from
(Thompson Sampling for linear contextual bandits, Agrawal & Goyal 2013)
rather than added as a UCB bonus, so the exploration strategy stays
consistent with the rest of this system (also Thompson Sampling).

Two users with different contexts now share the same small set of
coefficients per arm, instead of needing to land in the same discrete
cell -- this is what lets richer context (more features) help rather
than hurt cold-start.
"""

from __future__ import annotations
import json
import math
import os
from typing import Optional

import numpy as np

from .bandit import Arm, UserBandit, BanditStore, build_arm_grid


# ---------------------------------------------------------------------------
# Context feature engineering
# ---------------------------------------------------------------------------

MOOD_CATEGORIES = ("sad", "angry", "content", "excited")   # low/high valence x low/high arousal
TIME_BUCKETS = ("morning", "afternoon", "evening", "night", "unknown")

FEATURE_NAMES = (
    ["bias", "mood_distance", "start_valence", "start_arousal", "delta_valence", "delta_arousal"]
    + [f"mood_cat_{m}" for m in MOOD_CATEGORIES]
    + [f"time_{t}" for t in TIME_BUCKETS]
)
N_FEATURES = len(FEATURE_NAMES)


def mood_category_of(valence: float, arousal: float, midpoint: float = 0.5) -> str:
    if valence < midpoint and arousal < midpoint:
        return "sad"
    if valence < midpoint and arousal >= midpoint:
        return "angry"
    if valence >= midpoint and arousal < midpoint:
        return "content"
    return "excited"


def time_bucket_of(hour_of_day: Optional[int]) -> str:
    if hour_of_day is None:
        return "unknown"
    if 5 <= hour_of_day < 12:
        return "morning"
    if 12 <= hour_of_day < 17:
        return "afternoon"
    if 17 <= hour_of_day < 21:
        return "evening"
    return "night"


def build_context_features(start: tuple, target: tuple, hour_of_day: Optional[int] = None) -> np.ndarray:
    """
    Returns a fixed-length feature vector (see FEATURE_NAMES for the
    meaning of each entry). One-hot blocks (mood category, time of day)
    are mutually exclusive within their block; 'unknown' time is its own
    category so missing timestamps degrade gracefully instead of biasing
    toward any real time bucket.
    """
    v0, a0 = start
    vT, aT = target
    mood_distance = math.sqrt((vT - v0) ** 2 + (aT - a0) ** 2) / math.sqrt(2)  # normalize to ~[0,1]

    cat = mood_category_of(v0, a0)
    tb = time_bucket_of(hour_of_day)

    x = [1.0, mood_distance, v0, a0, (vT - v0), (aT - a0)]
    x += [1.0 if m == cat else 0.0 for m in MOOD_CATEGORIES]
    x += [1.0 if t == tb else 0.0 for t in TIME_BUCKETS]
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------
# Per-arm Bayesian linear (ridge) regression
# ---------------------------------------------------------------------------

class LinearArmModel:
    """
    Bayesian ridge regression for one arm's expected reward as a function
    of context: reward ~ N(x^T w, noise_var), prior w ~ N(0, (1/lam) I).

    A = lam*I + sum(x x^T)          (d x d)
    b = sum(x * reward)             (d,)
    posterior mean       = A^-1 b
    posterior covariance = noise_var * A^-1
    """

    def __init__(self, dim: int = N_FEATURES, lam: float = 1.0, noise_var: float = 0.25):
        self.dim = dim
        self.lam = lam
        self.noise_var = noise_var
        self.A = lam * np.eye(dim)
        self.b = np.zeros(dim)
        self.n = 0

    def posterior(self, x: np.ndarray) -> tuple[float, float]:
        """Predicted (mean, variance) of reward for context x, variance
        reflecting parameter uncertainty only (no + noise_var) -- this is
        handed to a UserBandit as arm_priors, whose own conjugate update
        adds each user's session noise separately, so noise shouldn't be
        double-counted here."""
        A_inv = np.linalg.inv(self.A)
        w_hat = A_inv @ self.b
        mean = float(x @ w_hat)
        var = float(self.noise_var * (x @ A_inv @ x))
        return mean, max(var, 1e-6)

    def update(self, x: np.ndarray, reward: float):
        self.A += np.outer(x, x)
        self.b += x * reward
        self.n += 1

    def to_dict(self) -> dict:
        return {"dim": self.dim, "lam": self.lam, "noise_var": self.noise_var,
                "A": self.A.tolist(), "b": self.b.tolist(), "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearArmModel":
        m = cls(dim=d["dim"], lam=d["lam"], noise_var=d["noise_var"])
        m.A = np.array(d["A"])
        m.b = np.array(d["b"])
        m.n = d["n"]
        return m


class LinearGlobalContextModel:
    """
    One LinearArmModel per arm, pooling reward data across ALL users.
    Drop-in replacement for hierarchical_bandit.GlobalContextPrior's role,
    but keyed by a continuous context feature vector instead of a
    discrete bucket string.
    """

    def __init__(self, arms: Optional[list[Arm]] = None, lam: float = 1.0, noise_var: float = 0.25):
        self.arms = arms if arms is not None else build_arm_grid()
        self.lam = lam
        self.noise_var = noise_var
        self.models: dict[tuple, LinearArmModel] = {
            a.as_tuple(): LinearArmModel(lam=lam, noise_var=noise_var) for a in self.arms
        }

    def get_prior(self, context_features: np.ndarray) -> dict:
        """{arm_tuple: (mean, var)} to seed a brand-new user's UserBandit,
        given this session's context."""
        return {key: model.posterior(context_features) for key, model in self.models.items()}

    def update(self, context_features: np.ndarray, arm: Arm, reward: float):
        self.models[arm.as_tuple()].update(context_features, reward)

    def n_observations(self) -> int:
        return sum(m.n for m in self.models.values())

    def to_dict(self) -> dict:
        return {
            "lam": self.lam,
            "noise_var": self.noise_var,
            "arms": [{"match_fraction": a.match_fraction, "max_jump": a.max_jump, "n_songs": a.n_songs}
                     for a in self.arms],
            "models": {",".join(map(str, k)): v.to_dict() for k, v in self.models.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinearGlobalContextModel":
        arms = [Arm(**a) for a in d["arms"]]
        gm = cls(arms=arms, lam=d["lam"], noise_var=d["noise_var"])
        for k_str, v in d["models"].items():
            parts = k_str.split(",")
            key = (float(parts[0]), float(parts[1]), int(parts[2]))
            gm.models[key] = LinearArmModel.from_dict(v)
        return gm

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LinearGlobalContextModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Manager tying per-user bandits + the linear global model together
# ---------------------------------------------------------------------------

class LinearHierarchicalBanditManager:
    """Same role as hierarchical_bandit.HierarchicalBanditManager, but
    warm-starting new users from LinearGlobalContextModel instead of a
    discrete-bucket GlobalContextPrior."""

    def __init__(self, base_dir: str, arms: Optional[list[Arm]] = None):
        self.arms = arms if arms is not None else build_arm_grid()
        self.user_store = BanditStore(base_dir=os.path.join(base_dir, "users"))
        self._global_path = os.path.join(base_dir, "linear_context_model.json")
        if os.path.exists(self._global_path):
            self.global_model = LinearGlobalContextModel.load(self._global_path)
        else:
            self.global_model = LinearGlobalContextModel(arms=self.arms)

    def get_bandit_for_session(self, user_id: str, start: tuple, target: tuple,
                                hour_of_day: Optional[int] = None):
        x = build_context_features(start, target, hour_of_day)
        user_path = self.user_store._path(user_id)
        is_new_user = not os.path.exists(user_path)

        if is_new_user:
            arm_priors = self.global_model.get_prior(x)
            bandit = UserBandit(user_id, arms=self.arms, arm_priors=arm_priors)
        else:
            bandit = self.user_store.get(user_id)

        return bandit, x, is_new_user

    def close_session(self, bandit: UserBandit, context_features: np.ndarray, arm: Arm, reward: float):
        bandit.update(arm, reward)
        self.global_model.update(context_features, arm, reward)
        self.user_store.save(bandit)
        self.global_model.save(self._global_path)
