"""
Per-User Personalization Bandit
--------------------------------
Learns personalized (match_fraction, max_jump, session_length) parameters
for the iso-principle trajectory planner using Gaussian Thompson Sampling
over a discretized arm grid.

Design rationale (see project discussion):
- Each "session" (one generated playlist) is a single decision -> single
  delayed scalar reward. There is no meaningful cross-session state
  transition to model, so a contextual multi-armed bandit is preferred
  over full RL: it converges faster, needs far fewer episodes, and is
  more interpretable / defensible in a paper than an MDP formulation.
- Reward is continuous (a weighted composite of mood self-report and
  behavioral signals -- see reward.py), so Gaussian (Normal-Normal)
  Thompson Sampling is used rather than Beta-Bernoulli, which would
  force binarizing a continuous outcome and throw away information.
- session_length (N) can be fixed by the user for a given session (in
  which case it is *context*, not a learned choice -- the bandit only
  optimizes over the remaining dimensions for that N) or left unset
  (in which case N is also a learned dimension with a personalized
  default).
"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Arm grid
# ---------------------------------------------------------------------------

DEFAULT_MATCH_FRACTIONS = (0.10, 0.25, 0.40)
DEFAULT_MAX_JUMPS = (0.10, 0.15, 0.25)
DEFAULT_N_CHOICES = (6, 10, 15)

# Structural floor: below this, there isn't room for a matching phase AND
# a meaningful eased transition. Enforced regardless of what a user picks.
N_MIN = 4


@dataclass(frozen=True)
class Arm:
    match_fraction: float
    max_jump: float
    n_songs: int

    def as_tuple(self):
        return (self.match_fraction, self.max_jump, self.n_songs)


def build_arm_grid(match_fractions: Sequence[float] = DEFAULT_MATCH_FRACTIONS,
                    max_jumps: Sequence[float] = DEFAULT_MAX_JUMPS,
                    n_choices: Sequence[int] = DEFAULT_N_CHOICES) -> list[Arm]:
    """All combinations of (match_fraction, max_jump, n_songs)."""
    return [Arm(mf, mj, n) for mf, mj, n in product(match_fractions, max_jumps, n_choices)]


def clamp_session_length(n_requested: Optional[int], n_min: int = N_MIN) -> Optional[int]:
    """Enforce the structural minimum session length if the user requested one."""
    if n_requested is None:
        return None
    return max(n_min, int(n_requested))


# ---------------------------------------------------------------------------
# Gaussian Thompson Sampling bandit
# ---------------------------------------------------------------------------

@dataclass
class ArmStats:
    n: int = 0
    sum_reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.sum_reward / self.n if self.n > 0 else 0.0


class UserBandit:
    """
    One instance per user. Maintains Normal-Normal conjugate posteriors
    over the mean reward of each arm in the grid.

    Prior:      mean_k ~ N(prior_mean, prior_var)
    Likelihood: reward | mean_k ~ N(mean_k, noise_var)   (noise_var assumed known/fixed)
    Posterior:  mean_k | data ~ N(post_mean_k, post_var_k)   (closed form, updated online)
    """

    def __init__(self,
                 user_id: str,
                 arms: Optional[list[Arm]] = None,
                 prior_mean: float = 0.0,
                 prior_var: float = 1.0,
                 noise_var: float = 0.25,
                 arm_priors: Optional[dict] = None):
        """
        arm_priors: optional {arm_tuple: (mean, var)} overriding the flat
        (prior_mean, prior_var) on a per-arm basis. This is how
        hierarchical_bandit.py warm-starts a brand-new user from a
        population-level (context-bucketed) posterior instead of a blank
        flat prior -- see hierarchical_bandit.HierarchicalBanditManager.
        """
        self.user_id = user_id
        self.arms = arms if arms is not None else build_arm_grid()
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.arm_priors = arm_priors or {}
        self.stats: dict[tuple, ArmStats] = {a.as_tuple(): ArmStats() for a in self.arms}
        self.history: list[dict] = []  # log of (arm, reward, session_index)

    # -- posterior math ----------------------------------------------------

    def _posterior(self, arm_key: tuple) -> tuple[float, float]:
        s = self.stats[arm_key]
        p_mean, p_var = self.arm_priors.get(arm_key, (self.prior_mean, self.prior_var))
        post_var = 1.0 / (1.0 / p_var + s.n / self.noise_var)
        post_mean = post_var * (p_mean / p_var + s.n * s.mean / self.noise_var)
        return post_mean, post_var

    # -- selection -----------------------------------------------------------

    def select_arm(self, fixed_n: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> Arm:
        """
        Thompson-sample a reward estimate for every eligible arm and return
        the arm with the highest sample.

        fixed_n: if the user explicitly chose a session length this
                 session, only arms with that n_songs are eligible
                 (N becomes context, not a learned choice, for this call).
        """
        rng = rng or np.random.default_rng()
        fixed_n = clamp_session_length(fixed_n)

        eligible = [a for a in self.arms if (fixed_n is None or a.n_songs == fixed_n)]
        if not eligible:
            # fixed_n didn't match any grid value (e.g. user typed an odd
            # number) -> fall back to the closest available n_songs choice
            all_n = sorted({a.n_songs for a in self.arms})
            closest_n = min(all_n, key=lambda n: abs(n - fixed_n))
            eligible = [a for a in self.arms if a.n_songs == closest_n]

        samples = []
        for a in eligible:
            mu, var = self._posterior(a.as_tuple())
            samples.append(rng.normal(mu, math.sqrt(max(var, 1e-9))))
        best = eligible[int(np.argmax(samples))]
        return best

    # -- update --------------------------------------------------------------

    def update(self, arm: Arm, reward: float):
        key = arm.as_tuple()
        s = self.stats[key]
        s.n += 1
        s.sum_reward += reward
        self.history.append({
            "session_index": len(self.history),
            "arm": key,
            "reward": reward,
        })

    # -- introspection ---------------------------------------------------

    def posterior_summary(self):
        import pandas as pd
        rows = []
        for a in self.arms:
            key = a.as_tuple()
            mu, var = self._posterior(key)
            s = self.stats[key]
            rows.append({
                "match_fraction": a.match_fraction,
                "max_jump": a.max_jump,
                "n_songs": a.n_songs,
                "n_pulls": s.n,
                "empirical_mean": s.mean,
                "posterior_mean": mu,
                "posterior_std": math.sqrt(var),
            })
        return pd.DataFrame(rows).sort_values("posterior_mean", ascending=False).reset_index(drop=True)

    def best_arm_estimate(self) -> Arm:
        """Greedy (no exploration) best arm by posterior mean -- useful for
        reporting a user's 'converged' personalized defaults."""
        best, best_mu = None, -math.inf
        for a in self.arms:
            mu, _ = self._posterior(a.as_tuple())
            if mu > best_mu:
                best, best_mu = a, mu
        return best

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "noise_var": self.noise_var,
            "arms": [asdict(a) for a in self.arms],
            "stats": {",".join(map(str, k)): asdict(v) for k, v in self.stats.items()},
            "arm_priors": {",".join(map(str, k)): list(v) for k, v in self.arm_priors.items()},
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserBandit":
        arms = [Arm(**a) for a in d["arms"]]
        arm_priors = {}
        for k_str, v in d.get("arm_priors", {}).items():
            parts = k_str.split(",")
            key = (float(parts[0]), float(parts[1]), int(parts[2]))
            arm_priors[key] = tuple(v)
        b = cls(d["user_id"], arms=arms, prior_mean=d["prior_mean"],
                 prior_var=d["prior_var"], noise_var=d["noise_var"], arm_priors=arm_priors)
        for k_str, v in d["stats"].items():
            parts = k_str.split(",")
            key = (float(parts[0]), float(parts[1]), int(parts[2]))
            b.stats[key] = ArmStats(**v)
        b.history = d.get("history", [])
        return b

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserBandit":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class BanditStore:
    """Simple on-disk registry mapping user_id -> UserBandit (JSON files)."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        return os.path.join(self.base_dir, f"{user_id}.json")

    def get(self, user_id: str) -> UserBandit:
        path = self._path(user_id)
        if os.path.exists(path):
            return UserBandit.load(path)
        return UserBandit(user_id)

    def save(self, bandit: UserBandit):
        bandit.save(self._path(bandit.user_id))
