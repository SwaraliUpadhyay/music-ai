"""
Composite session reward.

R = w1 * mood_progress + w2 * (replay_rate - skip_rate) + w3 * rating_norm

Weights renormalize automatically over whichever signals are actually
available for a given session (e.g. a user who skips the post-session
mood survey still contributes a reward from skip/replay behavior alone,
rather than being silently penalized with an imputed 0).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionSignals:
    mood_progress: Optional[float] = None   # in [0,1]; None if no pre/post self-report
    skip_rate: Optional[float] = None       # in [0,1]; fraction of songs skipped
    replay_rate: Optional[float] = None     # in [0,1]; fraction of songs replayed
    rating: Optional[float] = None          # 1-5 explicit rating, or None


DEFAULT_WEIGHTS = {"mood": 0.5, "behavior": 0.3, "rating": 0.2}


def compute_reward(signals: SessionSignals, weights: dict = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    terms = []
    weight_sum = 0.0

    if signals.mood_progress is not None:
        terms.append(weights["mood"] * signals.mood_progress)  # already [0,1]
        weight_sum += weights["mood"]

    if signals.skip_rate is not None and signals.replay_rate is not None:
        behavior = (signals.replay_rate - signals.skip_rate)  # in [-1,1]
        behavior_01 = (behavior + 1) / 2  # rescale to [0,1] to match other terms
        terms.append(weights["behavior"] * behavior_01)
        weight_sum += weights["behavior"]

    if signals.rating is not None:
        rating_01 = (signals.rating - 1) / 4  # 1-5 -> 0-1
        terms.append(weights["rating"] * rating_01)
        weight_sum += weights["rating"]

    if weight_sum == 0:
        return 0.0

    # Renormalize by whichever weights were actually used, then shift to
    # [-1, 1] so "neutral" sessions sit at 0 for the bandit's prior.
    r_01 = sum(terms) / weight_sum
    return 2 * r_01 - 1
