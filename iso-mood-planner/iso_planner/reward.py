"""
Composite session reward.

R = w1 * mood_progress + w2 * (replay_rate - skip_rate) + w3 * rating_norm
    + w4 * (1 - guardrail_load)

Weights renormalize automatically over whichever signals are actually
available for a given session (e.g. a user who skips the post-session
mood survey still contributes a reward from skip/replay behavior alone,
rather than being silently penalized with an imputed 0).

guardrail_load is optional and, when absent, the reward is identical to what
it was before this term existed, so older experiments are unaffected.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionSignals:
    mood_progress: Optional[float] = None   # in [0,1]; None if no pre/post self-report
    skip_rate: Optional[float] = None       # in [0,1]; fraction of songs skipped
    replay_rate: Optional[float] = None     # in [0,1]; fraction of songs replayed
    rating: Optional[float] = None          # 1-5 explicit rating, or None
    # How much of the playlist was played under a tightened step cap, in [0,1]
    # (SessionController.guardrail_load()). 0 = the guardrail never had to
    # tighten, 1 = pinned at its floor the whole session. It tracks skip_rate
    # closely but is not the same thing: a skip on the very last song costs
    # nothing (no later song is affected), an early skip costs more than a late
    # one, and replays after a skip shorten the penalty. None if unknown.
    guardrail_load: Optional[float] = None


DEFAULT_WEIGHTS = {"mood": 0.5, "behavior": 0.3, "rating": 0.2, "guardrail": 0.15}


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

    if signals.guardrail_load is not None:
        # "comfort": 1 when the guardrail never had to step in, 0 when it was
        # pinned at its floor. Counts against the arm (match_fraction, max_jump,
        # n_songs) that set the pace the listener kept fighting.
        load = min(1.0, max(0.0, signals.guardrail_load))
        w = weights.get("guardrail", DEFAULT_WEIGHTS["guardrail"])
        terms.append(w * (1.0 - load))
        weight_sum += w

    if weight_sum == 0:
        return 0.0

    # Renormalize by whichever weights were actually used, then shift to
    # [-1, 1] so "neutral" sessions sit at 0 for the bandit's prior.
    r_01 = sum(terms) / weight_sum
    return 2 * r_01 - 1
