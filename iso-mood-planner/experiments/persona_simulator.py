"""
Synthetic personas + session simulator.

Each persona has a "true" latent preference (match_fraction*, max_jump*,
n_songs*) that a real user would never report directly -- the bandit has
to discover it purely from simulated skip/replay/rating/mood-progress
feedback. This lets us test whether Thompson Sampling actually converges
to sensible per-user parameters before any real user study exists.
"""

from dataclasses import dataclass
import numpy as np

from iso_planner.bandit import Arm, DEFAULT_MATCH_FRACTIONS, DEFAULT_MAX_JUMPS, DEFAULT_N_CHOICES
from iso_planner.reward import SessionSignals, compute_reward

# Normalization ranges used to turn absolute parameter distance into a
# unitless quality score in ~[0,1].
_MF_RANGE = max(DEFAULT_MATCH_FRACTIONS) - min(DEFAULT_MATCH_FRACTIONS)   # 0.30
_MJ_RANGE = max(DEFAULT_MAX_JUMPS) - min(DEFAULT_MAX_JUMPS)               # 0.15
_N_RANGE = max(DEFAULT_N_CHOICES) - min(DEFAULT_N_CHOICES)                 # 9


@dataclass
class Persona:
    name: str
    true_match_fraction: float
    true_max_jump: float
    true_n_songs: int
    noise_std: float = 0.08   # how noisy/inconsistent this persona's feedback is
    rng_seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)

    def quality(self, arm: Arm) -> float:
        """1.0 = arm exactly matches this persona's true preference,
        decaying with normalized Euclidean distance in parameter space."""
        d_mf = (arm.match_fraction - self.true_match_fraction) / _MF_RANGE
        d_mj = (arm.max_jump - self.true_max_jump) / _MJ_RANGE
        d_n = (arm.n_songs - self.true_n_songs) / _N_RANGE
        dist = np.sqrt(d_mf**2 + d_mj**2 + d_n**2) / np.sqrt(3)
        return float(np.clip(1 - dist, 0.0, 1.0))

    def simulate_session(self, arm: Arm) -> tuple[float, SessionSignals]:
        """Simulate skip/replay/rating/mood-progress for one playlist under `arm`,
        then compute the composite reward via reward.compute_reward."""
        q = self.quality(arm)
        noisy_q = float(np.clip(q + self.rng.normal(0, self.noise_std), 0.0, 1.0))

        mood_progress = noisy_q
        # higher quality -> fewer skips, more replays (both scaled to plausible
        # behavioral ranges, not the full [0,1], since real users rarely skip
        # or replay *every* song)
        skip_rate = float(np.clip((1 - noisy_q) * 0.6 + self.rng.normal(0, 0.05), 0, 1))
        replay_rate = float(np.clip(noisy_q * 0.5 + self.rng.normal(0, 0.05), 0, 1))
        rating = float(np.clip(1 + 4 * noisy_q + self.rng.normal(0, 0.3), 1, 5))

        signals = SessionSignals(
            mood_progress=mood_progress,
            skip_rate=skip_rate,
            replay_rate=replay_rate,
            rating=rating,
        )
        reward = compute_reward(signals)
        return reward, signals


# ---------------------------------------------------------------------------
# Reference persona set used throughout the evaluation
# ---------------------------------------------------------------------------

def default_personas() -> list[Persona]:
    return [
        Persona("Fragile (needs long matching)", true_match_fraction=0.40,
                true_max_jump=0.10, true_n_songs=15, noise_std=0.06, rng_seed=1),
        Persona("Impatient (wants fast transition)", true_match_fraction=0.10,
                true_max_jump=0.25, true_n_songs=6, noise_std=0.06, rng_seed=2),
        Persona("Moderate (near current defaults)", true_match_fraction=0.25,
                true_max_jump=0.15, true_n_songs=10, noise_std=0.06, rng_seed=3),
        Persona("Noisy/inconsistent", true_match_fraction=0.25,
                true_max_jump=0.15, true_n_songs=10, noise_std=0.20, rng_seed=4),
    ]
