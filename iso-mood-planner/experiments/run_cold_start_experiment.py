"""
Cold-start experiment: simulate a stream of NEW users, all requesting a
"large mood swing" session (same context bucket), arriving one at a time.

Two conditions, run over the identical sequence of simulated users:
- BASELINE: every new user gets a fresh flat-prior UserBandit (today's
  behavior -- no sharing across users at all).
- WITH SHARING: every new user is warm-started via
  HierarchicalBanditManager from the current global context prior for
  the 'large' bucket, which itself keeps improving as more users'
  sessions are folded into it.

For each new user we measure mean regret (vs. that user's own oracle-best
arm) over just their FIRST 15 sessions -- i.e. the cold-start window,
before their individual posterior has had much chance to learn on its
own. If sharing helps, this should trend down as more users arrive
(baseline should stay flat/noisy, since it never accumulates anything
across users).
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iso_planner.bandit import UserBandit, build_arm_grid
from experiments.persona_simulator import Persona
from iso_planner.hierarchical_bandit import HierarchicalBanditManager, mood_distance, bucketize_mood_distance

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
T_FIRST = 15          # cold-start window measured per user
N_USERS = 40          # users arriving sequentially
BUCKET_START, BUCKET_TARGET = (0.10, 0.10), (0.90, 0.90)  # distance ~1.13 -> 'large'

# Population distribution for this bucket: individuals vary around these,
# representing "people with big mood swings tend to need a longer matching
# phase and gentler jumps" -- a population-level tendency the hierarchical
# prior should be able to pick up on across users.
POP_MATCH_FRACTION = (0.35, 0.07)   # (mean, std)
POP_MAX_JUMP = (0.12, 0.04)
POP_N_SONGS = (13, 2)


def sample_user_persona(idx: int, rng: np.random.Generator) -> Persona:
    mf = float(np.clip(rng.normal(*POP_MATCH_FRACTION), 0.05, 0.5))
    mj = float(np.clip(rng.normal(*POP_MAX_JUMP), 0.05, 0.3))
    n = int(np.clip(round(rng.normal(*POP_N_SONGS)), 4, 20))
    return Persona(f"user_{idx}", true_match_fraction=mf, true_max_jump=mj,
                    true_n_songs=n, noise_std=0.08, rng_seed=1000 + idx)


def oracle_quality(persona, arms):
    return max(persona.quality(a) for a in arms)


def run_user_baseline(persona, arms, T=T_FIRST, rng=None):
    bandit = UserBandit(user_id=persona.name, arms=arms)  # flat prior, no sharing
    rng = rng or np.random.default_rng()
    best_q = oracle_quality(persona, arms)
    regrets = []
    for _ in range(T):
        arm = bandit.select_arm(rng=rng)
        reward, _ = persona.simulate_session(arm)
        bandit.update(arm, reward)
        regrets.append(best_q - persona.quality(arm))
    return float(np.mean(regrets))


def run_user_with_sharing(persona, manager: HierarchicalBanditManager, arms, T=T_FIRST, rng=None):
    rng = rng or np.random.default_rng()
    bandit, bucket, is_new = manager.get_bandit_for_session(persona.name, BUCKET_START, BUCKET_TARGET)
    best_q = oracle_quality(persona, arms)
    regrets = []
    for _ in range(T):
        arm = bandit.select_arm(rng=rng)
        reward, _ = persona.simulate_session(arm)
        manager.close_session(bandit, bucket, arm, reward)
        regrets.append(best_q - persona.quality(arm))
    return float(np.mean(regrets))


def main():
    arms = build_arm_grid()
    rng = np.random.default_rng(42)

    tmp_dir = "/tmp/hierarchical_bandit_demo"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    manager = HierarchicalBanditManager(base_dir=tmp_dir, arms=arms)

    assert bucketize_mood_distance(mood_distance(BUCKET_START, BUCKET_TARGET)) == "large"

    baseline_regret, sharing_regret = [], []
    for i in range(N_USERS):
        persona = sample_user_persona(i, rng)
        baseline_regret.append(run_user_baseline(persona, arms, rng=rng))
        sharing_regret.append(run_user_with_sharing(persona, manager, arms, rng=rng))

    df = pd.DataFrame({
        "user_index": range(1, N_USERS + 1),
        "baseline_regret_first15": baseline_regret,
        "sharing_regret_first15": sharing_regret,
    })
    df.to_csv(f"{OUT_DIR}/cold_start_regret.csv", index=False)

    def rolling(x, w=5):
        return pd.Series(x).rolling(w, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["user_index"], df["baseline_regret_first15"], "o", alpha=0.3, color="tab:gray")
    ax.plot(df["user_index"], rolling(df["baseline_regret_first15"]), color="tab:gray",
            lw=2, label="baseline (no sharing), rolling mean")
    ax.plot(df["user_index"], df["sharing_regret_first15"], "o", alpha=0.3, color="tab:blue")
    ax.plot(df["user_index"], rolling(df["sharing_regret_first15"]), color="tab:blue",
            lw=2, label="with hierarchical sharing, rolling mean")
    ax.set_xlabel("new-user arrival order")
    ax.set_ylabel(f"mean regret over first {T_FIRST} sessions")
    ax.set_title("Cold-start regret as more users' data accumulates (bucket = 'large' mood swing)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cold_start_regret.png", dpi=130)
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nMean regret, first 10 arrivals  -- baseline: {np.mean(baseline_regret[:10]):.4f} | "
          f"sharing: {np.mean(sharing_regret[:10]):.4f}")
    print(f"Mean regret, last 10 arrivals   -- baseline: {np.mean(baseline_regret[-10:]):.4f} | "
          f"sharing: {np.mean(sharing_regret[-10:]):.4f}")


if __name__ == "__main__":
    main()
