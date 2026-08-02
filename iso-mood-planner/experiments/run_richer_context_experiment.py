"""
Richer-context ablation: does moving to a linear contextual model actually
help once we add mood_category and time_of_day on top of mood_distance --
or does naively extending the discrete-bucket approach do just as well?

Three conditions, run over the IDENTICAL sequence of simulated new users
(each with a randomly sampled context: start/target mood + hour of day,
and a true preference that genuinely depends on that context):

- NO SHARING:      fresh flat-prior UserBandit every time (baseline).
- EXTENDED BUCKETS: hierarchical_bandit.HierarchicalBanditManager, but
                    bucketed by (mood_distance x mood_category x
                    time_of_day) = up to 3*4*5 = 60 composite cells.
                    Naive generalization of the original approach.
- LINEAR CONTEXT:   contextual_bandit.LinearHierarchicalBanditManager,
                    sharing via a linear reward model over the full
                    continuous/one-hot context feature vector.

Expectation: EXTENDED BUCKETS should show WEAKER early gains than the
original single-feature bucketing (fragmentation -- each of ~60 cells
needs its own population), while LINEAR CONTEXT should show gains that
hold up (or improve) despite the richer context, since it shares
coefficients across all contexts rather than counting per-cell.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iso_planner.bandit import UserBandit, build_arm_grid
from experiments.persona_simulator import Persona
from iso_planner.hierarchical_bandit import (
    HierarchicalBanditManager, mood_distance, bucketize_mood_distance,
)
from iso_planner.contextual_bandit import (
    LinearHierarchicalBanditManager, build_context_features,
    mood_category_of, time_bucket_of, MOOD_CATEGORIES, TIME_BUCKETS,
)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
T_FIRST = 15
N_USERS = 60

ALL_COMPOSITE_BUCKETS = tuple(
    f"{d}|{c}|{t}"
    for d in ("small", "medium", "large")
    for c in MOOD_CATEGORIES
    for t in TIME_BUCKETS
)


def composite_bucket_fn(start, target, hour_of_day=None, **kwargs):
    d = bucketize_mood_distance(mood_distance(start, target))
    c = mood_category_of(*start)
    t = time_bucket_of(hour_of_day)
    return f"{d}|{c}|{t}"


def true_arm_params_for_context(start, target, hour_of_day, rng):
    """
    Structural population model: true preference genuinely depends on
    mood_distance, mood_category (derived from the start mood), AND
    time_of_day -- so a sharing mechanism that ignores 2 of these 3
    signals is leaving real, learnable structure on the table.
    """
    d = mood_distance(start, target) / np.sqrt(2)
    cat = mood_category_of(*start)
    tb = time_bucket_of(hour_of_day)

    match_fraction = 0.20 + 0.20 * d
    max_jump = 0.20 - 0.10 * d
    n_songs = 8 + 6 * d

    if cat == "sad":
        match_fraction += 0.05; max_jump -= 0.03
    elif cat == "angry":
        max_jump -= 0.05
    elif cat == "content":
        match_fraction -= 0.05
    elif cat == "excited":
        max_jump += 0.05

    if tb == "night":
        match_fraction += 0.05; max_jump -= 0.03
    elif tb == "morning":
        match_fraction -= 0.03

    match_fraction = float(np.clip(match_fraction + rng.normal(0, 0.03), 0.05, 0.5))
    max_jump = float(np.clip(max_jump + rng.normal(0, 0.02), 0.05, 0.3))
    n_songs = int(np.clip(round(n_songs + rng.normal(0, 1.5)), 4, 20))
    return match_fraction, max_jump, n_songs


def sample_context_and_persona(idx, rng):
    start = (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95)))
    target = (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95)))
    hour_of_day = int(rng.integers(0, 24))
    mf, mj, n = true_arm_params_for_context(start, target, hour_of_day, rng)
    persona = Persona(f"user_{idx}", true_match_fraction=mf, true_max_jump=mj,
                       true_n_songs=n, noise_std=0.08, rng_seed=3000 + idx)
    return start, target, hour_of_day, persona


def oracle_quality(persona, arms):
    return max(persona.quality(a) for a in arms)


def run_baseline(persona, arms, rng):
    bandit = UserBandit(user_id=persona.name, arms=arms)
    best_q = oracle_quality(persona, arms)
    regrets = []
    for _ in range(T_FIRST):
        arm = bandit.select_arm(rng=rng)
        reward, _ = persona.simulate_session(arm)
        bandit.update(arm, reward)
        regrets.append(best_q - persona.quality(arm))
    return float(np.mean(regrets))


def run_bucket_sharing(persona, start, target, hour_of_day, manager, arms, rng):
    bandit, bucket, _ = manager.get_bandit_for_session(persona.name, start, target, hour_of_day=hour_of_day)
    best_q = oracle_quality(persona, arms)
    regrets = []
    for _ in range(T_FIRST):
        arm = bandit.select_arm(rng=rng)
        reward, _ = persona.simulate_session(arm)
        manager.close_session(bandit, bucket, arm, reward)
        regrets.append(best_q - persona.quality(arm))
    return float(np.mean(regrets))


def run_linear_sharing(persona, start, target, hour_of_day, manager, arms, rng):
    bandit, ctx, _ = manager.get_bandit_for_session(persona.name, start, target, hour_of_day=hour_of_day)
    best_q = oracle_quality(persona, arms)
    regrets = []
    for _ in range(T_FIRST):
        arm = bandit.select_arm(rng=rng)
        reward, _ = persona.simulate_session(arm)
        manager.close_session(bandit, ctx, arm, reward)
        regrets.append(best_q - persona.quality(arm))
    return float(np.mean(regrets))


def main():
    arms = build_arm_grid()
    rng = np.random.default_rng(7)

    bucket_dir = "/tmp/extended_bucket_demo"
    linear_dir = "/tmp/linear_context_demo"
    shutil.rmtree(bucket_dir, ignore_errors=True)
    shutil.rmtree(linear_dir, ignore_errors=True)

    bucket_manager = HierarchicalBanditManager(
        base_dir=bucket_dir, arms=arms,
        buckets=ALL_COMPOSITE_BUCKETS, bucket_fn=composite_bucket_fn,
    )
    linear_manager = LinearHierarchicalBanditManager(base_dir=linear_dir, arms=arms)

    rows = []
    for i in range(N_USERS):
        start, target, hour_of_day, persona = sample_context_and_persona(i, rng)

        baseline_regret = run_baseline(persona, arms, rng)
        bucket_regret = run_bucket_sharing(persona, start, target, hour_of_day, bucket_manager, arms, rng)
        linear_regret = run_linear_sharing(persona, start, target, hour_of_day, linear_manager, arms, rng)

        rows.append({
            "user_index": i + 1,
            "no_sharing": baseline_regret,
            "extended_buckets": bucket_regret,
            "linear_context": linear_regret,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/richer_context_ablation.csv", index=False)

    def rolling(x, w=8):
        return pd.Series(x).rolling(w, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"no_sharing": "tab:gray", "extended_buckets": "tab:orange", "linear_context": "tab:blue"}
    labels = {"no_sharing": "no sharing (baseline)",
              "extended_buckets": "extended discrete buckets (60 cells)",
              "linear_context": "linear contextual model"}
    for col in ["no_sharing", "extended_buckets", "linear_context"]:
        ax.plot(df["user_index"], df[col], "o", alpha=0.15, color=colors[col])
        ax.plot(df["user_index"], rolling(df[col]), color=colors[col], lw=2, label=labels[col])
    ax.set_xlabel("new-user arrival order")
    ax.set_ylabel(f"mean regret over first {T_FIRST} sessions")
    ax.set_title("Richer context: linear model vs. naive bucket extension vs. no sharing")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/richer_context_ablation.png", dpi=130)
    plt.close(fig)

    print(df.to_string(index=False))
    print(f"\nMean regret, first 15 arrivals -- "
          f"no_sharing: {df['no_sharing'][:15].mean():.4f} | "
          f"extended_buckets: {df['extended_buckets'][:15].mean():.4f} | "
          f"linear_context: {df['linear_context'][:15].mean():.4f}")
    print(f"Mean regret, last 15 arrivals  -- "
          f"no_sharing: {df['no_sharing'][-15:].mean():.4f} | "
          f"extended_buckets: {df['extended_buckets'][-15:].mean():.4f} | "
          f"linear_context: {df['linear_context'][-15:].mean():.4f}")


if __name__ == "__main__":
    main()
