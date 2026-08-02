"""
Glue code: bandit-selected parameters -> real plan_playlist() call.

This is what a production session flow would look like:

    store = BanditStore(base_dir=".../bandits")
    bandit = store.get(user_id)
    arm = bandit.select_arm(fixed_n=user_requested_n)   # None if user didn't set N
    playlist = run_session(bandit, arm, pool, start_mood, target_mood)
    ... collect skip/replay/rating/mood-report from the user while it plays ...
    reward = compute_reward(SessionSignals(...))
    bandit.update(arm, reward)
    store.save(bandit)
"""

import pandas as pd

from iso_planner.bandit import Arm, UserBandit, BanditStore, clamp_session_length
from iso_planner.trajectory_planner import plan_playlist
from iso_planner.reward import SessionSignals, compute_reward


def run_session(bandit: UserBandit, pool: pd.DataFrame, start, target,
                 fixed_n: int | None = None, rng=None):
    """Select an arm via Thompson Sampling and generate the actual playlist."""
    arm = bandit.select_arm(fixed_n=fixed_n, rng=rng)
    playlist = plan_playlist(
        pool, start, target,
        n_songs=arm.n_songs,
        match_fraction=arm.match_fraction,
        max_jump=arm.max_jump,
    )
    return arm, playlist


def close_session(bandit: UserBandit, arm: Arm, signals: SessionSignals) -> float:
    """Score the session and feed it back into the bandit."""
    reward = compute_reward(signals)
    bandit.update(arm, reward)
    return reward


if __name__ == "__main__":
    pool = pd.read_csv("data/combined_song_pool.csv")
    store = BanditStore(base_dir="outputs/bandit_store_demo")

    user_id = "demo_user_1"
    bandit = store.get(user_id)

    start_mood = (0.20, 0.30)   # sad
    target_mood = (0.75, 0.35)  # calm & content

    # Session 1: user doesn't fix N -> bandit also picks session length
    arm, playlist = run_session(bandit, pool, start_mood, target_mood, fixed_n=None)
    print(f"Selected arm: match_fraction={arm.match_fraction}, "
          f"max_jump={arm.max_jump}, n_songs={arm.n_songs}")
    print(playlist[["title", "artist", "valence", "arousal",
                     "target_valence", "target_arousal"]].to_string(index=False))

    # Simulate feedback for this session (in production this comes from the
    # app: skip/replay counts, an explicit rating, and a pre/post mood survey)
    signals = SessionSignals(mood_progress=0.7, skip_rate=0.1, replay_rate=0.3, rating=4)
    reward = close_session(bandit, arm, signals)
    print(f"\nSession reward: {reward:.3f}")

    # Session 2: user explicitly requests a short 5-song session -> N is now
    # context (clamped to the structural floor if needed), only
    # (match_fraction, max_jump) are learned/selected for this call.
    arm2, playlist2 = run_session(bandit, pool, start_mood, target_mood, fixed_n=5)
    print(f"\n[Session 2, user fixed N=5 -> clamped to {arm2.n_songs}] "
          f"match_fraction={arm2.match_fraction}, max_jump={arm2.max_jump}")

    store.save(bandit)
    print(f"\nBandit state saved for '{user_id}'. Posterior summary (top 5 arms):")
    print(bandit.posterior_summary().head(5).to_string(index=False))
