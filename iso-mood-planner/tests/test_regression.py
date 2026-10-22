"""
Lightweight regression tests -- run with `pytest` or plain `python -m tests.test_regression`.
Not a full test suite, just enough to catch obvious breakage from future edits.
"""

import numpy as np
import pandas as pd

from iso_planner import (
    plan_playlist, generate_trajectory, select_next_candidate,
    UserBandit, build_arm_grid, InSessionGuardrail, SessionController,
    HierarchicalBanditManager, mood_distance, bucketize_mood_distance,
    SessionSignals, compute_reward, stratified_train_test_split,
    LinearGlobalContextModel, build_context_features, mood_category_of, time_bucket_of,
)


def _toy_pool(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "song_id": [f"song_{i}" for i in range(n)],
        "title": [f"Title {i}" for i in range(n)],
        "artist": ["Artist"] * n,
        "valence": rng.uniform(0, 1, n),
        "arousal": rng.uniform(0, 1, n),
        "source": ["toy"] * n,
    })


def test_plan_playlist_shape_and_uniqueness():
    pool = _toy_pool()
    out = plan_playlist(pool, start=(0.2, 0.3), target=(0.8, 0.4), n_songs=8)
    assert len(out) == 8
    assert out["song_id"].nunique() == 8  # no duplicate songs


def test_generate_trajectory_endpoints():
    wps = generate_trajectory((0.0, 0.0), (1.0, 1.0), n_songs=10, match_fraction=0.25)
    assert wps[0] == (0.0, 0.0)
    assert abs(wps[-1][0] - 1.0) < 1e-9 and abs(wps[-1][1] - 1.0) < 1e-9


def test_bandit_selects_and_updates():
    bandit = UserBandit("test_user", arms=build_arm_grid())
    rng = np.random.default_rng(0)
    arm = bandit.select_arm(rng=rng)
    bandit.update(arm, reward=0.5)
    assert bandit.stats[arm.as_tuple()].n == 1


def test_guardrail_tightens_on_skip_and_recovers():
    g = InSessionGuardrail(base_max_jump=0.15, tighten_factor=0.5, recovery_factor=0.4)
    assert g.effective_max_jump == 0.15
    g.register_feedback("skip")
    assert g.effective_max_jump < 0.15
    tightened = g.effective_max_jump
    g.register_feedback("replay")
    assert g.effective_max_jump > tightened


def test_mood_distance_bucketing():
    d = mood_distance((0.1, 0.1), (0.9, 0.9))
    assert bucketize_mood_distance(d) == "large"
    assert bucketize_mood_distance(0.1) == "small"


def test_reward_renormalizes_with_missing_signals():
    full = compute_reward(SessionSignals(mood_progress=0.8, skip_rate=0.1, replay_rate=0.3, rating=4))
    partial = compute_reward(SessionSignals(mood_progress=0.8))  # no behavioral/rating signals
    assert -1.0 <= full <= 1.0
    assert -1.0 <= partial <= 1.0


def test_stratified_split_covers_all_rows_no_overlap():
    pool = _toy_pool(n=200)
    pool["source"] = np.tile(["A", "B"], 100)
    train, test = stratified_train_test_split(pool, test_size=0.2, seed=1)
    assert len(train) + len(test) == len(pool)
    assert set(train["song_id"]).isdisjoint(set(test["song_id"]))
    # roughly 20% in test (allow slack for small-stratum rounding)
    assert 0.1 < len(test) / len(pool) < 0.3


def test_mood_category_and_time_bucket_helpers():
    assert mood_category_of(0.1, 0.1) == "sad"
    assert mood_category_of(0.1, 0.9) == "angry"
    assert mood_category_of(0.9, 0.1) == "content"
    assert mood_category_of(0.9, 0.9) == "excited"
    assert time_bucket_of(None) == "unknown"
    assert time_bucket_of(23) == "night"
    assert time_bucket_of(9) == "morning"


def test_linear_context_model_updates_and_shrinks_variance():
    arms = build_arm_grid()
    model = LinearGlobalContextModel(arms=arms)
    x = build_context_features((0.2, 0.3), (0.8, 0.4), hour_of_day=22)
    _, var_before = model.get_prior(x)[arms[0].as_tuple()]
    for _ in range(10):
        model.update(x, arms[0], 0.6)
    _, var_after = model.get_prior(x)[arms[0].as_tuple()]
    assert var_after < var_before  # more data -> less uncertainty about this arm's reward here


def test_classify_listening_thresholds():
    from iso_planner import classify_listening, SKIP_BELOW
    assert classify_listening(0.05) == "skip"
    assert classify_listening(SKIP_BELOW - 0.01) == "skip"
    assert classify_listening(SKIP_BELOW) == "neutral"
    assert classify_listening(1.0) == "neutral"
    assert classify_listening(0.1, replays=1) == "replay"   # played again beats leaving early
    assert classify_listening(7.0) == "neutral"             # clamped, not an error


def test_reward_guardrail_term_penalises_load_and_is_backward_compatible():
    base = SessionSignals(mood_progress=0.6, skip_rate=0.1, replay_rate=0.2, rating=4)
    r_old = compute_reward(base)
    calm = SessionSignals(0.6, 0.1, 0.2, 4, guardrail_load=0.0)
    stormy = SessionSignals(0.6, 0.1, 0.2, 4, guardrail_load=0.9)
    assert compute_reward(stormy) < compute_reward(calm)
    assert compute_reward(SessionSignals(0.6, 0.1, 0.2, 4, guardrail_load=None)) == r_old
    assert -1.0 <= compute_reward(stormy) <= 1.0


def test_session_controller_guardrail_load_and_triggers():
    pool = _toy_pool(200, seed=1)

    def run(fb):
        sc = SessionController(pool, (0.2, 0.3), (0.8, 0.6), n_songs=8, max_jump=0.3)
        for f in fb:
            sc.next_song(); sc.submit_feedback(f)
        return sc

    clean = run(["neutral"] * 8)
    early = run(["skip"] + ["neutral"] * 7)
    last = run(["neutral"] * 7 + ["skip"])
    allskip = run(["skip"] * 8)
    assert clean.guardrail_load() == 0.0 and clean.guardrail_triggers() == 0
    assert last.guardrail_load() == 0.0 and last.guardrail_triggers() == 1  # nothing after it to affect
    assert early.guardrail_load() > 0.0
    assert allskip.guardrail_load() > early.guardrail_load()
    assert allskip.guardrail_load() <= 1.0


if __name__ == "__main__":
    import sys
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
