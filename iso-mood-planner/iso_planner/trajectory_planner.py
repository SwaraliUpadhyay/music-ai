"""
ISO-Trajectory Planner
-----------------------
Plans a smooth, personalized song sequence that moves a listener's mood
from a starting valence-arousal state to a target state, operationalizing
the music-therapy "iso principle" (match current mood, then gradually shift).
"""

import pandas as pd
import numpy as np


def ease_in_out(t):
    """Smoothstep easing: slow start, fast middle, slow end. t in [0,1]."""
    return t * t * (3 - 2 * t)


def generate_trajectory(start, target, n_songs, match_fraction=0.25):
    """
    Generate N ideal (valence, arousal) waypoints from start to target.

    match_fraction: proportion of the playlist that stays near the start
                     mood before the transition begins (the 'iso' matching
                     phase). E.g. 0.25 -> first 25% of songs stay close to
                     the starting mood.
    """
    v0, a0 = start
    vT, aT = target
    waypoints = []
    match_n = max(1, int(n_songs * match_fraction))

    for i in range(n_songs):
        if i < match_n:
            # matching phase: stay near start, tiny controlled drift
            t_local = i / max(1, match_n - 1) if match_n > 1 else 0
            v = v0 + 0.05 * t_local * (vT - v0)
            a = a0 + 0.05 * t_local * (aT - a0)
        else:
            # transition phase: eased interpolation to target
            t_local = (i - match_n) / max(1, (n_songs - match_n - 1))
            t_eased = ease_in_out(min(1.0, max(0.0, t_local)))
            v = v0 + t_eased * (vT - v0)
            a = a0 + t_eased * (aT - a0)
        waypoints.append((v, a))
    return waypoints


def select_next_candidate(pool: pd.DataFrame, used: set, prev_va, waypoint, max_jump: float):
    """
    Atomic selection step: pick the single best next song from `pool`
    (excluding indices in `used`) for one waypoint, subject to a max_jump
    cap relative to `prev_va` (the previous song's (valence, arousal), or
    None for the first song in a playlist).

    If no candidate satisfies the cap, it is relaxed (x1.5) until at least
    one candidate qualifies, so selection never fails outright.

    Factored out so both plan_playlist() (calls it once per waypoint,
    upfront, for a full batch playlist) and session_controller.py's
    SessionController (calls it one song at a time, with max_jump
    dynamically tightened/relaxed between calls by the in-session
    guardrail in reaction to real-time skip/replay feedback) share
    exactly the same selection rule.

    Returns (row, index, cap_used) -- cap_used is the effective cap after
    any auto-relaxation, exposed so callers can log/inspect it.
    """
    wv, wa = waypoint
    candidates = pool[~pool.index.isin(used)].copy()
    d_curve = np.sqrt((candidates['valence'] - wv) ** 2 + (candidates['arousal'] - wa) ** 2)

    cap_used = max_jump
    if prev_va is not None:
        d_prev = np.sqrt((candidates['valence'] - prev_va[0]) ** 2 +
                          (candidates['arousal'] - prev_va[1]) ** 2)
        cap = max_jump
        allowed = candidates[d_prev <= cap]
        while allowed.empty:
            cap *= 1.5
            allowed = candidates[d_prev <= cap]
        cap_used = cap
        candidates = allowed
        d_curve = d_curve.loc[candidates.index]

    best_idx = d_curve.idxmin()
    row = pool.loc[best_idx]
    return row, best_idx, cap_used


def plan_playlist(song_pool: pd.DataFrame, start, target, n_songs=10,
                   match_fraction=0.25, max_jump=0.15):
    """
    Select an ordered list of real songs from song_pool that follows the
    ideal trajectory from start -> target, while enforcing a hard cap on
    how far (in valence-arousal space) consecutive songs can be from each
    other -- this is what operationalizes 'gradual' per the iso principle.

    song_pool: DataFrame with columns ['song_id','title','artist','valence','arousal']
    max_jump: max allowed euclidean V-A distance between consecutive songs.
              If no candidate satisfies it, the cap is relaxed (x1.5) until
              at least one candidate qualifies, so the planner never fails outright.
    """
    waypoints = generate_trajectory(start, target, n_songs, match_fraction)
    pool = song_pool.copy().reset_index(drop=True)
    used = set()
    selected = []
    prev_va = None

    for wp in waypoints:
        row, best_idx, _cap_used = select_next_candidate(pool, used, prev_va, wp, max_jump)
        selected.append({
            'song_id': row['song_id'], 'title': row['title'], 'artist': row['artist'],
            'valence': row['valence'], 'arousal': row['arousal'],
            'target_valence': round(wp[0], 3), 'target_arousal': round(wp[1], 3),
            'source': row.get('source', '')
        })
        used.add(best_idx)
        prev_va = (row['valence'], row['arousal'])

    return pd.DataFrame(selected)


if __name__ == "__main__":
    import os
    pool = pd.read_csv('data/combined_song_pool.csv')

    # Example: sad/melancholic -> calm & positive (anxiety-reduction use case)
    start_mood = (0.20, 0.30)   # low valence, low-moderate arousal = sad
    target_mood = (0.75, 0.35)  # high valence, low arousal = calm & content

    playlist = plan_playlist(pool, start_mood, target_mood, n_songs=10, match_fraction=0.25)
    pd.set_option('display.width', 120)
    print(playlist[['title', 'artist', 'valence', 'arousal', 'target_valence', 'target_arousal', 'source']])

    os.makedirs('outputs', exist_ok=True)
    playlist.to_csv('outputs/example_playlist.csv', index=False)
