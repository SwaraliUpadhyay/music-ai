"""
Within-Session Reactive Guardrail
-----------------------------------
The bandit (bandit.py) personalizes (match_fraction, max_jump, n_songs)
*between* sessions -- it needs a full session's worth of feedback before
it can update, so it can't react to "this specific song was a bad jump"
in the middle of a 10-song playlist. That's a real gap: if song 3 lands
badly, the listener still has to sit through songs 4-10 planned under the
same (possibly wrong-paced) settings before anything adjusts.

This module is the missing piece: a deterministic (non-learned) rule that
reacts immediately, within the current playlist, to skip/replay feedback
on the song that was just played, by tightening or relaxing the max_jump
cap used to pick the *next* song only.

Design:
- On skip: shrink the cap (tighten_factor < 1), pulling the next choice
  closer to the last accepted song -- treat a skip as "that step felt too
  big / wrong direction," and take a smaller, safer step next.
- On replay/like: no change needed (the pace was fine) -- but if the cap
  had been tightened by an earlier skip, let it relax partway back toward
  the base setting (recovery_factor), so a single bad reaction doesn't
  permanently slow down the rest of the playlist.
- On neutral (song just played through, no explicit signal): same partial
  recovery as replay, just slower -- treated as a weak positive signal.
- A floor (floor_ratio) prevents the cap from collapsing to near-zero
  after repeated skips, which could otherwise strand the selector with
  no viable candidates before the outer relax-cap loop even kicks in.

This is a guardrail, not a learning rule: it never touches match_fraction,
never touches the ideal waypoint curve, and resets to the base max_jump at
the start of every new session. Long-term personalization is still the
bandit's job; this only smooths reactions *within* one sitting.
"""

from dataclasses import dataclass, field

import pandas as pd

from .trajectory_planner import generate_trajectory, select_next_candidate


@dataclass
class InSessionGuardrail:
    base_max_jump: float
    tighten_factor: float = 0.6   # multiply cap by this after a skip
    recovery_factor: float = 0.5  # fraction of the way back toward 1.0 after a good/neutral signal
    floor_ratio: float = 0.3      # cap never shrinks below floor_ratio * base_max_jump
    theta: float = field(default=1.0, init=False)  # current tightness multiplier, 1.0 = base
    theta_history: list = field(default_factory=list, init=False)

    @property
    def effective_max_jump(self) -> float:
        return self.base_max_jump * self.theta

    def register_feedback(self, feedback: str):
        """feedback: 'skip' | 'replay' | 'neutral'"""
        if feedback == "skip":
            self.theta = max(self.floor_ratio, self.theta * self.tighten_factor)
        elif feedback == "replay":
            self.theta = min(1.0, self.theta + self.recovery_factor * (1.0 - self.theta))
        elif feedback == "neutral":
            self.theta = min(1.0, self.theta + 0.5 * self.recovery_factor * (1.0 - self.theta))
        else:
            raise ValueError(f"unknown feedback type: {feedback!r}")
        self.theta_history.append(self.theta)


class SessionController:
    """
    Stepwise version of plan_playlist(): generates the same ideal waypoint
    curve upfront, but selects and returns one song at a time, so feedback
    on song i can influence the selection of song i+1 via the guardrail.
    """

    def __init__(self, song_pool: pd.DataFrame, start, target, n_songs=10,
                 match_fraction=0.25, max_jump=0.15, guardrail: InSessionGuardrail = None):
        self.waypoints = generate_trajectory(start, target, n_songs, match_fraction)
        self.pool = song_pool.copy().reset_index(drop=True)
        self.used = set()
        self.prev_va = None
        self.guardrail = guardrail or InSessionGuardrail(base_max_jump=max_jump)
        self.step_idx = 0
        self.log = []  # per-step record: waypoint, cap requested/used, feedback

    def has_next(self) -> bool:
        return self.step_idx < len(self.waypoints)

    def next_song(self):
        """Select and return the next song (as a dict), using the guardrail's
        current effective_max_jump. Does not advance feedback state -- call
        submit_feedback() after the song is played to react before the
        following call to next_song()."""
        if not self.has_next():
            raise StopIteration("session already complete")

        wp = self.waypoints[self.step_idx]
        cap_requested = self.guardrail.effective_max_jump
        row, best_idx, cap_used = select_next_candidate(self.pool, self.used, self.prev_va, wp, cap_requested)

        result = {
            'song_id': row['song_id'], 'title': row['title'], 'artist': row['artist'],
            'valence': row['valence'], 'arousal': row['arousal'],
            'target_valence': round(wp[0], 3), 'target_arousal': round(wp[1], 3),
            'source': row.get('source', ''),
            'step_idx': self.step_idx,
            'cap_requested': round(cap_requested, 4),
            'cap_used': round(cap_used, 4),  # may exceed cap_requested if auto-relaxed for lack of candidates
        }

        self.used.add(best_idx)
        self.prev_va = (row['valence'], row['arousal'])
        self.log.append(result)
        self.step_idx += 1
        return result

    def submit_feedback(self, feedback: str):
        """Report skip/replay/neutral for the song just returned by next_song()."""
        self.guardrail.register_feedback(feedback)
        if self.log:
            self.log[-1]['feedback'] = feedback

    def guardrail_triggers(self) -> int:
        """How many songs so far were skipped, i.e. how many times the
        guardrail was told to tighten."""
        return sum(1 for e in self.log if e.get("feedback") == "skip")

    def guardrail_load(self) -> float:
        """
        How far, on average, the guardrail held the step cap below its base
        value while this session's songs were being chosen, scaled to [0, 1].

            0.0  the cap was never tightened
            1.0  the cap sat at its floor for every song

        This is the quantity the reward penalises. It is highly correlated with
        the skip count but time-aware: a skip only matters for the songs chosen
        after it, so a skip on the last song scores 0, an early skip scores
        more than a late one, and replays after a skip let the cap recover
        sooner. (Two skips back to back and two far apart score about the same.)
        """
        if not self.log:
            return 0.0
        g = self.guardrail
        span = 1.0 - g.floor_ratio
        if span <= 0 or g.base_max_jump <= 0:
            return 0.0
        tight = [max(0.0, 1.0 - e["cap_requested"] / g.base_max_jump) for e in self.log]
        return min(1.0, (sum(tight) / len(tight)) / span)

    def run_full(self, feedback_fn):
        """
        Convenience driver: repeatedly calls next_song(), asks feedback_fn(song)
        for a feedback label, and submits it -- for scripted/simulated demos.
        Returns the full log as a DataFrame.
        """
        while self.has_next():
            song = self.next_song()
            feedback = feedback_fn(song)
            self.submit_feedback(feedback)
        return pd.DataFrame(self.log)
