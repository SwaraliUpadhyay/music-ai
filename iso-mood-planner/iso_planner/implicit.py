"""
Implicit feedback from listening behaviour.

The guardrail and the bandit both consume three labels per song:
'skip' | 'replay' | 'neutral'. Until now those came from explicit buttons.
When audio is actually playing, the listener's behaviour already says most
of it, so the label can be derived from how much of the song they heard:

    heard less than `skip_below` of the clip   -> 'skip'    (they bailed early)
    played it again after it finished          -> 'replay'  (they wanted more)
    anything else (heard most of it)           -> 'neutral' (let it play through)

Explicit buttons still win: a "Loved it" press is a replay and a "Skip" press
is a skip, whatever the playback fraction was. This module only covers the
implicit case, so a session can run with no button presses at all.

The 0.30 threshold is a starting guess, not a tuned value. For a 30-second
preview it means "left within the first nine seconds". Real listening data
(see experiments/analyze_session_log.py) is what should adjust it.
"""

import math

SKIP_BELOW = 0.30


def classify_listening(listened_fraction: float, replays: int = 0,
                       skip_below: float = SKIP_BELOW) -> str:
    """Map playback behaviour on one song to 'skip' | 'replay' | 'neutral'.

    listened_fraction: furthest point reached as a share of the clip, in [0, 1].
    replays: how many times the listener restarted the clip after it had ended.
    """
    if not isinstance(replays, int) or replays < 0:
        raise ValueError("replays must be a non-negative whole number")
    if listened_fraction is None or not math.isfinite(listened_fraction):
        raise ValueError("listened_fraction must be a finite number")
    f = min(1.0, max(0.0, float(listened_fraction)))
    if replays > 0:
        return "replay"
    if f < skip_below:
        return "skip"
    return "neutral"
