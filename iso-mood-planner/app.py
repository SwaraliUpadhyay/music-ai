"""
Tiny Flask backend for the iso-mood-planner frontend.

It wires together the pieces that already exist in iso_planner/ so a live
session works end to end:

    start   -> LinearHierarchicalBanditManager picks a personalised arm
               (match_fraction, max_jump, n_songs); new users are warm-started
               from the shared contextual model
    next    -> SessionController picks one song at a time
    feedback-> InSessionGuardrail reacts to skip / replay / neutral at once
    finish  -> skip/replay rates + optional rating + optional "how do you feel
               now" mood become a SessionSignals reward, which updates both the
               user's bandit and the shared model

Run from the repo root:

    pip install -e . flask
    python webapp/app.py            # then open http://127.0.0.1:5050

State: bandit files are written to outputs/web_bandits/ (git-ignored).
Live sessions are kept in memory, so restarting the server drops any session
that is still in progress (bandit learning from finished sessions is kept).
"""

from __future__ import annotations

import math
import os
import re
import threading
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from iso_planner import (
    LinearHierarchicalBanditManager,
    SessionController,
    SessionSignals,
    compute_reward,
    mood_category_of,
)

HERE = Path(__file__).resolve().parent
# Works whether this file sits in iso-mood-planner/ or in iso-mood-planner/webapp/
ROOT = next((p for p in (HERE, HERE.parent) if (p / "data" / "combined_song_pool.csv").exists()), HERE.parent)
POOL_PATH = Path(os.environ.get("ISO_POOL", ROOT / "data" / "combined_song_pool.csv"))
STORE_DIR = Path(os.environ.get("ISO_STORE", ROOT / "outputs" / "web_bandits"))

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")

if not POOL_PATH.exists():
    raise SystemExit(f"Song pool not found at {POOL_PATH}. Run from the iso-mood-planner folder, "
                     "or set ISO_POOL to the path of combined_song_pool.csv.")
POOL = pd.read_csv(POOL_PATH)
MANAGER = LinearHierarchicalBanditManager(base_dir=str(STORE_DIR))

_lock = threading.Lock()
SESSIONS: dict[str, dict] = {}
MAX_LIVE_SESSIONS = 500

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")  # user_id becomes a file name -> keep it strict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _va(value, name):
    """Parse a [valence, arousal] pair; both must be numbers in [0, 1]."""
    try:
        v, a = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError(f"{name} must be [valence, arousal]")
    if not (math.isfinite(v) and math.isfinite(a) and 0 <= v <= 1 and 0 <= a <= 1):
        raise ValueError(f"{name} values must be between 0 and 1")
    return (v, a)


def _display(row: dict) -> dict:
    """Most of the pool (DEAM / Emotify) has no real title or artist, only ids
    like 'DEAM_song_1146' or 'rock_track_141'. Show those honestly as dataset
    clips instead of pretending they are named songs."""
    title, artist, source = str(row["title"]), row.get("artist"), str(row.get("source", ""))
    has_artist = isinstance(artist, str) and artist.strip() and artist.lower() != "nan"
    if has_artist:
        return {"display_title": title, "display_artist": artist, "placeholder": False}
    m = re.match(r"^DEAM_song_(\d+)$", title)
    if m:
        return {"display_title": f"Clip #{m.group(1)}", "display_artist": "DEAM dataset", "placeholder": True}
    m = re.match(r"^([a-z]+)_track_(\d+)$", title)
    if m:
        return {"display_title": f"{m.group(1).title()} track #{m.group(2)}",
                "display_artist": "Emotify dataset", "placeholder": True}
    return {"display_title": title, "display_artist": source or "dataset", "placeholder": True}


def _plain_song(song: dict) -> dict:
    out = {
        "song_id": str(song["song_id"]),
        "valence": float(song["valence"]),
        "arousal": float(song["arousal"]),
        "target_valence": float(song["target_valence"]),
        "target_arousal": float(song["target_arousal"]),
        "source": str(song.get("source", "")),
        "step_idx": int(song["step_idx"]),
        "cap_requested": float(song["cap_requested"]),
        "cap_used": float(song["cap_used"]),
    }
    out.update(_display(song))
    return out


def _arm_dict(arm) -> dict:
    return {"match_fraction": arm.match_fraction, "max_jump": arm.max_jump, "n_songs": arm.n_songs}


def _get_session(sid: str):
    s = SESSIONS.get(sid)
    if s is None:
        return None, _bad("unknown or expired session", 404)
    return s, None


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "songs": int(len(POOL)), "live_sessions": len(SESSIONS)})


@app.post("/api/session")
def start_session():
    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id", ""))
    if not USER_ID_RE.match(user_id):
        return _bad("user_id must be 1-40 letters, digits, '-' or '_'")
    try:
        start = _va(body.get("start"), "start")
        target = _va(body.get("target"), "target")
    except ValueError as e:
        return _bad(str(e))

    fixed_n = body.get("n_songs")
    if fixed_n is not None:
        try:
            fixed_n = int(fixed_n)
        except (TypeError, ValueError):
            return _bad("n_songs must be a whole number or null")
        if not 1 <= fixed_n <= 50:
            return _bad("n_songs must be between 1 and 50")

    hour = body.get("hour")
    if hour is not None:
        try:
            hour = int(hour)
        except (TypeError, ValueError):
            return _bad("hour must be 0-23 or null")
        if not 0 <= hour <= 23:
            return _bad("hour must be 0-23 or null")

    with _lock:
        bandit, ctx, is_new = MANAGER.get_bandit_for_session(user_id, start, target, hour_of_day=hour)
        arm = bandit.select_arm(fixed_n=fixed_n)
        controller = SessionController(
            POOL, start, target,
            n_songs=arm.n_songs, match_fraction=arm.match_fraction, max_jump=arm.max_jump,
        )
        if len(SESSIONS) >= MAX_LIVE_SESSIONS:  # drop the oldest unfinished session
            SESSIONS.pop(next(iter(SESSIONS)))
        sid = uuid.uuid4().hex
        SESSIONS[sid] = {
            "user_id": user_id, "bandit": bandit, "ctx": ctx, "arm": arm,
            "controller": controller, "start": start, "target": target,
            "feedback": [], "awaiting_feedback": False, "is_new_user": is_new,
        }

    return jsonify({
        "session_id": sid,
        "arm": _arm_dict(arm),
        "n_songs": arm.n_songs,
        "match_songs": max(1, int(arm.n_songs * arm.match_fraction)),
        "waypoints": [[float(v), float(a)] for v, a in controller.waypoints],
        "is_new_user": is_new,
        "start_category": mood_category_of(*start),
        "target_category": mood_category_of(*target),
    })


@app.post("/api/session/<sid>/next")
def next_song(sid):
    with _lock:
        s, err = _get_session(sid)
        if err:
            return err
        ctl = s["controller"]
        if s["awaiting_feedback"]:
            return _bad("send feedback for the current song first", 409)
        if not ctl.has_next():
            return jsonify({"done": True, "played": len(s["feedback"])})
        song = _plain_song(ctl.next_song())
        s["awaiting_feedback"] = True
        base = ctl.guardrail.base_max_jump
        return jsonify({
            "done": False,
            "song": song,
            "index": song["step_idx"],
            "total": len(ctl.waypoints),
            "in_match_phase": song["step_idx"] < max(1, int(len(ctl.waypoints) * s["arm"].match_fraction)),
            "gentle_mode": ctl.guardrail.effective_max_jump < base * 0.9,
        })


@app.post("/api/session/<sid>/feedback")
def feedback(sid):
    body = request.get_json(silent=True) or {}
    kind = body.get("feedback")
    if kind not in ("skip", "replay", "neutral"):
        return _bad("feedback must be skip, replay or neutral")
    with _lock:
        s, err = _get_session(sid)
        if err:
            return err
        if not s["awaiting_feedback"]:
            return _bad("no song is waiting for feedback", 409)
        ctl = s["controller"]
        ctl.submit_feedback(kind)
        s["feedback"].append(kind)
        s["awaiting_feedback"] = False
        return jsonify({
            "ok": True,
            "gentle_mode": ctl.guardrail.effective_max_jump < ctl.guardrail.base_max_jump * 0.9,
        })


@app.post("/api/session/<sid>/finish")
def finish(sid):
    body = request.get_json(silent=True) or {}

    rating = body.get("rating")
    if rating is not None:
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return _bad("rating must be 1-5 or null")
        if not 1 <= rating <= 5:
            return _bad("rating must be 1-5 or null")

    end_mood = body.get("end_mood")
    if end_mood is not None:
        try:
            end_mood = _va(end_mood, "end_mood")
        except ValueError as e:
            return _bad(str(e))

    with _lock:
        s, err = _get_session(sid)
        if err:
            return err
        played = len(s["feedback"])
        if played == 0:
            SESSIONS.pop(sid, None)
            return jsonify({"learned": False, "reason": "no songs were played"})

        skip_rate = s["feedback"].count("skip") / played
        replay_rate = s["feedback"].count("replay") / played

        # mood_progress: how much of the start->target distance the listener
        # closed, by their own end-of-session mood. None if they didn't say.
        progress = None
        if end_mood is not None:
            d0 = math.dist(s["start"], s["target"])
            d1 = math.dist(end_mood, s["target"])
            progress = 1.0 if d0 < 1e-6 else max(0.0, min(1.0, (d0 - d1) / d0))

        signals = SessionSignals(mood_progress=progress, skip_rate=skip_rate,
                                 replay_rate=replay_rate, rating=rating)
        reward = compute_reward(signals)
        MANAGER.close_session(s["bandit"], s["ctx"], s["arm"], reward)
        best = s["bandit"].best_arm_estimate()
        sessions_done = len(s["bandit"].history)
        SESSIONS.pop(sid, None)

    return jsonify({
        "learned": True,
        "reward": reward,
        "mood_progress": progress,
        "skip_rate": skip_rate,
        "replay_rate": replay_rate,
        "played": played,
        "sessions_done": sessions_done,
        "best_arm": _arm_dict(best),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # 5000 is taken by AirPlay on recent macOS
    app.run(host="127.0.0.1", port=port, debug=False)
