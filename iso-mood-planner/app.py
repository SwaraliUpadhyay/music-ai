"""
Tiny Flask backend for the iso-mood-planner frontend.

It wires together the pieces that already exist in iso_planner/ so a live
session works end to end:

    start   -> LinearHierarchicalBanditManager picks a personalised arm
               (match_fraction, max_jump, n_songs); new users are warm-started
               from the shared contextual model
    next    -> SessionController picks one song at a time
    feedback-> InSessionGuardrail reacts to skip / replay / neutral at once. The
               label comes from a button press or, when audio is playing, from
               how much of the clip was heard (iso_planner.implicit)
    finish  -> skip/replay rates + guardrail load + optional rating + optional
               "how do you feel now" mood become a SessionSignals reward, which
               updates both the user's bandit and the shared model. Every
               finished session is also appended to outputs/session_log.csv

Run from the repo root:

    pip install -e . flask
    python webapp/app.py            # then open http://127.0.0.1:5050

State: bandit files are written to outputs/web_bandits/ (git-ignored).
Live sessions are kept in memory, so restarting the server drops any session
that is still in progress (bandit learning from finished sessions is kept).
"""

from __future__ import annotations

import csv
import datetime
import difflib
import hashlib
import json
import math
import os
import re
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from iso_planner import (
    LinearHierarchicalBanditManager,
    SessionController,
    SessionSignals,
    classify_listening,
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
POOL_BY_ID = {str(r["song_id"]): r for r in reversed(POOL.to_dict("records"))}  # first row wins on duplicate ids
PREVIEW_CACHE_PATH = STORE_DIR.parent / "preview_cache.json"
SESSION_LOG_PATH = STORE_DIR.parent / "session_log.csv"
SESSION_LOG_FIELDS = [
    "time_utc", "user", "start_v", "start_a", "target_v", "target_a",
    "match_fraction", "max_jump", "n_songs", "played",
    "skip_rate", "replay_rate", "from_listening", "n_with_audio", "mean_listened",
    "guardrail_load", "triggers", "mood_progress", "rating", "reward",
]

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


# ---------------------------------------------------------------------------
# audio previews (30-second clips from public search APIs, nothing is stored)
# ---------------------------------------------------------------------------
# Only songs with a real title and artist can be looked up (the PMEmo rows).
# DEAM / Emotify rows are bare ids, so they simply have no audio and the page
# falls back to its buttons-only mode.

_cache_lock = threading.Lock()
try:
    with open(PREVIEW_CACHE_PATH, encoding="utf-8") as _f:
        _preview_cache: dict = json.load(_f)
except (OSError, ValueError):
    _preview_cache = {}


def _save_preview_cache():
    try:
        PREVIEW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREVIEW_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_preview_cache, f)
    except OSError:
        pass  # the cache is only an optimisation


def _norm(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)             # "(feat. X)", "[Remastered]"
    t = re.sub(r"\b(feat|ft|featuring)\b.*$", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _title_ok(candidate: str, wanted: str) -> bool:
    return difflib.SequenceMatcher(None, _norm(candidate), _norm(wanted)).ratio() >= 0.85


def _artist_ok(candidate: str, wanted: str) -> bool:
    nc, nw = _norm(candidate), _norm(wanted)
    if not nc or not nw:
        return False
    if nw in nc or nc in nw:
        return True
    if difflib.SequenceMatcher(None, nc, nw).ratio() >= 0.8:
        return True
    return any(p and _norm(p) in nc for p in re.split(r"[;,&/]|\band\b", str(wanted)))


def _http_json(url: str, timeout: float = 4.0):
    req = urllib.request.Request(url, headers={"User-Agent": "iso-mood-planner/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _search_itunes(title: str, artist: str):
    q = urllib.parse.urlencode({"term": f"{_norm(title)} {_norm(artist)}", "media": "music",
                                "entity": "song", "limit": 8})
    for r in _http_json("https://itunes.apple.com/search?" + q).get("results", []):
        if r.get("previewUrl"):
            yield r.get("trackName", ""), r.get("artistName", ""), r["previewUrl"]


def _search_deezer(title: str, artist: str):
    q = urllib.parse.urlencode({"q": f'artist:"{_norm(artist)}" track:"{_norm(title)}"', "limit": 8})
    for r in _http_json("https://api.deezer.com/search?" + q).get("data", []):
        if r.get("preview"):
            yield r.get("title", ""), (r.get("artist") or {}).get("name", ""), r["preview"]


def find_preview(song_id: str, title: str, artist: str):
    """Return {url, source, title, artist} or None. A confirmed 'no match' is cached;
    a network error is not, so it is retried next time."""
    with _cache_lock:
        if song_id in _preview_cache:
            return _preview_cache[song_id]
    found, had_error = None, False
    for source, search in (("iTunes", _search_itunes), ("Deezer", _search_deezer)):
        try:
            for t, a, url in search(title, artist):
                if _title_ok(t, title) and _artist_ok(a, artist):
                    found = {"url": url, "source": source, "title": t, "artist": a}
                    break
        except Exception:
            had_error = True
        if found:
            break
    if found or not had_error:
        with _cache_lock:
            _preview_cache[song_id] = found
            _save_preview_cache()
    return found


def _log_session(row: dict):
    """Append one finished session to outputs/session_log.csv. This is the data
    experiments/analyze_session_log.py uses to ask whether listening behaviour
    alone predicts mood progress. Logging must never break a session."""
    try:
        SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not SESSION_LOG_PATH.exists()
        with open(SESSION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SESSION_LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in SESSION_LOG_FIELDS})
    except OSError:
        pass


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


@app.get("/api/preview/<song_id>")
def preview(song_id):
    row = POOL_BY_ID.get(song_id)
    if row is None:
        return _bad("unknown song", 404)
    info = _display(row)
    if info["placeholder"]:
        return jsonify({"url": None, "reason": "dataset clip: no title or artist to search for"})
    hit = find_preview(song_id, str(row["title"]), str(row["artist"]))
    if not hit:
        return jsonify({"url": None, "reason": "no matching preview found"})
    return jsonify({"url": hit["url"], "source": hit["source"]})


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
            "feedback": [], "signals": [], "awaiting_feedback": False, "is_new_user": is_new,
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
    """Feedback on the song just played. Two ways to send it:

      {"feedback": "skip"|"replay"|"neutral"}        an explicit button press
      {"listened_fraction": 0.12, "replays": 0}      listening behaviour; the
                                                     label is derived from it

    An explicit press may also carry listened_fraction / replays so the log
    keeps the behaviour behind it. If both are given, the button wins."""
    body = request.get_json(silent=True) or {}

    frac = body.get("listened_fraction")
    if frac is not None:
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            return _bad("listened_fraction must be a number between 0 and 1, or null")
        if not (math.isfinite(frac) and 0 <= frac <= 1):
            return _bad("listened_fraction must be a number between 0 and 1, or null")
    try:
        replays = int(body.get("replays", 0))
    except (TypeError, ValueError):
        return _bad("replays must be a whole number")
    if not 0 <= replays <= 20:
        return _bad("replays must be between 0 and 20")

    kind = body.get("feedback")
    if kind is None:
        if frac is None:
            return _bad("send feedback (skip, replay or neutral) or listened_fraction")
        kind, source = classify_listening(frac, replays), "listening"
    elif kind in ("skip", "replay", "neutral"):
        source = "button"
    else:
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
        s["signals"].append({"feedback": kind, "source": source, "listened_fraction": frac})
        s["awaiting_feedback"] = False
        return jsonify({
            "ok": True,
            "feedback": kind,
            "source": source,
            "listened_fraction": frac,
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

        # How much of the playlist was played under a tightened step cap. This
        # lands on the arm that set the pace, so settings that keep tripping
        # the guardrail score lower for this listener.
        ctl = s["controller"]
        guardrail_load = ctl.guardrail_load()
        triggers = ctl.guardrail_triggers()

        heard = [x["listened_fraction"] for x in s["signals"] if x["listened_fraction"] is not None]
        from_listening = sum(1 for x in s["signals"] if x["source"] == "listening")
        mean_listened = (sum(heard) / len(heard)) if heard else None

        signals = SessionSignals(mood_progress=progress, skip_rate=skip_rate,
                                 replay_rate=replay_rate, rating=rating,
                                 guardrail_load=guardrail_load)
        reward = compute_reward(signals)
        MANAGER.close_session(s["bandit"], s["ctx"], s["arm"], reward)
        best = s["bandit"].best_arm_estimate()
        sessions_done = len(s["bandit"].history)

        _log_session({
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "user": hashlib.sha1(s["user_id"].encode()).hexdigest()[:10],
            "start_v": round(s["start"][0], 4), "start_a": round(s["start"][1], 4),
            "target_v": round(s["target"][0], 4), "target_a": round(s["target"][1], 4),
            "match_fraction": s["arm"].match_fraction, "max_jump": s["arm"].max_jump,
            "n_songs": s["arm"].n_songs, "played": played,
            "skip_rate": round(skip_rate, 4), "replay_rate": round(replay_rate, 4),
            "from_listening": from_listening, "n_with_audio": len(heard),
            "mean_listened": None if mean_listened is None else round(mean_listened, 4),
            "guardrail_load": round(guardrail_load, 4), "triggers": triggers,
            "mood_progress": None if progress is None else round(progress, 4),
            "rating": rating, "reward": round(reward, 4),
        })
        SESSIONS.pop(sid, None)

    return jsonify({
        "learned": True,
        "reward": reward,
        "mood_progress": progress,
        "skip_rate": skip_rate,
        "replay_rate": replay_rate,
        "guardrail_load": guardrail_load,
        "guardrail_triggers": triggers,
        "from_listening": from_listening,
        "played": played,
        "sessions_done": sessions_done,
        "best_arm": _arm_dict(best),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # 5000 is taken by AirPlay on recent macOS
    app.run(host="127.0.0.1", port=port, debug=False)
