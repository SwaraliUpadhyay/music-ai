"""
Web API tests with Flask's test client (no browser, no network).
Run: python tests/test_webapp.py   (from iso-mood-planner, with PYTHONPATH set to it)
app.py may live in iso-mood-planner/ or iso-mood-planner/webapp/.
"""
import csv
import importlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "webapp"):
    sys.path.insert(0, str(p))
os.environ["ISO_STORE"] = tempfile.mkdtemp() + "/web_bandits"
app_mod = importlib.import_module("app")
client = app_mod.app.test_client()


def _start():
    r = client.post("/api/session", json={"user_id": "tester", "start": [0.2, 0.7],
                                          "target": [0.7, 0.4]})
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def test_listening_is_turned_into_labels_and_logged():
    s = _start()
    sid = s["session_id"]
    labels = []
    plan = [{"listened_fraction": 0.1}, {"listened_fraction": 1.0},
            {"listened_fraction": 0.2, "replays": 1}, {"feedback": "skip", "listened_fraction": 1.0}]
    for body in plan:
        assert client.post(f"/api/session/{sid}/next").status_code == 200
        r = client.post(f"/api/session/{sid}/feedback", json=body)
        assert r.status_code == 200, r.get_data(as_text=True)
        labels.append(r.get_json()["feedback"])
    assert labels == ["skip", "neutral", "replay", "skip"]   # the button wins when both are sent
    fin = client.post(f"/api/session/{sid}/finish", json={}).get_json()
    assert fin["from_listening"] == 3
    assert fin["guardrail_triggers"] == 2
    assert 0.0 <= fin["guardrail_load"] <= 1.0
    log = app_mod.SESSION_LOG_PATH
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 1 and rows[0]["user"] != "tester"   # user id is hashed
    assert rows[0]["triggers"] == "2"


def test_bad_feedback_is_rejected():
    sid = _start()["session_id"]
    client.post(f"/api/session/{sid}/next")
    for body in ({}, {"feedback": "nope"}, {"listened_fraction": "x"}, {"listened_fraction": float("nan")}):
        r = client.post(f"/api/session/{sid}/feedback", json=body)
        assert r.status_code == 400, (body, r.status_code)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("OK:", t.__name__)
    print(f"\n{len(tests)} tests passed.")
