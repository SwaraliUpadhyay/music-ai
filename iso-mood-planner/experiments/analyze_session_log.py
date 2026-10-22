"""
Does listening behaviour stand in for the end-of-session survey?

Reads outputs/session_log.csv (written by the web app, one row per finished
session) and, for sessions where the listener also gave an end mood, compares

    behaviour-only reward   skip_rate, replay_rate, guardrail_load  (no survey)
    survey signal           mood_progress

If the two move together, the survey can be dropped and the bandit can learn
from behaviour alone. Prints Pearson and Spearman correlations with a
bootstrap 95% interval. With fewer than MIN_SESSIONS paired sessions it says so
instead of reporting a number nobody should trust.

    python -m experiments.analyze_session_log [path/to/session_log.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from iso_planner import SessionSignals, compute_reward

MIN_SESSIONS = 20
N_BOOT = 2000


def behaviour_reward(row) -> float:
    return compute_reward(SessionSignals(
        skip_rate=row["skip_rate"], replay_rate=row["replay_rate"],
        guardrail_load=row["guardrail_load"]))


def _corr(x, y, method):
    return pd.Series(x).corr(pd.Series(y), method=method)


def analyse(df: pd.DataFrame, seed=0) -> dict:
    paired = df.dropna(subset=["mood_progress", "skip_rate", "replay_rate", "guardrail_load"]).copy()
    out = {"n_total": len(df), "n_paired": len(paired)}
    if len(paired) < MIN_SESSIONS:
        out["verdict"] = f"too few sessions ({len(paired)} with an end mood, need {MIN_SESSIONS})"
        return out
    x = paired.apply(behaviour_reward, axis=1).to_numpy()
    y = paired["mood_progress"].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        out["verdict"] = "one of the signals never varies, correlation is undefined"
        return out
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) and np.std(y[i]):
            boots.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.update(pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(_corr(x, y, "spearman")),
               ci_low=float(lo), ci_high=float(hi))
    out["verdict"] = ("behaviour tracks the survey (CI above 0)" if lo > 0
                      else "no clear link between behaviour and the survey yet (CI includes 0)")
    return out


def main(path=None):
    here = Path(__file__).resolve().parent.parent
    path = Path(path) if path else here / "outputs" / "session_log.csv"
    if not path.exists():
        print(f"No session log at {path}. Run the web app and finish some sessions first.")
        return None
    res = analyse(pd.read_csv(path))
    print(f"sessions logged: {res['n_total']}, with an end mood: {res['n_paired']}")
    if "pearson" in res:
        print(f"Pearson {res['pearson']:+.2f}  Spearman {res['spearman']:+.2f}  "
              f"95% CI [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]")
    print(res["verdict"])
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
