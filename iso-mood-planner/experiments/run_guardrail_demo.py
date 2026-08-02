"""
Demo: guardrail ON vs OFF under an identical scripted skip pattern.

Scenario: the listener skips songs at steps 3 and 4 (0-indexed), as if two
consecutive songs felt like too big/wrong a jump. We replay the *exact
same* feedback script against two controllers -- one with the guardrail
active (tighten_factor < 1), one with it disabled (tighten_factor = 1.0,
i.e. cap never actually tightens) -- so any difference in the resulting
max_jump trace and song path is attributable purely to the guardrail.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from iso_planner.session_controller import SessionController, InSessionGuardrail

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SKIP_STEPS = {3, 4}


def feedback_fn(song):
    return "skip" if song["step_idx"] in SKIP_STEPS else "neutral"


def run(pool, start, target, n_songs, match_fraction, max_jump, guardrail_active: bool):
    guardrail = InSessionGuardrail(
        base_max_jump=max_jump,
        tighten_factor=0.5 if guardrail_active else 1.0,
        recovery_factor=0.4,
        floor_ratio=0.3,
    )
    controller = SessionController(pool, start, target, n_songs=n_songs,
                                    match_fraction=match_fraction, max_jump=max_jump,
                                    guardrail=guardrail)
    log = controller.run_full(feedback_fn)
    return log


def main():
    pool = pd.read_csv("data/combined_song_pool.csv")
    start, target = (0.20, 0.30), (0.75, 0.35)
    n_songs, match_fraction, max_jump = 12, 0.25, 0.15

    log_on = run(pool, start, target, n_songs, match_fraction, max_jump, guardrail_active=True)
    log_off = run(pool, start, target, n_songs, match_fraction, max_jump, guardrail_active=False)

    log_on.to_csv(f"{OUT_DIR}/guardrail_on_log.csv", index=False)
    log_off.to_csv(f"{OUT_DIR}/guardrail_off_log.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(log_on["step_idx"], log_on["cap_requested"], "o-", color="tab:blue", label="guardrail ON")
    axes[0].plot(log_off["step_idx"], log_off["cap_requested"], "o--", color="tab:gray", label="guardrail OFF")
    for s in SKIP_STEPS:
        axes[0].axvline(s, color="red", alpha=0.25, lw=8)
    axes[0].set_ylabel("effective max_jump cap")
    axes[0].set_title("Requested max_jump cap per step (red bands = skipped song)")
    axes[0].legend()

    axes[1].plot(log_on["step_idx"], log_on["valence"], "o-", color="tab:blue", label="valence path (guardrail ON)")
    axes[1].plot(log_off["step_idx"], log_off["valence"], "o--", color="tab:gray", label="valence path (guardrail OFF)")
    axes[1].plot(log_on["step_idx"], log_on["target_valence"], "k:", alpha=0.5, label="ideal waypoint valence")
    for s in SKIP_STEPS:
        axes[1].axvline(s, color="red", alpha=0.25, lw=8)
    axes[1].set_xlabel("step (song index in session)")
    axes[1].set_ylabel("valence")
    axes[1].set_title("Resulting valence path")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/guardrail_demo.png", dpi=130)
    plt.close(fig)

    print("=== guardrail ON ===")
    print(log_on[["step_idx", "title", "valence", "cap_requested", "cap_used", "feedback"]].to_string(index=False))
    print("\n=== guardrail OFF ===")
    print(log_off[["step_idx", "title", "valence", "cap_requested", "cap_used", "feedback"]].to_string(index=False))


if __name__ == "__main__":
    main()
