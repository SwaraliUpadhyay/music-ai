"""
Convergence experiment: for each synthetic persona, run the per-user
Thompson Sampling bandit for T simulated sessions and check whether it
converges toward that persona's true (match_fraction, max_jump, n_songs).

Outputs (written to ./outputs/):
- convergence_<persona>.png   : reward curve + chosen-parameter traces vs true values
- regret_all_personas.png     : cumulative regret comparison across personas
- convergence_summary.csv     : final learned params vs true params, per persona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from iso_planner.bandit import UserBandit, build_arm_grid
from experiments.persona_simulator import default_personas

T_SESSIONS = 60
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def oracle_quality(persona, arms):
    """Best achievable quality on the grid, for regret computation."""
    return max(persona.quality(a) for a in arms)


def run_persona(persona, arms, T=T_SESSIONS, seed=0):
    bandit = UserBandit(user_id=persona.name.replace(" ", "_"), arms=arms)
    rng = np.random.default_rng(seed)

    rewards, qualities, chosen_mf, chosen_mj, chosen_n = [], [], [], [], []
    best_q = oracle_quality(persona, arms)
    regret = []

    for t in range(T):
        arm = bandit.select_arm(rng=rng)
        reward, signals = persona.simulate_session(arm)
        bandit.update(arm, reward)

        q = persona.quality(arm)
        rewards.append(reward)
        qualities.append(q)
        chosen_mf.append(arm.match_fraction)
        chosen_mj.append(arm.max_jump)
        chosen_n.append(arm.n_songs)
        regret.append(best_q - q)

    return {
        "bandit": bandit,
        "rewards": rewards,
        "qualities": qualities,
        "chosen_mf": chosen_mf,
        "chosen_mj": chosen_mj,
        "chosen_n": chosen_n,
        "cum_regret": np.cumsum(regret),
    }


def rolling_mean(x, window=5):
    x = np.array(x, dtype=float)
    return pd.Series(x).rolling(window, min_periods=1).mean().values


def plot_persona(persona, result, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sessions = np.arange(1, len(result["rewards"]) + 1)

    axes[0, 0].plot(sessions, result["rewards"], alpha=0.35, color="tab:blue", label="reward")
    axes[0, 0].plot(sessions, rolling_mean(result["rewards"]), color="tab:blue", lw=2, label="rolling mean (5)")
    axes[0, 0].set_title("Session reward over time")
    axes[0, 0].set_xlabel("session")
    axes[0, 0].set_ylabel("reward")
    axes[0, 0].legend()

    axes[0, 1].plot(sessions, result["chosen_mf"], "o", ms=3, alpha=0.5, color="tab:orange", label="chosen match_fraction")
    axes[0, 1].axhline(persona.true_match_fraction, color="k", ls="--", label="true match_fraction")
    axes[0, 1].set_title("match_fraction over time")
    axes[0, 1].set_xlabel("session")
    axes[0, 1].legend()

    axes[1, 0].plot(sessions, result["chosen_mj"], "o", ms=3, alpha=0.5, color="tab:green", label="chosen max_jump")
    axes[1, 0].axhline(persona.true_max_jump, color="k", ls="--", label="true max_jump")
    axes[1, 0].set_title("max_jump over time")
    axes[1, 0].set_xlabel("session")
    axes[1, 0].legend()

    axes[1, 1].plot(sessions, result["chosen_n"], "o", ms=3, alpha=0.5, color="tab:red", label="chosen n_songs")
    axes[1, 1].axhline(persona.true_n_songs, color="k", ls="--", label="true n_songs")
    axes[1, 1].set_title("n_songs over time")
    axes[1, 1].set_xlabel("session")
    axes[1, 1].legend()

    fig.suptitle(f"Bandit convergence — persona: {persona.name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_regret(all_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, result in all_results.items():
        ax.plot(result["cum_regret"], label=name)
    ax.set_xlabel("session")
    ax.set_ylabel("cumulative regret (quality units)")
    ax.set_title("Cumulative regret vs. oracle best arm, by persona")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    arms = build_arm_grid()
    personas = default_personas()

    all_results = {}
    summary_rows = []

    for persona in personas:
        result = run_persona(persona, arms)
        all_results[persona.name] = result

        safe_name = persona.name.split(" ")[0].lower().replace("/", "_")
        plot_persona(persona, result, f"{OUT_DIR}/convergence_{safe_name}.png")

        best = result["bandit"].best_arm_estimate()
        summary_rows.append({
            "persona": persona.name,
            "true_match_fraction": persona.true_match_fraction,
            "learned_match_fraction": best.match_fraction,
            "true_max_jump": persona.true_max_jump,
            "learned_max_jump": best.max_jump,
            "true_n_songs": persona.true_n_songs,
            "learned_n_songs": best.n_songs,
            "final_quality": persona.quality(best),
            "mean_reward_last_10": float(np.mean(result["rewards"][-10:])),
        })

    plot_regret(all_results, f"{OUT_DIR}/regret_all_personas.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{OUT_DIR}/convergence_summary.csv", index=False)
    pd.set_option("display.width", 140)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
