# iso-mood-planner

AI-driven music-based mood regulation using the music-therapy **iso principle**:
match a listener's current mood first, then gradually shift them toward a
target mood — never jump straight to "happy" music. Built as a course
project targeting a regional/mid-tier IEEE conference submission (e.g.
ICRITO/ACCAI-tier), with personalization added on top of the base planner.

## What's in here

1. **Trajectory planner** — given a start mood and target mood (in
   valence-arousal space), generates an eased two-phase curve (a short
   "matching phase" near the start mood, then a gradual transition) and
   selects real songs from a pooled dataset that follow it, subject to a
   hard cap on how big a mood-jump between consecutive songs is allowed.
2. **Per-user personalization** — a Thompson Sampling bandit learns each
   user's preferred `(match_fraction, max_jump, session_length)` from
   session feedback (skip/replay/rating/mood self-report), instead of
   using one fixed setting for everyone.
3. **Within-session guardrail** — a deterministic rule that reacts
   immediately if a song is skipped mid-playlist (tightening the next
   jump), rather than waiting for the next session's bandit update.
4. **Cross-user contextual sharing** — new users are warm-started from a
   population-level model (a linear contextual bandit over mood-distance,
   mood-category, and time-of-day) instead of starting from scratch every
   time, cutting cold-start regret roughly in half — and doing so more
   robustly than naive discrete bucketing once multiple context features
   are involved.

## Repo structure

```
iso-mood-planner/
├── iso_planner/                  # the installable core library
│   ├── __init__.py
│   ├── trajectory_planner.py     # ideal V-A curve + batch song selection
│   ├── session_controller.py     # stepwise selection + in-session guardrail
│   ├── bandit.py                 # per-user Thompson Sampling bandit
│   ├── hierarchical_bandit.py    # cross-user contextual sharing (discrete buckets)
│   ├── contextual_bandit.py      # cross-user sharing via a linear contextual model
│   ├── reward.py                 # composite session reward function
│   └── data_utils.py             # stratified train/test split for the song pool
├── experiments/                  # simulation scripts (not installed, run as scripts)
│   ├── persona_simulator.py      # synthetic users with hidden true preferences
│   ├── run_experiment.py         # per-user bandit convergence test
│   ├── run_guardrail_demo.py     # guardrail ON vs OFF demo
│   ├── run_cold_start_experiment.py  # cross-user sharing / cold-start test
│   ├── run_richer_context_experiment.py  # linear model vs. extended buckets ablation
│   ├── make_dataset_split.py     # generates data/song_pool_{dev,test}.csv
│   └── evaluate_on_test_pool.py  # final metrics, held-out test pool only
├── examples/
│   └── integration_demo.py       # minimal real end-to-end usage example
├── tests/
│   └── test_regression.py        # sanity checks for core behavior
├── data/
│   ├── combined_song_pool.csv    # merged song pool (see Data section)
│   ├── song_pool_dev.csv         # generated: 80%, stratified by source+VA-quadrant
│   └── song_pool_test.csv        # generated: 20%, held out for final evaluation
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Setup

```bash
git clone <your-repo-url>
cd iso-mood-planner
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Installing with `-e .` makes `iso_planner` importable from anywhere in the
repo (`from iso_planner.bandit import UserBandit`, etc.) without needing to
fuss with `sys.path`.

## Usage

Run everything from the repo root.

```bash
# Minimal real usage example: bandit picks parameters -> real playlist generated
python3 -m examples.integration_demo

# Does the per-user bandit actually converge to sensible personal settings?
python3 -m experiments.run_experiment

# Does the in-session guardrail actually react to a skip?
python3 -m experiments.run_guardrail_demo

# Does warm-starting new users from a shared context prior reduce cold-start regret?
python3 -m experiments.run_cold_start_experiment

# Richer context (mood category + time of day): linear model vs. naive bucket extension
python3 -m experiments.run_richer_context_experiment

# Sanity tests
python3 tests/test_regression.py

# Generate the stratified dev/test song-pool split (run once)
python3 -m experiments.make_dataset_split

# Final quantitative metrics -- held-out test pool only
python3 -m experiments.evaluate_on_test_pool
```

Experiment scripts write plots/CSVs to a local `outputs/` folder (git-ignored).

## Results so far

**Per-user convergence** (60 simulated sessions/persona): 3 of 4 synthetic
personas converge exactly to their true `(match_fraction, max_jump, n_songs)`;
a deliberately noisy/inconsistent persona converges more slowly and
imperfectly (final quality 0.71 vs 1.00) — an honest, reportable limitation
of per-user bandits under noisy feedback.

**Guardrail**: after two consecutive skips, the effective jump cap tightens
from 0.15 → 0.075 → 0.045, then recovers gradually as neutral feedback
resumes — confirmed against an identical guardrail-off control run.

**Cross-user sharing**: mean regret over a new user's first 15 sessions
drops from 0.385 (no sharing) to 0.248 (with sharing) even early on, and to
0.145 by the 40th arriving user, while the no-sharing baseline stays flat
(~0.36-0.39) — the signature of a working cold-start mechanism.

**Richer context (linear model vs. naive bucket extension)**: with 60
simulated new users whose true preferences genuinely depend on
mood_distance AND mood_category AND time_of_day, naively extending
discrete bucketing to 3 dimensions (60 cells) barely beats no-sharing
early on (0.312 vs 0.320 baseline) — fragmentation undermines the benefit.
A linear contextual model (Bayesian ridge regression per arm, sampled via
Thompson Sampling — the LinUCB-style approach) shares coefficients across
all contexts instead of counting per-cell, and shows a clear advantage
from the first arrivals (0.279) that keeps improving (0.208 by arrival 60).

## Data

`data/combined_song_pool.csv` merges three public music-emotion datasets,
normalized to a consistent 0-1 valence/arousal scale:
- **DEAM** (Database for Emotional Analysis of Music) — Aljanaki, Yang &
  Soleymani, "Developing a benchmark for emotional analysis of music,"
  PLOS ONE, 2017.
- **PMEmo** — Zhang, Zhang, Li, Yang & Sun, "The PMEmo Dataset for Music
  Emotion Recognition," ICMR, 2018.
- **Emotify** — Aljanaki, Wiering & Veltkamp, "Studying emotion induced by
  music through a crowdsourcing game," Information Processing &
  Management, 2016.

**Important caveat for the paper**: Emotify's valence-arousal values are
*derived*, not directly measured — converted from its categorical GEMS-9
mood ratings via a category-to-dimension mapping. State this explicitly in
any methods section; don't present it as equivalent to DEAM/PMEmo's
directly-annotated values. (Double-check the exact citations above against
the original papers before submission — reproduced here from memory.)

## Dev/test split — what it's for and what it isn't

`plan_playlist` doesn't fit any parameters to the song pool (it's
nearest-neighbor selection against existing valence/arousal labels), so
there's no classic overfitting risk in the sense of memorizing training
songs. But the *defaults* (`max_jump=0.15`, `match_fraction=0.25`, the
bandit's arm-grid bounds) were chosen by observing behavior on the full
pool — so reporting final quantitative results on that same full pool
would be mild leakage (tuned and evaluated on the same data).

`experiments/make_dataset_split.py` produces a stratified 80/20 split
(stratified by `source` × valence-arousal quadrant, so both splits have
proportionally similar mood-space coverage — a naive random split could
by chance strip a whole mood region from the test set and make the
algorithm look artificially worse). Use `song_pool_dev.csv` for all
exploratory work; run `experiments/evaluate_on_test_pool.py` for the
numbers that actually belong in the paper.

**This does NOT apply to the bandit/persona experiments** — convergence
and cold-start results operate purely in `(match_fraction, max_jump,
n_songs)` arm-space against simulated feedback and never touch the song
pool, so there's no leakage risk there regardless of how the pool is
split.

**Where a split will matter far more**: if this project later trains a
model (e.g. predicting valence/arousal from audio features to expand
beyond these 3 datasets' hand-labeled songs), that model needs its own
rigorous train/val/test split — non-negotiable, unlike the planner's
current lookup-based approach.

## Known limitations / honest scope

- The 27-arm discrete grid is a deliberate simplification of what is
  really a continuous parameter space.
- Per-user bandits and the within-session guardrail are currently
  independent — a user who trips the guardrail often isn't yet reflected
  in their bandit's reward signal.
- All personalization results so far are from **simulated personas**, not
  real users — a small user study is the natural next step before the
  convergence/cold-start numbers can be claimed as validated.
- Context sharing now supports mood_distance, mood_category, and
  time_of_day via the linear model (`contextual_bandit.py`); the discrete
  single-feature bucketing (`hierarchical_bandit.py`) is kept for
  comparison/ablation purposes, not as the recommended path forward.
  Further context features (e.g. day-of-week, listening-history signals)
  remain unexplored.
- This is positioned as a course-project-to-conference-paper pipeline; a
  patent is explicitly out of scope / future-work-only.

## Roadmap

1. Ablation: does skip/replay alone predict mood_progress well enough to
   drop reliance on self-report surveys?
2. Small real user study to replace/validate the persona simulations.
3. Feed guardrail-activation frequency into the bandit's reward as an
   auxiliary signal.
4. Continuous (Gaussian Process) action space once real data justifies it
   — natural next step after the linear contextual model, since both
   move away from discrete grids/buckets toward continuous representations.
5. Extend the linear contextual model's feature set (day-of-week, recent
   listening history) now that the infrastructure supports it cheaply.
