"""
iso_planner
-----------
Core library for AI-driven, iso-principle music mood regulation:
- trajectory_planner: ideal valence-arousal curve + batch song selection
- session_controller:  stepwise selection + within-session reactive guardrail
- bandit:              per-user Thompson Sampling personalization
- hierarchical_bandit:  cross-user contextual sharing (cold-start acceleration)
- reward:              composite session reward function
"""

from .trajectory_planner import generate_trajectory, plan_playlist, select_next_candidate
from .session_controller import InSessionGuardrail, SessionController
from .bandit import Arm, UserBandit, BanditStore, build_arm_grid
from .hierarchical_bandit import GlobalContextPrior, HierarchicalBanditManager, mood_distance, bucketize_mood_distance
from .contextual_bandit import (
    LinearArmModel, LinearGlobalContextModel, LinearHierarchicalBanditManager,
    build_context_features, FEATURE_NAMES, MOOD_CATEGORIES, TIME_BUCKETS,
    mood_category_of, time_bucket_of,
)
from .reward import SessionSignals, compute_reward
from .implicit import classify_listening, SKIP_BELOW
from .data_utils import stratified_train_test_split, stratum_coverage_report

__all__ = [
    "generate_trajectory", "plan_playlist", "select_next_candidate",
    "InSessionGuardrail", "SessionController",
    "Arm", "UserBandit", "BanditStore", "build_arm_grid",
    "GlobalContextPrior", "HierarchicalBanditManager", "mood_distance", "bucketize_mood_distance",
    "LinearArmModel", "LinearGlobalContextModel", "LinearHierarchicalBanditManager",
    "build_context_features", "FEATURE_NAMES", "MOOD_CATEGORIES", "TIME_BUCKETS",
    "mood_category_of", "time_bucket_of",
    "SessionSignals", "compute_reward",
    "classify_listening", "SKIP_BELOW",
    "stratified_train_test_split", "stratum_coverage_report",
]
