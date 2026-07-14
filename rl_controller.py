#!/usr/bin/env python3
"""
Online REINFORCE Policy Controller — Adaptive Green AI
Implements the Decision-based-on-RL layer described in the workflow diagram.

Algorithm: Online REINFORCE with EMA Baseline and Dirichlet Exploration

Design principles (no offline training, no UI control):
  - Weights are adapted automatically after every real inference outcome.
  - No batch jobs, no user sliders, no scheduled retraining.
  - State survives restarts via `data/rl_state.json`.
  - Each tenant tier learns its own independent policy.

Mathematical formulation
─────────────────────────
Policy π_θ selects a routing candidate by CSS score using weights θ = (w_c, w_l, w_a, w_cost).

Selection likelihood (softmax over CSS scores):
    log π_θ(a|s) = CSS_score(a) / T  −  log Σ exp(CSS_score(a') / T)

REINFORCE gradient (policy gradient theorem):
    ∇_θ J(θ) = (R − b) · ∇_θ log π_θ(a|s)

where:
    R   = observed multi-objective reward (0–1)
    b   = running EMA baseline (variance reduction)
    ∇_θ log π_θ = score_i(selected) − Σ_a' π(a'|s) · score_i(a')
               = component_score_i(selected) − mean_component_i(all candidates)

Update rule:
    w_i ← w_i + α_t · (R − b) · grad_i
    α_t = α_0 / (1 + √t)          (decaying learning rate)
    Project onto simplex with floor w_i ≥ w_min

Reward function:
    R = λ_sla · r_sla + λ_c · r_carbon + λ_acc · r_accuracy + λ_cost · r_cost
    r_sla  = clamp(1 − max(0, Δt / sla_ms), 0, 1)   (SLA conformance)
    r_c    = 1 − min(actual_carbon / carbon_ref, 1)   (lower carbon better)
    r_acc  = {1.0: clean, 0.65: quality-retry, 0.30: fallback, 0.10: timeout}
    r_cost = 1 − min(actual_cost / cost_ref, 1)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RL_STATE_PATH = Path(os.getenv("RL_STATE_PATH", DATA_DIR / "rl_state.json"))
POLICY_CONFIG_PATH = Path(os.getenv("POLICY_CONFIG_PATH", BASE_DIR / "config" / "policies.json"))

# Initial learning rate (decays as α₀ / (1 + √t))
ALPHA_0 = float(os.getenv("RL_ALPHA_0", "0.06"))
# EMA decay for baseline estimate (β ≈ 0.95 → smoothes over ~20 episodes)
BASELINE_EMA_BETA = float(os.getenv("RL_BASELINE_BETA", "0.95"))
# Softmax temperature for action probability (higher → more uniform exploration)
SOFTMAX_TEMP = float(os.getenv("RL_SOFTMAX_TEMP", "0.25"))
# Dirichlet noise concentration for exploration (α < 1 → sparse perturbation)
DIRICHLET_ALPHA = float(os.getenv("RL_DIRICHLET_ALPHA", "0.3"))
# Fraction of Dirichlet noise mixed into weights during exploration
DIRICHLET_EPSILON = float(os.getenv("RL_DIRICHLET_EPSILON", "0.15"))
# Minimum weight per coefficient (prevents degenerate all-zero policies)
W_MIN = float(os.getenv("RL_W_MIN", "0.05"))
# Meta-weights for reward components
REWARD_LAMBDA_SLA = float(os.getenv("RL_REWARD_LAMBDA_SLA", "0.35"))
REWARD_LAMBDA_CARBON = float(os.getenv("RL_REWARD_LAMBDA_CARBON", "0.30"))
REWARD_LAMBDA_ACCURACY = float(os.getenv("RL_REWARD_LAMBDA_ACCURACY", "0.25"))
REWARD_LAMBDA_COST = float(os.getenv("RL_REWARD_LAMBDA_COST", "0.10"))
# Carbon reference for normalisation (gCO₂/request at full model)
CARBON_REFERENCE_G = float(os.getenv("RL_CARBON_REF_G", "0.05"))
# Cost reference (max cost_units)
COST_REFERENCE = float(os.getenv("RL_COST_REF", "1.0"))
# Policy version auto-increments when reward plateaus for this many episodes
CONVERGENCE_WINDOW = int(os.getenv("RL_CONVERGENCE_WINDOW", "50"))
CONVERGENCE_TOLERANCE = float(os.getenv("RL_CONVERGENCE_TOL", "0.005"))
# Max reward history kept per tier
MAX_REWARD_HISTORY = int(os.getenv("RL_MAX_HISTORY", "200"))
# Exploration flag can be toggled at runtime (not via UI)
EXPLORATION_ENABLED = os.getenv("RL_EXPLORATION", "true").lower() in {"1", "true", "yes"}

# ── Per-zone policy fine-tuning (Option 1 extension) ──────────────────────────
# A grid zone gets its own specialised state once it has accumulated at least
# RL_ZONE_MIN_EPISODES outcomes. Until then, the per-tier global policy is used.
# This keeps weights bounded (same simplex projection) and means a low-traffic
# zone never drifts off a small sample.
ZONE_LEARNING_ENABLED = os.getenv("RL_ZONE_LEARNING", "true").lower() in {"1", "true", "yes"}
ZONE_MIN_EPISODES = int(os.getenv("RL_ZONE_MIN_EPISODES", "20"))
# Reward weight for the grid-quality signal: rewards routing to a low-carbon
# candidate when the grid is dirty. Bounded; included in the simplex sum already
# implicitly because all λ's are normalised below.
REWARD_LAMBDA_GRID_QUALITY = float(os.getenv("RL_REWARD_LAMBDA_GRID_QUALITY", "0.0"))
# Grid intensity (gCO2/kWh) above which the grid-quality term applies fully.
GRID_DIRTY_THRESHOLD = float(os.getenv("RL_GRID_DIRTY_THRESHOLD", "500.0"))

# ── Default initial weights (from policies.json; overridden by rl_state.json) ─
# Carbon-dominant cold start: Adaptive Green AI's core promise is that the final
# routing decision is the greenest feasible candidate. RL is free to learn away
# from these over time when reward signals justify it, but the starting point
# must already reflect the sustainability objective. Four-dim simplex (no region
# axis here; region weight stays in policies.json and is applied on top of CSS).
_DEFAULT_WEIGHTS_BY_TIER = {
    "standard": {"carbon": 0.58, "latency": 0.15, "accuracy": 0.19, "cost": 0.08},
    "premium":  {"carbon": 0.47, "latency": 0.21, "accuracy": 0.23, "cost": 0.09},
    "esg":      {"carbon": 0.73, "latency": 0.08, "accuracy": 0.13, "cost": 0.06},
    "batch":    {"carbon": 0.68, "latency": 0.06, "accuracy": 0.14, "cost": 0.12},
}
_COEFFICIENT_KEYS = ("carbon", "latency", "accuracy", "cost")
_TIERS = list(_DEFAULT_WEIGHTS_BY_TIER.keys())


# ── RL state per tier ─────────────────────────────────────────────────────────

class _TierState:
    """Runtime RL state for one tenant tier."""

    __slots__ = (
        "weights", "episode_count", "baseline_ema",
        "reward_history", "policy_version", "last_updated",
    )

    def __init__(self, weights: dict[str, float]):
        self.weights: dict[str, float] = dict(weights)
        self.episode_count: int = 0
        self.baseline_ema: float = 0.5       # initialise at midpoint
        self.reward_history: deque[float] = deque(maxlen=MAX_REWARD_HISTORY)
        self.policy_version: int = 1
        self.last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "episode_count": self.episode_count,
            "baseline_ema": round(self.baseline_ema, 6),
            "reward_history": list(self.reward_history),
            "policy_version": self.policy_version,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_TierState":
        obj = cls(d.get("weights", {}))
        obj.episode_count = int(d.get("episode_count", 0))
        obj.baseline_ema = float(d.get("baseline_ema", 0.5))
        for r in d.get("reward_history", []):
            obj.reward_history.append(float(r))
        obj.policy_version = int(d.get("policy_version", 1))
        obj.last_updated = float(d.get("last_updated", 0.0))
        return obj


# ── Main controller ───────────────────────────────────────────────────────────

class RLPolicyController:
    """
    Online REINFORCE policy controller.

    Adapts (w_carbon, w_latency, w_accuracy, w_cost) per tenant tier using
    actual inference outcomes. No offline training, no user controls.
    """

    def __init__(
        self,
        state_path: str | Path = RL_STATE_PATH,
        policy_config_path: str | Path = POLICY_CONFIG_PATH,
    ):
        self._state_path = Path(state_path)
        self._policy_config_path = Path(policy_config_path)
        self._lock = threading.RLock()
        self._tier_states: dict[str, _TierState] = {}
        # Zone-specialised states keyed by "tier::zone". A zone state is
        # initialised lazily from its parent tier the first time we see traffic
        # for that zone, and is only used for action selection once it has
        # accumulated at least ZONE_MIN_EPISODES outcomes.
        self._zone_states: dict[str, _TierState] = {}
        self._dirty = False          # pending save flag
        self._save_thread: threading.Thread | None = None
        self._running = True

        self._load()
        self._start_background_saver()

    @staticmethod
    def _zone_key(tier: str, zone: str | None) -> str | None:
        if not zone:
            return None
        z = str(zone).strip().lower()
        if not z or z == "primary":
            return None
        return f"{tier}::{z}"

    def _get_or_init_zone_state(self, tier: str, zone_key: str) -> _TierState:
        """Lazily seed a zone state from its parent tier."""
        state = self._zone_states.get(zone_key)
        if state is not None:
            return state
        parent = self._tier_states[tier]
        state = _TierState(dict(parent.weights))
        state.baseline_ema = parent.baseline_ema
        self._zone_states[zone_key] = state
        return state

    # ── Public API ────────────────────────────────────────────────────────────

    def get_policy(
        self,
        user_tier: str,
        context: dict[str, Any] | None = None,
        explore: bool = EXPLORATION_ENABLED,
        zone: str | None = None,
    ) -> dict[str, Any]:
        """
        Return current policy coefficients for this tier (with optional exploration noise).

        `context` is informational only at this stage; included for future
        contextual-bandit extension (feature vector per request).

        When ``zone`` is provided and the per-zone state has accumulated at
        least :data:`ZONE_MIN_EPISODES` outcomes, the zone-specialised weights
        are returned instead. This lets grids with different carbon profiles
        learn distinct trade-offs without harming low-traffic zones.

        Returns a dict with keys: carbon, latency, accuracy, cost, tier, version,
        exploration_applied, source ("tier" or "zone"), rl_controlled=True.
        """
        tier = _normalise_tier(user_tier)
        zone_key = self._zone_key(tier, zone) if ZONE_LEARNING_ENABLED else None
        with self._lock:
            tier_state = self._tier_states[tier]
            source = "tier"
            state = tier_state
            if zone_key is not None:
                zone_state = self._get_or_init_zone_state(tier, zone_key)
                if zone_state.episode_count >= ZONE_MIN_EPISODES:
                    state = zone_state
                    source = "zone"

            weights = dict(state.weights)
            exploration_applied = False

            if explore and DIRICHLET_EPSILON > 0:
                noise = _dirichlet_noise(len(_COEFFICIENT_KEYS), DIRICHLET_ALPHA)
                for i, k in enumerate(_COEFFICIENT_KEYS):
                    weights[k] = (
                        (1 - DIRICHLET_EPSILON) * weights[k]
                        + DIRICHLET_EPSILON * noise[i]
                    )
                weights = _project_simplex(weights)
                exploration_applied = True

            return {
                **weights,
                "tier": tier,
                "zone": (zone_key.split("::", 1)[1] if zone_key else None),
                "source": source,
                "version": state.policy_version,
                "episode_count": state.episode_count,
                "baseline_ema": round(state.baseline_ema, 4),
                "exploration_applied": exploration_applied,
                "rl_controlled": True,
            }

    def record_outcome(
        self,
        user_tier: str,
        actual_latency_ms: float,
        sla_ms: float,
        actual_carbon_g: float,
        actual_cost_units: float,
        accuracy_outcome: str,          # "clean" | "quality_retry" | "fallback" | "timeout"
        selected_scores: dict[str, float],   # {carbon_score, latency_score, accuracy_score, cost_score}
        all_candidate_scores: list[dict[str, float]],  # list of per-candidate score dicts
        request_id: str = "",
        zone: str | None = None,
        grid_carbon_g_per_kwh: float | None = None,
    ) -> dict[str, Any]:
        """
        Core online learning step: observe actual outcome and update weights.

        Called asynchronously after each inference completes.
        Thread-safe; non-blocking for the calling request handler.

        Updates both the global per-tier policy *and* (when ``zone`` is given
        and zone learning is enabled) the per-zone specialised policy. Each
        update is independently bounded by the simplex projection.

        Returns the update metadata (logged to audit trail).
        """
        tier = _normalise_tier(user_tier)
        zone_key = self._zone_key(tier, zone) if ZONE_LEARNING_ENABLED else None

        # ── 1. Compute reward signal ──────────────────────────────────────────
        reward = _compute_reward(
            actual_latency_ms=actual_latency_ms,
            sla_ms=sla_ms,
            actual_carbon_g=actual_carbon_g,
            actual_cost_units=actual_cost_units,
            accuracy_outcome=accuracy_outcome,
            grid_carbon_g_per_kwh=grid_carbon_g_per_kwh,
            selected_scores=selected_scores,
            all_candidate_scores=all_candidate_scores,
        )

        grads = _compute_policy_gradient(
            selected_scores,
            all_candidate_scores,
            softmax_temp=SOFTMAX_TEMP,
        )

        with self._lock:
            tier_update = self._apply_update(self._tier_states[tier], reward, grads)
            zone_update = None
            if zone_key is not None:
                zone_state = self._get_or_init_zone_state(tier, zone_key)
                zone_update = self._apply_update(zone_state, reward, grads)

            self._dirty = True

            update_meta = {
                "request_id": request_id,
                "tier": tier,
                "zone": (zone_key.split("::", 1)[1] if zone_key else None),
                "reward": round(reward, 6),
                "grads": {k: round(v, 6) for k, v in grads.items()},
                "tier_update": tier_update,
                "zone_update": zone_update,
            }

        logger.debug(
            "RL update: tier=%s zone=%s R=%.4f", tier, zone_key or "-", reward,
        )
        return update_meta

    def _apply_update(
        self,
        state: _TierState,
        reward: float,
        grads: dict[str, float],
    ) -> dict[str, Any]:
        """Apply one REINFORCE step to a single state object. Caller holds lock."""
        t = state.episode_count
        advantage = reward - state.baseline_ema
        alpha_t = ALPHA_0 / (1.0 + math.sqrt(max(t, 1)))

        old_weights = dict(state.weights)
        for k in _COEFFICIENT_KEYS:
            state.weights[k] += alpha_t * advantage * grads.get(k, 0.0)
        state.weights = _project_simplex(state.weights)

        state.baseline_ema = (
            BASELINE_EMA_BETA * state.baseline_ema
            + (1.0 - BASELINE_EMA_BETA) * reward
        )
        state.reward_history.append(round(reward, 6))
        state.episode_count += 1
        state.last_updated = time.time()

        converged = _check_convergence(state.reward_history)
        if converged:
            state.policy_version += 1

        return {
            "episode": t + 1,
            "advantage": round(advantage, 6),
            "alpha_t": round(alpha_t, 6),
            "old_weights": {k: round(v, 4) for k, v in old_weights.items()},
            "new_weights": {k: round(v, 4) for k, v in state.weights.items()},
            "baseline_ema": round(state.baseline_ema, 6),
            "policy_version": state.policy_version,
            "converged_this_step": converged,
        }

    def status(self) -> dict[str, Any]:
        """Return per-tier and per-zone RL status (read-only)."""
        with self._lock:
            tiers = {tier: _summarise_state(state) for tier, state in self._tier_states.items()}
            zones: dict[str, dict[str, Any]] = {}
            for key, state in self._zone_states.items():
                tier, zone = key.split("::", 1)
                bucket = zones.setdefault(tier, {})
                summary = _summarise_state(state)
                summary["active"] = state.episode_count >= ZONE_MIN_EPISODES
                bucket[zone] = summary
            return {
                "rl_enabled": True,
                "alpha_0": ALPHA_0,
                "baseline_ema_beta": BASELINE_EMA_BETA,
                "dirichlet_epsilon": DIRICHLET_EPSILON,
                "exploration_enabled": EXPLORATION_ENABLED,
                "zone_learning_enabled": ZONE_LEARNING_ENABLED,
                "zone_min_episodes": ZONE_MIN_EPISODES,
                "reward_lambda_grid_quality": REWARD_LAMBDA_GRID_QUALITY,
                "grid_dirty_threshold": GRID_DIRTY_THRESHOLD,
                "w_min": W_MIN,
                "tiers": tiers,
                "zones": zones,
            }

    def reward_history(self, tier: str | None = None, last_n: int = 100) -> dict[str, Any]:
        """Return recent reward observations per tier."""
        with self._lock:
            if tier:
                t = _normalise_tier(tier)
                history = list(self._tier_states[t].reward_history)[-last_n:]
                return {t: history}
            return {
                t: list(state.reward_history)[-last_n:]
                for t, state in self._tier_states.items()
            }

    def reset_tier(self, tier: str) -> dict[str, Any]:
        """
        Safety valve: reset a tier's weights to initial policy.
        NOT exposed through UI — only callable via /api/rl/reset/{tier}.
        Drops any zone-specialised states for that tier as well.
        """
        tier = _normalise_tier(tier)
        initial = _load_initial_weights(self._policy_config_path)
        with self._lock:
            self._tier_states[tier] = _TierState(initial.get(tier, _DEFAULT_WEIGHTS_BY_TIER[tier]))
            for key in list(self._zone_states.keys()):
                if key.startswith(f"{tier}::"):
                    del self._zone_states[key]
            self._dirty = True

        logger.info("RL tier=%s reset to initial policy (zone states cleared)", tier)
        return self.status()

    def reset_zone(self, tier: str, zone: str) -> dict[str, Any]:
        """Drop the zone-specialised state for (tier, zone); falls back to tier policy."""
        tier = _normalise_tier(tier)
        zone_key = self._zone_key(tier, zone)
        with self._lock:
            if zone_key and zone_key in self._zone_states:
                del self._zone_states[zone_key]
                self._dirty = True
        return self.status()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._running = False
        if self._dirty:
            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load RL state. Falls back to initial policy from policies.json, then defaults."""
        initial_weights = _load_initial_weights(self._policy_config_path)

        if self._state_path.exists():
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8"))
                for tier in _TIERS:
                    tier_data = payload.get("tiers", {}).get(tier)
                    if tier_data:
                        self._tier_states[tier] = _TierState.from_dict(tier_data)
                    else:
                        self._tier_states[tier] = _TierState(
                            initial_weights.get(tier, _DEFAULT_WEIGHTS_BY_TIER[tier])
                        )
                for key, zone_data in (payload.get("zones") or {}).items():
                    if "::" not in key:
                        continue
                    self._zone_states[key] = _TierState.from_dict(zone_data)
                logger.info(
                    "RL state loaded from %s; tiers=%s zones=%d",
                    self._state_path,
                    {t: self._tier_states[t].episode_count for t in _TIERS},
                    len(self._zone_states),
                )
                return
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("RL state file corrupt, resetting: %s", exc)

        for tier in _TIERS:
            self._tier_states[tier] = _TierState(
                initial_weights.get(tier, _DEFAULT_WEIGHTS_BY_TIER[tier])
            )
        logger.info("RL controller initialised from initial policy weights")

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "saved_at": _utc_now_iso(),
                    "tiers": {t: s.to_dict() for t, s in self._tier_states.items()},
                    "zones": {k: s.to_dict() for k, s in self._zone_states.items()},
                }
            self._state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self._dirty = False
        except OSError as exc:
            logger.error("RL state save failed: %s", exc)

    def _start_background_saver(self) -> None:
        def _loop():
            while self._running:
                time.sleep(15)            # save at most every 15 s
                if self._dirty:
                    self._save()
        t = threading.Thread(target=_loop, daemon=True, name="rl-saver")
        t.start()
        self._save_thread = t


# ── Reward computation ────────────────────────────────────────────────────────

def _compute_reward(
    actual_latency_ms: float,
    sla_ms: float,
    actual_carbon_g: float,
    actual_cost_units: float,
    accuracy_outcome: str,
    grid_carbon_g_per_kwh: float | None = None,
    selected_scores: dict[str, float] | None = None,
    all_candidate_scores: list[dict[str, float]] | None = None,
) -> float:
    """
    Multi-objective reward ∈ [0, 1].
    Higher is better; combines SLA conformance, carbon, accuracy, cost, and
    (optionally) a grid-quality term that rewards picking a greener-than-average
    candidate when the grid itself is dirty.
    """
    # SLA conformance: linear penalty for overshoot
    sla_ms = max(sla_ms, 1.0)
    over = max(0.0, actual_latency_ms - sla_ms)
    r_sla = max(0.0, 1.0 - (over / sla_ms))

    # Carbon: 0 carbon → reward 1; carbon = CARBON_REFERENCE_G → reward 0
    r_carbon = max(0.0, 1.0 - (actual_carbon_g / max(CARBON_REFERENCE_G, 1e-9)))
    r_carbon = min(r_carbon, 1.0)

    # Accuracy: based on quality guardrail outcome
    accuracy_map = {
        "clean":         1.00,
        "quality_retry": 0.65,
        "fallback":      0.30,
        "timeout":       0.05,
    }
    r_accuracy = accuracy_map.get(accuracy_outcome, 0.50)

    # Cost: lower cost_units → higher reward
    r_cost = max(0.0, 1.0 - (actual_cost_units / max(COST_REFERENCE, 1e-9)))
    r_cost = min(r_cost, 1.0)

    # Grid-quality bonus (optional). 0 when the term is disabled or no grid
    # data is supplied. Bounded to [0, 1] before λ-weighting; total contribution
    # is capped by REWARD_LAMBDA_GRID_QUALITY, which defaults to 0 so existing
    # behaviour is preserved unless explicitly opted in.
    r_grid = 0.0
    lambda_grid = max(0.0, min(REWARD_LAMBDA_GRID_QUALITY, 0.25))
    if lambda_grid > 0 and grid_carbon_g_per_kwh is not None and selected_scores and all_candidate_scores:
        sel_cscore = float(selected_scores.get("carbon_score", 0.0))
        avg_cscore = sum(float(c.get("carbon_score", 0.0)) for c in all_candidate_scores) / max(len(all_candidate_scores), 1)
        # Dirtiness factor in [0, 1]: 0 when grid is clean, 1 at/above threshold
        dirt = min(max(float(grid_carbon_g_per_kwh) / max(GRID_DIRTY_THRESHOLD, 1.0), 0.0), 1.0)
        # Relative greenness in [0, 1]: how much greener than the candidate mean
        rel = max(0.0, min(sel_cscore - avg_cscore + 0.5, 1.0))
        r_grid = dirt * rel

    reward = (
        REWARD_LAMBDA_SLA      * r_sla
        + REWARD_LAMBDA_CARBON  * r_carbon
        + REWARD_LAMBDA_ACCURACY * r_accuracy
        + REWARD_LAMBDA_COST    * r_cost
        + lambda_grid           * r_grid
    )
    return round(float(min(max(reward, 0.0), 1.0)), 6)


# ── Policy gradient computation ───────────────────────────────────────────────

def _compute_policy_gradient(
    selected_scores: dict[str, float],
    all_candidate_scores: list[dict[str, float]],
    softmax_temp: float = SOFTMAX_TEMP,
) -> dict[str, float]:
    """
    REINFORCE log-probability gradient:
        ∇_w_i log π = score_i(selected) − E_π[score_i]

    where E_π[score_i] is the softmax-weighted average of score_i over all candidates.
    This is the correct likelihood-ratio gradient for a softmax policy.
    """
    if not all_candidate_scores:
        return {k: 0.0 for k in _COEFFICIENT_KEYS}

    # Compute softmax probabilities over CSS scores of all candidates
    css_scores = [c.get("css_score", 0.0) for c in all_candidate_scores]
    probs = _softmax([s / max(softmax_temp, 1e-6) for s in css_scores])

    # E_π[score_i] for each coefficient component
    expected: dict[str, float] = {}
    score_key_map = {
        "carbon":   "carbon_score",
        "latency":  "latency_score",
        "accuracy": "accuracy_score",
        "cost":     "cost_score",
    }
    for coef_key, score_key in score_key_map.items():
        expected[coef_key] = sum(
            p * c.get(score_key, 0.0) for p, c in zip(probs, all_candidate_scores)
        )

    # Gradient = selected_score_i − E_π[score_i]
    grads: dict[str, float] = {}
    for coef_key, score_key in score_key_map.items():
        grads[coef_key] = selected_scores.get(score_key, 0.0) - expected[coef_key]

    return grads


# ── Simplex projection ────────────────────────────────────────────────────────

def _project_simplex(weights: dict[str, float], w_min: float = W_MIN) -> dict[str, float]:
    """
    Project weights onto the probability simplex with a floor constraint w_i ≥ w_min.
    1. Clip all below w_min to w_min.
    2. Clip all above 1 to bound.
    3. Renormalise to sum = 1.
    Repeats until stable (handles the constraint propagation).
    """
    keys = list(_COEFFICIENT_KEYS)
    w = {k: float(weights.get(k, 1.0 / len(keys))) for k in keys}

    for _ in range(10):    # converges in ≤ 3 iterations in practice
        # Clip to [w_min, 1]
        w = {k: max(w_min, min(1.0, v)) for k, v in w.items()}
        total = sum(w.values())
        if total <= 0:
            total = 1.0
        w = {k: v / total for k, v in w.items()}
        if abs(sum(w.values()) - 1.0) < 1e-9:
            break

    return w


# ── Dirichlet noise ───────────────────────────────────────────────────────────

def _dirichlet_noise(n: int, alpha: float) -> list[float]:
    """
    Sample from Dirichlet(α, …, α) using the Gamma trick.
    Provides sparse perturbation to weights (à la AlphaZero).
    """
    samples = [random.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(samples) or 1.0
    return [s / total for s in samples]


# ── Softmax ───────────────────────────────────────────────────────────────────

def _softmax(logits: list[float]) -> list[float]:
    max_l = max(logits) if logits else 0.0
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


# ── Convergence detection ─────────────────────────────────────────────────────

def _check_convergence(history: deque[float]) -> bool:
    """
    Detect reward plateau: std-dev of last CONVERGENCE_WINDOW rewards < tolerance.
    Signals that the policy has found a stable region.
    """
    if len(history) < CONVERGENCE_WINDOW:
        return False
    recent = list(history)[-CONVERGENCE_WINDOW:]
    mean = sum(recent) / len(recent)
    variance = sum((r - mean) ** 2 for r in recent) / len(recent)
    return math.sqrt(variance) < CONVERGENCE_TOLERANCE


# ── State summarisation ───────────────────────────────────────────────────────

def _summarise_state(state: _TierState) -> dict[str, Any]:
    history = list(state.reward_history)
    recent = history[-20:] if len(history) >= 20 else history
    return {
        "weights": {k: round(v, 4) for k, v in state.weights.items()},
        "episode_count": state.episode_count,
        "policy_version": state.policy_version,
        "baseline_ema": round(state.baseline_ema, 4),
        "recent_avg_reward": round(sum(recent) / max(len(recent), 1), 4),
        "reward_trend": _reward_trend(history),
        "last_updated_iso": _epoch_to_iso(state.last_updated) if state.last_updated else None,
        "reward_history_count": len(history),
    }


# ── Reward trend ──────────────────────────────────────────────────────────────

def _reward_trend(history: list[float]) -> str:
    """Returns 'improving', 'stable', 'declining', or 'insufficient_data'."""
    if len(history) < 10:
        return "insufficient_data"
    mid = len(history) // 2
    first_half = sum(history[:mid]) / mid
    second_half = sum(history[mid:]) / max(len(history) - mid, 1)
    diff = second_half - first_half
    if diff > 0.02:
        return "improving"
    if diff < -0.02:
        return "declining"
    return "stable"


# ── Initial weight loader ─────────────────────────────────────────────────────

def _load_initial_weights(config_path: Path) -> dict[str, dict[str, float]]:
    """Load starting weights from policies.json (never modified by RL)."""
    try:
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            tiers = payload.get("tiers", {})
            result = {}
            for tier, cfg in tiers.items():
                raw = {k: float(cfg.get(k, 0.25)) for k in _COEFFICIENT_KEYS}
                result[tier] = _project_simplex(raw)
            if result:
                return result
    except Exception as exc:
        logger.warning("Could not load initial weights from %s: %s", config_path, exc)
    return {tier: _project_simplex(dict(w)) for tier, w in _DEFAULT_WEIGHTS_BY_TIER.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_tier(tier: str) -> str:
    t = (tier or "standard").strip().lower()
    return t if t in _TIERS else "standard"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Module-level singleton ────────────────────────────────────────────────────

_rl_instance: RLPolicyController | None = None
_rl_lock = threading.Lock()


def get_rl_controller() -> RLPolicyController:
    global _rl_instance
    if _rl_instance is None:
        with _rl_lock:
            if _rl_instance is None:
                _rl_instance = RLPolicyController()
    return _rl_instance
