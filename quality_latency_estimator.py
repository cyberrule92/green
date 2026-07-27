"""
quality_latency_estimator.py — learned per-prompt quality/latency estimator.

Motivation
----------
Prior-art model routers (e.g. the NVIDIA LLM Router blueprint) pick a model by
learning, from real usage, the *quality*, *latency* and *cost* each candidate
model will deliver for a given prompt — a CLIP/embedding front-end feeding a
small trained network. Our CSS ranker (`routing_policies.rank_routing_candidates`,
M1) instead scores candidates against **static** per-model accuracy/latency
baselines drawn from `config/routing_targets.json`. That is robust but blind to
per-prompt difficulty: a 12-token trivia prompt and a 400-token multi-step
reasoning prompt get the *same* accuracy/latency figures for a given model.

This module adds the learned-estimator idea *underneath* CSS without touching
the carbon dimension. For each (prompt, candidate variant) it predicts:

  * an **accuracy residual**  Δa  → adjusted_accuracy = baseline_accuracy + Δa
  * a **latency scale**       s   → adjusted_latency  = baseline_latency  · s

Those adjusted figures replace the static baselines that feed CSS's
`accuracy_score` and `latency_score`. Carbon (op + embodied) is computed
independently and is deliberately *not* adjusted here — the greenest-feasible
invariant is preserved; the estimator only sharpens the feasibility signals.

Design constraints
------------------
* **Dependency-free** — no torch/numpy. Plain Python linear models, one weight
  vector per (variant, target). Fits the repo's "no heavy ML in the request
  path" style and matches `rl_controller.py`.
* **Cold-start = identity.** All weights initialise to 0 ⇒ Δa = 0, s = 1.0, so a
  fresh deployment routes *exactly* as before. The estimator only ever moves off
  baseline once it has observed evidence. This makes it safe to ship enabled.
* **Online-learned.** Updated once per completed request from the observed
  wall-clock latency and an accuracy-outcome proxy (the same signal RL uses),
  via clamped SGD with L2 regularisation. Persisted to `data/ql_estimator_state.json`.
* **Conservative clamps.** Accuracy residual ∈ [-0.15, +0.15]; latency scale ∈
  [0.6, 2.5]. Even a badly-fit model cannot swing CSS wildly.

The learning signal is a *residual*, so the estimator degrades gracefully: with
sparse or noisy data it stays near baseline rather than diverging.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("green_ai.ql_estimator")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATE_PATH = Path(os.getenv("QL_ESTIMATOR_STATE_PATH", DATA_DIR / "ql_estimator_state.json"))

ENABLED = os.getenv("QL_ESTIMATOR_ENABLED", "true").lower() in {"1", "true", "yes"}

LR = float(os.getenv("QL_ESTIMATOR_LR", "0.02"))            # SGD learning rate
L2 = float(os.getenv("QL_ESTIMATOR_L2", "0.001"))           # ridge regularisation
# Minimum observations for a (variant) model before its predictions are trusted;
# below this the estimator still returns baseline (warm-up guard on top of the
# identity cold-start).
WARMUP_MIN_OBS = int(os.getenv("QL_ESTIMATOR_WARMUP", "8"))

# Conservative clamps (see module docstring).
ACC_RESIDUAL_CLAMP = float(os.getenv("QL_ESTIMATOR_ACC_CLAMP", "0.15"))
LAT_SCALE_MIN = float(os.getenv("QL_ESTIMATOR_LAT_MIN", "0.6"))
LAT_SCALE_MAX = float(os.getenv("QL_ESTIMATOR_LAT_MAX", "2.5"))
# Output-length head. Wider band than latency: verbosity varies far more between
# variants than speed does. The three-arm benchmark measured TinyLlama emitting
# 1.6x the tokens of Qwen2.5-1.5B on the same prompts (123.9 vs 75.5), which is
# why the "greener" model cost MORE carbon -- duration, and therefore carbon,
# scales with how much a model chooses to say.
LEN_SCALE_MIN = float(os.getenv("QL_ESTIMATOR_LEN_MIN", "0.35"))
LEN_SCALE_MAX = float(os.getenv("QL_ESTIMATOR_LEN_MAX", "3.0"))

# Accuracy-outcome → observed-accuracy factor. `clean` leaves accuracy at
# baseline (factor 1.0 ⇒ residual target 0); degraded outcomes pull the observed
# accuracy *below* baseline so the estimator learns to down-rate that variant on
# prompts with these features. Never inflates above baseline — a well-behaved,
# one-sided signal. Unknown/other outcomes are treated as clean.
_OUTCOME_FACTORS = (
    ("timeout", 0.50),
    ("insufficient", 0.80),
    ("off-topic", 0.65),
    ("retry", 0.90),
    ("fallback", 0.70),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _outcome_factor(accuracy_outcome: str) -> float:
    """Map a (possibly compound) accuracy_outcome string to an accuracy factor."""
    ao = (accuracy_outcome or "").lower()
    factor = 1.0
    for needle, f in _OUTCOME_FACTORS:
        if needle in ao:
            factor = min(factor, f)
    return factor


# ── Feature extraction ────────────────────────────────────────────────────────
# Feature vector is derived purely from the semantic profile (already computed by
# `infer_prompt_profile`) so both `adjust()` and `observe()` see identical inputs
# for the same request. First component is a bias term.
FEATURE_NAMES = (
    "bias",
    "token_count_norm",
    "complexity",
    "has_documents",
    "attachment_chars_norm",
    "is_stem",
    "is_reasoning",
    "conversation_depth_norm",
)
N_FEATURES = len(FEATURE_NAMES)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def extract_features(semantic_profile: dict[str, Any]) -> list[float]:
    """Build the fixed-length feature vector from a semantic profile dict."""
    sp = semantic_profile or {}
    token_count = float(sp.get("token_count") or 0.0)
    complexity = float(sp.get("complexity_score") or 0.0)
    has_docs = 1.0 if sp.get("has_attachments") else 0.0
    attach_chars = float(sp.get("attachment_characters") or 0.0)
    is_stem = 1.0 if sp.get("stem_domain") else 0.0
    intent = str(sp.get("intent") or "").lower()
    mode = str(sp.get("mode") or "").lower()
    is_reasoning = 1.0 if (mode == "accurate" or intent in {"analysis", "implementation", "troubleshooting"}) else 0.0
    conv_depth = float(sp.get("conversation_message_count") or 0.0)
    return [
        1.0,
        _clamp01(token_count / 512.0),
        _clamp01(complexity),
        has_docs,
        _clamp01(attach_chars / 4000.0),
        is_stem,
        is_reasoning,
        _clamp01(conv_depth / 12.0),
    ]


def _dot(w: list[float], x: list[float]) -> float:
    return math.fsum(wi * xi for wi, xi in zip(w, x))


# ── Per-variant learned model ─────────────────────────────────────────────────
class _VariantModel:
    """Three linear heads for one variant: accuracy residual, latency-scale
    residual, and output-length-scale residual."""

    __slots__ = ("w_acc", "w_lat", "w_len", "n_obs")

    def __init__(self, w_acc: list[float] | None = None, w_lat: list[float] | None = None,
                 w_len: list[float] | None = None, n_obs: int = 0):
        self.w_acc = w_acc if w_acc is not None else [0.0] * N_FEATURES
        self.w_lat = w_lat if w_lat is not None else [0.0] * N_FEATURES
        self.w_len = w_len if w_len is not None else [0.0] * N_FEATURES
        self.n_obs = n_obs

    def predict(self, x: list[float]) -> tuple[float, float, float]:
        """Return (accuracy_residual, latency_scale, length_scale)."""
        acc_resid = _dot(self.w_acc, x)
        acc_resid = max(-ACC_RESIDUAL_CLAMP, min(ACC_RESIDUAL_CLAMP, acc_resid))
        # Latency and length heads both predict a residual around scale 1.0.
        lat_scale = 1.0 + _dot(self.w_lat, x)
        lat_scale = max(LAT_SCALE_MIN, min(LAT_SCALE_MAX, lat_scale))
        len_scale = 1.0 + _dot(self.w_len, x)
        len_scale = max(LEN_SCALE_MIN, min(LEN_SCALE_MAX, len_scale))
        return acc_resid, lat_scale, len_scale

    def update(self, x: list[float], acc_target_resid: float, lat_target_resid: float,
               len_target_resid: float | None = None) -> None:
        """One clamped SGD step with L2 for each head.

        ``len_target_resid=None`` means the observation carried no usable output
        length, so the length head is left untouched. Passing 0.0 instead would
        train it *toward* identity, which is a different and wrong claim: absence
        of a measurement is not evidence that the model was averagely verbose.
        """
        acc_err = _dot(self.w_acc, x) - acc_target_resid
        lat_err = _dot(self.w_lat, x) - lat_target_resid
        len_err = None if len_target_resid is None else _dot(self.w_len, x) - len_target_resid
        for i in range(N_FEATURES):
            self.w_acc[i] -= LR * (acc_err * x[i] + L2 * self.w_acc[i])
            self.w_lat[i] -= LR * (lat_err * x[i] + L2 * self.w_lat[i])
            if len_err is not None:
                self.w_len[i] -= LR * (len_err * x[i] + L2 * self.w_len[i])
        self.n_obs += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "w_acc": [round(w, 6) for w in self.w_acc],
            "w_lat": [round(w, 6) for w in self.w_lat],
            "w_len": [round(w, 6) for w in self.w_len],
            "n_obs": self.n_obs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_VariantModel":
        def _fit(v: Any) -> list[float]:
            v = list(v or [])
            if len(v) < N_FEATURES:
                v = v + [0.0] * (N_FEATURES - len(v))
            return [float(z) for z in v[:N_FEATURES]]
        # w_len is absent from state written before the length head existed;
        # _fit turns that into zeros, i.e. an identity length prediction, so an
        # older state file keeps working and simply learns the head from scratch.
        return cls(_fit(d.get("w_acc")), _fit(d.get("w_lat")),
                   _fit(d.get("w_len")), int(d.get("n_obs", 0)))


# ── Estimator ─────────────────────────────────────────────────────────────────
class QualityLatencyEstimator:
    def __init__(self, state_path: str | Path = STATE_PATH):
        self._state_path = Path(state_path)
        self._lock = threading.RLock()
        self._models: dict[str, _VariantModel] = {}
        self._dirty = False
        self._running = True
        self._updates = 0
        self._load()
        self._start_background_saver()

    # -- prediction --
    def adjust(
        self,
        semantic_profile: dict[str, Any],
        variant: str,
        baseline_accuracy: float,
        baseline_latency_ms: float,
        baseline_output_tokens: float = 0.0,
    ) -> dict[str, float]:
        """
        Return adjusted accuracy / latency / expected output length for one
        candidate variant.

        Always returns baseline values when disabled, in warm-up, or on any
        error — the caller can use the result unconditionally.
        """
        result = {
            "accuracy": baseline_accuracy,
            "latency_ms": baseline_latency_ms,
            "output_tokens": baseline_output_tokens,
            "accuracy_residual": 0.0,
            "latency_scale": 1.0,
            "length_scale": 1.0,
            "applied": False,
        }
        if not ENABLED:
            return result
        try:
            variant = (variant or "").lower()
            with self._lock:
                model = self._models.get(variant)
                n_obs = model.n_obs if model else 0
            if model is None or n_obs < WARMUP_MIN_OBS:
                return result
            x = extract_features(semantic_profile)
            acc_resid, lat_scale, len_scale = model.predict(x)
            result.update({
                "accuracy": max(0.0, min(1.0, baseline_accuracy + acc_resid)),
                "latency_ms": max(1.0, baseline_latency_ms * lat_scale),
                "output_tokens": max(1.0, baseline_output_tokens * len_scale)
                if baseline_output_tokens > 0 else baseline_output_tokens,
                "accuracy_residual": round(acc_resid, 4),
                "latency_scale": round(lat_scale, 4),
                "length_scale": round(len_scale, 4),
                "applied": True,
            })
        except Exception as exc:  # never let the estimator break routing
            logger.warning("ql_estimator adjust failed (using baseline): %s", exc)
        return result

    # -- learning --
    def observe(
        self,
        semantic_profile: dict[str, Any],
        variant: str,
        baseline_accuracy: float,
        baseline_latency_ms: float,
        actual_latency_ms: float,
        accuracy_outcome: str,
        baseline_output_tokens: float = 0.0,
        actual_output_tokens: float = 0.0,
    ) -> None:
        """Update the selected variant's model from one observed outcome."""
        if not ENABLED:
            return
        try:
            variant = (variant or "").lower()
            if not variant or baseline_latency_ms <= 0.0:
                return
            x = extract_features(semantic_profile)

            # Accuracy residual target: baseline·factor − baseline (≤ 0).
            factor = _outcome_factor(accuracy_outcome)
            acc_target_resid = baseline_accuracy * factor - baseline_accuracy

            # Latency residual target: observed/baseline − 1, pre-clamped to the
            # same band the predictor is allowed to output so a single slow
            # outlier (queueing/RAG/guardrails) cannot drag the weights past the
            # usable range.
            ratio = actual_latency_ms / baseline_latency_ms
            lat_target_resid = max(LAT_SCALE_MIN - 1.0, min(LAT_SCALE_MAX - 1.0, ratio - 1.0))

            # Output-length residual target, clamped to the predictor's own band
            # for the same reason as latency. None when either side is unknown, so
            # update() skips the length head rather than being taught that the
            # response was averagely verbose.
            len_target_resid: float | None = None
            if baseline_output_tokens > 0 and actual_output_tokens > 0:
                len_ratio = actual_output_tokens / baseline_output_tokens
                len_target_resid = max(LEN_SCALE_MIN - 1.0,
                                       min(LEN_SCALE_MAX - 1.0, len_ratio - 1.0))

            with self._lock:
                model = self._models.get(variant)
                if model is None:
                    model = _VariantModel()
                    self._models[variant] = model
                model.update(x, acc_target_resid, lat_target_resid, len_target_resid)
                self._dirty = True
                self._updates += 1
        except Exception as exc:
            logger.warning("ql_estimator observe failed (non-critical): %s", exc)

    # -- introspection (for observability endpoint) --
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": ENABLED,
                "warmup_min_obs": WARMUP_MIN_OBS,
                "learning_rate": LR,
                "l2": L2,
                "length_scale_band": [LEN_SCALE_MIN, LEN_SCALE_MAX],
                "total_updates": self._updates,
                "feature_names": list(FEATURE_NAMES),
                "variants": {
                    v: {
                        "n_obs": m.n_obs,
                        "trusted": m.n_obs >= WARMUP_MIN_OBS,
                        "w_acc": [round(w, 5) for w in m.w_acc],
                        "w_lat": [round(w, 5) for w in m.w_lat],
                        "w_len": [round(w, 5) for w in m.w_len],
                    }
                    for v, m in sorted(self._models.items())
                },
            }

    # -- persistence --
    def _load(self) -> None:
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            models = payload.get("variants", {})
            with self._lock:
                self._models = {
                    str(v): _VariantModel.from_dict(d) for v, d in models.items()
                }
                self._updates = int(payload.get("total_updates", 0))
            logger.info("ql_estimator loaded %d variant models", len(self._models))
        except Exception as exc:
            logger.warning("ql_estimator load failed (starting fresh): %s", exc)
            self._models = {}

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "saved_at": _utc_now_iso(),
                    "total_updates": self._updates,
                    "feature_names": list(FEATURE_NAMES),
                    "variants": {v: m.to_dict() for v, m in self._models.items()},
                }
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._state_path)   # atomic
            self._dirty = False
        except OSError as exc:
            logger.error("ql_estimator state save failed: %s", exc)

    def _start_background_saver(self) -> None:
        def _loop():
            while self._running:
                time.sleep(15)
                if self._dirty:
                    self._save()
        t = threading.Thread(target=_loop, daemon=True, name="ql-estimator-saver")
        t.start()
        self._save_thread = t


# ── Singleton accessor ────────────────────────────────────────────────────────
_estimator: QualityLatencyEstimator | None = None
_estimator_lock = threading.Lock()


def get_quality_latency_estimator() -> QualityLatencyEstimator:
    global _estimator
    if _estimator is None:
        with _estimator_lock:
            if _estimator is None:
                _estimator = QualityLatencyEstimator()
    return _estimator
