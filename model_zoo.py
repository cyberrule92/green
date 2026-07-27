#!/usr/bin/env python3
"""
Model Zoo Service — Adaptive Green AI
Implements the versioned model registry described in Section 3.3 of the paper.
Provides:
- LLMCarbon operational and embodied carbon computation (Section 3.4.2)
- MoE sparse FLOP accounting and all-to-all overhead (Section 5.1)
- Expert health monitoring (Section 5.3)
- Hardware efficiency lookups from LLMCarbon tables (Section 4.1)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
# Resolve MODEL_ZOO_PATH: if env var is a relative path, anchor it to BASE_DIR
# so the zoo is always found regardless of the working directory at launch-time.
_raw_zoo_path = os.getenv("MODEL_ZOO_PATH", "")
if _raw_zoo_path:
    _p = Path(_raw_zoo_path)
    MODEL_ZOO_PATH = _p if _p.is_absolute() else BASE_DIR / _p
else:
    MODEL_ZOO_PATH = BASE_DIR / "config" / "model_zoo.json"

# LLMCarbon constants (Section 4.1)
# Watts to kWh conversion factor
_W_TO_KWH = 1.0 / (1000.0 * 3600.0)
# gCO2/kWh to gCO2/J
_GCO2_KWH_TO_GCO2_J = 1.0 / 3_600_000.0
# Default embodied carbon assumptions (LLMCarbon Section 4.1)
_DEFAULT_LIFETIME_YEARS = 5.0
_DEFAULT_ANNUAL_INFERENCES = 100_000
# Seconds per year
_SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Embodied amortisation basis.
#
# Embodied carbon must be amortised over a quantity that is INDEPENDENT of the
# duration it is later multiplied by. The legacy basis (annual_inference_volume ×
# latency_ms_p50) was not: callers passed latency_ms_p50 back in as the duration,
# the two cancelled, and every request was billed the same constant regardless of
# how long it actually held the device.
#
# device_utilization is the fraction of the device's wall-clock lifetime spent
# serving. It is the single free parameter in the embodied term and the one a
# reviewer should challenge; see `embodied_defaults` in config/model_zoo.json.
_DEFAULT_DEVICE_UTILIZATION = 0.35
# Fraction of the physical board this deployment owns. A vGPU slice is billed
# only for its share of the board's manufacturing carbon.
_DEFAULT_DEVICE_SHARE = 1.0


class ModelZooService:
    """
    Versioned model registry implementing LLMCarbon carbon accounting.

    Each model entry stores:
    - FLOP count (dense and sparse for MoE)
    - Hardware efficiency (HE) from empirical tables
    - PUE per data-centre type
    - Manufacturing carbon (mfg_carbon_kg) for embodied amortisation
    - MoE topology: num_experts, active_experts_k, all_to_all_overhead_ratio
    - Multi-region metadata: grid_zone, network_latency_ms
    """

    def __init__(self, zoo_path: str | Path = MODEL_ZOO_PATH):
        self.zoo_path = Path(zoo_path)
        self._lock = threading.RLock()
        self._models: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._expert_health: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(m) for m in self._models]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            for m in self._models:
                if m.get("id") == model_id:
                    return dict(m)
        return None

    def get_version(self) -> str:
        return self._metadata.get("version", "unversioned")

    def available_targets(self) -> list[dict[str, Any]]:
        """Return models flagged as available, enriched with live expert health."""
        with self._lock:
            targets = []
            for m in self._models:
                entry = dict(m)
                if m.get("moe") and m.get("id") in self._expert_health:
                    entry["expert_health"] = dict(self._expert_health[m["id"]])
                targets.append(entry)
        return targets

    # ------------------------------------------------------------------
    # LLMCarbon Carbon Computation (Section 3.4.2 & 4.1)
    # ------------------------------------------------------------------

    def compute_operational_carbon(
        self,
        model_id: str,
        grid_carbon_g_per_kwh: float,
        inference_duration_s: float,
        token_count: int = 256,
    ) -> dict[str, float]:
        """
        Compute operational carbon using the LLMCarbon formula:
            C_op = (FLOPs / HE) × TDP × PUE × CI × t / (1000 × 3600)

        For MoE models HE is adjusted for all-to-all communication overhead:
            HE_moe = HE × (1 - all_to_all_overhead_ratio)

        Returns dict with breakdown fields.
        """
        model = self.get_model(model_id)
        if not model:
            return {"total_carbon_g": 0.0, "error": f"Model {model_id} not found"}

        # Choose FLOP count: sparse for MoE, dense otherwise
        if model.get("moe") and model.get("flop_count_per_token_sparse"):
            flop_per_token = float(model["flop_count_per_token_sparse"])
            sparse_path = True
        else:
            flop_per_token = float(model.get("flop_count_per_token", 0))
            sparse_path = False

        total_flops = flop_per_token * max(token_count, 1)

        # Hardware Efficiency — from LLMCarbon empirical tables
        he = float(model.get("hardware_efficiency", 0.6))

        # MoE all-to-all penalty (Section 5.1)
        all_to_all_ratio = float(model.get("all_to_all_overhead_ratio", 0.0))
        if model.get("moe") and all_to_all_ratio > 0:
            he_effective = he * (1.0 - all_to_all_ratio)
        else:
            he_effective = he
        # Same numerical guard as compute_request_carbon: he_effective is a
        # divisor below, and an entry with hardware_efficiency 0 or an
        # all_to_all_overhead_ratio of 1.0 would make it zero. No shipped model
        # does that today, but register_model() and the model-zoo auto-updater
        # can add entries that would, and the two functions must not disagree
        # about whether such an entry is priceable.
        he_effective = max(he_effective, 0.05)

        # TDP (peak power in Watts) and PUE
        tdp_w = float(model.get("power_tdp_w", 100.0))
        pue = float(model.get("pue", 1.3))

        # LLMCarbon operational formula
        # C_op (gCO2) = (FLOPs / HE_eff) × TDP × PUE × CI × t  [all in consistent units]
        # Simplified to power-based form: energy = TDP × t × PUE / HE_eff (Joules-equivalent factor)
        # Then C_op = energy_kwh × CI_g_per_kwh
        energy_kwh = (tdp_w * inference_duration_s * pue) / (he_effective * 1000.0 * 3600.0)
        regional_multiplier = float(model.get("region_carbon_multiplier", 1.0))
        op_carbon_g = energy_kwh * grid_carbon_g_per_kwh * regional_multiplier

        return {
            "op_carbon_g": round(op_carbon_g, 8),
            "energy_kwh": round(energy_kwh, 10),
            "tdp_w": tdp_w,
            "pue": pue,
            "hardware_efficiency": he,
            "he_effective": round(he_effective, 4),
            "flop_per_token": flop_per_token,
            "total_flops": total_flops,
            "sparse_path": sparse_path,
            "all_to_all_overhead_ratio": all_to_all_ratio,
            "grid_carbon_g_per_kwh": grid_carbon_g_per_kwh,
            "inference_duration_s": inference_duration_s,
        }

    def compute_embodied_rate(self, model_id: str) -> dict[str, float]:
        """
        Embodied carbon rate in gCO2e per device-second (LLMCarbon Section 4.1):

            rate = (mfg_carbon_kg × 1000 × device_share)
                 / (device_lifetime_years × seconds_per_year × device_utilization)

        The denominator is deliberately built from device wall-clock lifetime and a
        duty cycle, NOT from annual_inference_volume × latency_ms_p50. The latter is
        the duration this rate gets multiplied by, so the two cancelled and every
        request was billed an identical constant. See _DEFAULT_DEVICE_UTILIZATION.
        """
        model = self.get_model(model_id)
        if not model:
            return {"emb_rate_g_per_device_s": 0.0}

        mfg_carbon_kg = float(model.get("mfg_carbon_kg", 143.0))
        lifetime_years = float(model.get("device_lifetime_years", _DEFAULT_LIFETIME_YEARS))
        utilization = float(model.get("device_utilization", _DEFAULT_DEVICE_UTILIZATION))
        share = float(model.get("device_share", _DEFAULT_DEVICE_SHARE))

        device_seconds = lifetime_years * _SECONDS_PER_YEAR * utilization
        if device_seconds <= 0:
            return {"emb_rate_g_per_device_s": 0.0}

        rate = (mfg_carbon_kg * 1000.0 * share) / device_seconds

        return {
            "emb_rate_g_per_device_s": rate,
            "mfg_carbon_kg": mfg_carbon_kg,
            "lifetime_years": lifetime_years,
            "device_utilization": utilization,
            "device_share": share,
            "amortised_device_seconds": device_seconds,
        }

    def compute_embodied_carbon(
        self,
        model_id: str,
        inference_duration_s: float,
    ) -> dict[str, float]:
        """
        Embodied carbon attributable to a single request: rate × duration.
        Pass a MEASURED duration — passing latency_ms_p50 back in is what made the
        legacy implementation collapse to a constant.
        """
        rate_info = self.compute_embodied_rate(model_id)
        rate = float(rate_info.get("emb_rate_g_per_device_s", 0.0))
        if rate <= 0:
            return {"emb_carbon_g": 0.0}

        return {
            "emb_carbon_g": round(rate * max(inference_duration_s, 0.0), 8),
            "mfg_carbon_kg": rate_info.get("mfg_carbon_kg", 0.0),
            "lifetime_years": rate_info.get("lifetime_years", 0.0),
            "device_utilization": rate_info.get("device_utilization", 0.0),
            "device_share": rate_info.get("device_share", 0.0),
            "emb_rate_g_per_s": round(rate, 10),
        }

    def compute_total_carbon(
        self,
        model_id: str,
        grid_carbon_g_per_kwh: float,
        inference_duration_s: float,
        token_count: int = 256,
    ) -> dict[str, float]:
        """
        Total carbon = operational + embodied (LLMCarbon unified formula).
        Returns a breakdown dict suitable for audit logging.
        """
        op = self.compute_operational_carbon(
            model_id, grid_carbon_g_per_kwh, inference_duration_s, token_count
        )
        emb = self.compute_embodied_carbon(model_id, inference_duration_s)
        total = op.get("op_carbon_g", 0.0) + emb.get("emb_carbon_g", 0.0)
        return {
            "total_carbon_g": round(total, 8),
            "operational": op,
            "embodied": emb,
        }

    def compute_request_carbon(
        self,
        model_id: str,
        grid_carbon_g_per_kwh: float,
        measured_duration_s: float,
        output_tokens: int = 0,
        input_tokens: int = 0,
    ) -> dict[str, Any]:
        """
        EX-POST carbon for one inference leg, billed against MEASURED wall-clock.

        This is the source of truth for what a request actually cost. It is distinct
        from the ex-ante `estimated_carbon_g` produced during CSS ranking, which is a
        spec-derived forecast used to CHOOSE a candidate before it runs.

            energy_j = tdp_w × measured_duration_s × pue / he_effective
            op_g     = energy_j / 3.6e6 × grid_ci × region_carbon_multiplier
            emb_g    = emb_rate_g_per_device_s × measured_duration_s

        LIMITATION: tdp_w is a spec constant, not a reading. GPU power telemetry is
        unavailable on a vGPU slice (nvidia-smi power.draw returns [N/A]), so the
        operational term is an UPPER BOUND on true draw, not a measurement. Duration
        and token counts are measured.
        """
        model = self.get_model(model_id)
        if not model:
            return {"total_carbon_g": 0.0, "error": f"Model {model_id} not found"}

        duration_s = max(float(measured_duration_s), 0.0)

        he = float(model.get("hardware_efficiency", 0.6))
        all_to_all = float(model.get("all_to_all_overhead_ratio", 0.0))
        he_effective = he * (1.0 - all_to_all) if model.get("moe") and all_to_all > 0 else he
        he_effective = max(he_effective, 0.05)  # numerical guard

        tdp_w = float(model.get("power_tdp_w", 100.0))
        pue = float(model.get("pue", 1.3))
        region_mult = float(model.get("region_carbon_multiplier", 1.0))

        energy_j = (tdp_w * duration_s * pue) / he_effective
        op_carbon_g = (energy_j / 3_600_000.0) * grid_carbon_g_per_kwh * region_mult

        rate_info = self.compute_embodied_rate(model_id)
        emb_rate = float(rate_info.get("emb_rate_g_per_device_s", 0.0))
        emb_carbon_g = emb_rate * duration_s

        total_carbon_g = op_carbon_g + emb_carbon_g

        return {
            "model_id": model_id,
            "model_variant": model.get("model_variant", ""),
            "duration_s": round(duration_s, 4),
            "energy_j": round(energy_j, 4),
            "energy_wh": round(energy_j / 3600.0, 8),
            "energy_kwh": round(energy_j / 3_600_000.0, 10),
            "op_carbon_g": round(op_carbon_g, 8),
            "emb_carbon_g": round(emb_carbon_g, 8),
            "total_carbon_g": round(total_carbon_g, 8),
            "emb_rate_g_per_device_s": round(emb_rate, 10),
            "device_utilization": rate_info.get("device_utilization", 0.0),
            "device_share": rate_info.get("device_share", 0.0),
            "tdp_w": tdp_w,
            "pue": pue,
            "he_effective": round(he_effective, 4),
            "grid_carbon_g_per_kwh": grid_carbon_g_per_kwh,
            "output_tokens": int(output_tokens),
            "input_tokens": int(input_tokens),
            # None, not carbon/1, when no tokens are attributed to this leg. A
            # leg discarded by a quality retry produced tokens the user never
            # saw and that are not counted anywhere, so its per-token rate is
            # unknown — dividing by a floor of 1 would book the whole leg's
            # carbon as the cost of a single token and write that into the ledger.
            "ug_per_output_token": (
                round((total_carbon_g / int(output_tokens)) * 1e6, 3)
                if int(output_tokens) > 0
                else None
            ),
            "power_basis": "spec_tdp_upper_bound",
        }

    # ------------------------------------------------------------------
    # MoE Profiling (Section 5.1)
    # ------------------------------------------------------------------

    def estimate_moe_communication_time_ms(
        self,
        model_id: str,
        token_count: int,
        d_model: int = 4096,
    ) -> float:
        """
        Estimate all-to-all communication time for MoE models.
        T_comm = (k × token_count × d_model × element_size) / bandwidth_Bps
        Paper Section 5.1 formula.
        """
        model = self.get_model(model_id)
        if not model or not model.get("moe"):
            return 0.0

        k = int(model.get("active_experts_k", 2))
        bandwidth_gbps = float(model.get("expert_bandwidth_gbps", 100.0))
        bandwidth_bps = bandwidth_gbps * 1e9
        element_bytes = 2  # fp16

        comm_bytes = k * token_count * d_model * element_bytes
        comm_s = comm_bytes / bandwidth_bps
        return round(comm_s * 1000.0, 3)  # ms

    def compute_load_balance_metric(self, token_routing: list[int]) -> float:
        """
        Load balance metric per FT-MoE (Section 5.2):
            LB = N × min(n_e) / sum(n_e)
        Returns value in [0, 1]; <0.8 indicates imbalance.
        """
        if not token_routing:
            return 1.0
        total = sum(token_routing)
        if total == 0:
            return 1.0
        n = len(token_routing)
        return round(n * min(token_routing) / total, 4)

    # ------------------------------------------------------------------
    # MoE Expert Placement (Section 5.1, Algorithm 3)
    # ------------------------------------------------------------------

    def plan_expert_placement(
        self,
        model_id: str,
        device_topology: list[dict[str, Any]] | None = None,
        token_routing: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Distribute MoE experts across available devices to minimise all-to-all
        traffic and balance per-device load (paper §5.1, Algorithm 3).

        Strategy (load- and capacity-aware round-robin):
          1. Filter devices to those whose hardware family appears in
             ``hardware_affinity`` and that report unavailable expert slots.
          2. Sort experts by descending estimated load (from ``token_routing``
             when supplied; otherwise uniform). Heaviest experts placed first
             so they land on the highest-capacity devices.
          3. Assign each expert to the device with the smallest cumulative
             load that still has expert-slot headroom.
          4. Estimate all-to-all comm overhead and per-device load skew and
             return the placement plan.

        Returns ``{"placement": {expert_id: device_id}, "device_loads": {...},
        "skew": float, "estimated_comm_ms": float, "fallback": bool}``.
        ``fallback=True`` indicates that a viable placement could not be
        found and the caller should route to a dense fallback per §5.3.
        """
        model = self.get_model(model_id)
        if not model or not model.get("moe"):
            return {"placement": {}, "fallback": False, "reason": "not_an_moe_model"}

        num_experts = int(model.get("num_experts", 0))
        if num_experts <= 0:
            return {"placement": {}, "fallback": True, "reason": "num_experts_unset"}

        affinity = set(model.get("hardware_affinity") or [])
        topology = device_topology or self._default_device_topology(model)
        # Keep only devices that are compatible and have capacity > 0
        eligible = [
            d for d in topology
            if (not affinity or d.get("hardware_family") in affinity)
            and int(d.get("expert_slots", 0)) > 0
        ]
        if not eligible:
            return {"placement": {}, "fallback": True, "reason": "no_eligible_devices"}

        # Per-expert estimated load (token count). Default uniform.
        if token_routing and len(token_routing) >= num_experts:
            loads = [(idx, float(token_routing[idx])) for idx in range(num_experts)]
        else:
            loads = [(idx, 1.0) for idx in range(num_experts)]
        # Heaviest first
        loads.sort(key=lambda kv: kv[1], reverse=True)

        # Mutable per-device state (capacity + cumulative load)
        device_state = {
            d["id"]: {
                "remaining": int(d["expert_slots"]),
                "load": 0.0,
                "capacity": float(d.get("relative_capacity", 1.0)) or 1.0,
                "hardware_family": d.get("hardware_family"),
            }
            for d in eligible
        }

        placement: dict[int, str] = {}
        for expert_id, load in loads:
            # Pick the device with the smallest load/capacity ratio that has
            # remaining slots. Stable tie-breaker on device id.
            candidates = [
                (state["load"] / state["capacity"], device_id)
                for device_id, state in device_state.items()
                if state["remaining"] > 0
            ]
            if not candidates:
                # Out of expert slots before all experts placed
                return {
                    "placement": placement,
                    "fallback": True,
                    "reason": "insufficient_expert_slots",
                    "placed": len(placement),
                    "required": num_experts,
                }
            candidates.sort()
            chosen = candidates[0][1]
            placement[expert_id] = chosen
            device_state[chosen]["load"] += load
            device_state[chosen]["remaining"] -= 1

        # Skew metric: 1 = perfectly balanced, 0 = pathological
        per_device_loads = [s["load"] for s in device_state.values() if s["load"] > 0]
        if per_device_loads:
            skew = round(min(per_device_loads) / max(per_device_loads), 4)
        else:
            skew = 1.0

        device_loads = {
            device_id: {
                "load":            round(state["load"], 3),
                "remaining_slots": state["remaining"],
                "hardware_family": state["hardware_family"],
            }
            for device_id, state in device_state.items()
        }

        # Cross-device comm overhead estimate (heuristic: scales with the
        # fraction of experts not co-located on the same device)
        unique_devices = {device_id for device_id in placement.values()}
        cross_device_factor = max(0.0, (len(unique_devices) - 1) / max(num_experts, 1))
        base_comm_ms = self.estimate_moe_communication_time_ms(model_id, token_count=256)
        estimated_comm_ms = round(base_comm_ms * (1.0 + cross_device_factor), 3)

        return {
            "model_id":          model_id,
            "placement":         {str(k): v for k, v in placement.items()},
            "device_loads":      device_loads,
            "skew":              skew,
            "estimated_comm_ms": estimated_comm_ms,
            "unique_devices":    sorted(unique_devices),
            "fallback":          False,
            "strategy":          "load-aware-round-robin",
        }

    def _default_device_topology(self, model: dict[str, Any]) -> list[dict[str, Any]]:
        """
        When no live device topology is supplied, synthesise one from the
        model's ``hardware_affinity`` so the placement planner can still run
        in single-node prototype mode. Each affinity entry becomes a synthetic
        device with capacity sized for the active-expert count.
        """
        affinity = list(model.get("hardware_affinity") or ["nvidia-a100"])
        num_experts = int(model.get("num_experts", 8))
        # Spread expert slots roughly evenly across affinity entries; ensure
        # cumulative slots >= num_experts so placement can't trivially fail.
        slots_per_device = max(1, (num_experts + len(affinity) - 1) // len(affinity))
        return [
            {
                "id":               f"synthetic-{family}-{idx}",
                "hardware_family":  family,
                "expert_slots":     slots_per_device,
                "relative_capacity": 1.0,
            }
            for idx, family in enumerate(affinity)
        ]

    # ------------------------------------------------------------------
    # Expert Health Monitoring (Section 5.3)
    # ------------------------------------------------------------------

    def update_expert_health(
        self,
        model_id: str,
        healthy_experts: int,
        total_experts: int,
    ) -> None:
        with self._lock:
            self._expert_health[model_id] = {
                "healthy_experts": healthy_experts,
                "total_experts": total_experts,
                "health_ratio": round(healthy_experts / max(total_experts, 1), 3),
                "last_check_ts": time.time(),
                "last_check_iso": _utc_now_iso(),
                "degraded": healthy_experts < total_experts,
            }

    def get_expert_health(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._expert_health.get(model_id, {}))

    def is_moe_healthy(self, model_id: str, min_ratio: float = 0.75) -> bool:
        health = self.get_expert_health(model_id)
        if not health:
            return True  # assume healthy if no data
        return health.get("health_ratio", 1.0) >= min_ratio

    # ------------------------------------------------------------------
    # FT-MoE Reconciler (Section 5.3)
    # ------------------------------------------------------------------

    def reconcile_moe_health(
        self,
        health_probe: callable | None = None,
        rebalance_threshold: float = 0.75,
        disable_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        One reconciler tick (paper §5.3):
          1. For every MoE model, ask the supplied ``health_probe(model_id)``
             for current ``(healthy_experts, total_experts)`` (when no probe
             is supplied, use the registry's last-known health).
          2. Update internal health state and the model's ``available`` flag:
             - ratio < disable_threshold  → mark model unavailable (route to
               dense fallback)
             - disable_threshold ≤ ratio < rebalance_threshold → keep model
               available but flag ``degraded=True`` so the placement planner
               recomputes on the next request
             - ratio ≥ rebalance_threshold → restore availability if it was
               previously disabled by us.
          3. Trigger an automated repair signal (``repair_requested``) when a
             previously-healthy model degrades — callers wire this to whatever
             expert-replica restart mechanism they have (Kubernetes, vLLM
             server, etc.).
        Returns a per-model action summary suitable for logging.
        """
        actions: dict[str, dict[str, Any]] = {}
        with self._lock:
            moe_models = [m for m in self._models if m.get("moe")]

        for model in moe_models:
            model_id = model.get("id")
            try:
                if health_probe is not None:
                    healthy, total = health_probe(model_id)
                else:
                    cached = self.get_expert_health(model_id)
                    healthy = int(cached.get("healthy_experts", model.get("num_experts", 0)))
                    total = int(cached.get("total_experts", model.get("num_experts", 0)))
            except Exception as exc:  # noqa: BLE001 — probe may be flaky
                logger.warning("Expert health probe failed for %s: %s", model_id, exc)
                actions[model_id] = {"action": "probe_failed", "error": str(exc)}
                continue

            self.update_expert_health(model_id, healthy, total)
            ratio = healthy / max(total, 1)

            with self._lock:
                live_entry = next((m for m in self._models if m.get("id") == model_id), None)
                if live_entry is None:
                    actions[model_id] = {"action": "model_missing"}
                    continue
                was_available = bool(live_entry.get("available"))
                was_disabled_by_reconciler = bool(live_entry.get("disabled_by_reconciler"))

                if ratio < disable_threshold:
                    if was_available:
                        live_entry["available"] = False
                        live_entry["disabled_by_reconciler"] = True
                        live_entry["disabled_reason"] = f"expert_health_ratio={ratio:.2f}"
                        action_label = "disabled"
                    else:
                        # Already unavailable (operator-disabled or previously
                        # disabled by us). Don't claim to have disabled it again.
                        action_label = "already_unavailable"
                    actions[model_id] = {
                        "action":           action_label,
                        "ratio":            round(ratio, 3),
                        "repair_requested": True,
                    }
                elif ratio < rebalance_threshold:
                    live_entry["degraded"] = True
                    live_entry["disabled_reason"] = None
                    actions[model_id] = {
                        "action":           "degraded_rebalance",
                        "ratio":            round(ratio, 3),
                        "repair_requested": True,
                    }
                else:
                    # Healthy: clear flags; restore availability only if WE
                    # were the ones that disabled it.
                    live_entry["degraded"] = False
                    if was_disabled_by_reconciler and not was_available:
                        live_entry["available"] = True
                        live_entry["disabled_by_reconciler"] = False
                        live_entry["disabled_reason"] = None
                        actions[model_id] = {
                            "action":           "restored",
                            "ratio":            round(ratio, 3),
                            "repair_requested": False,
                        }
                    else:
                        actions[model_id] = {
                            "action":           "ok",
                            "ratio":            round(ratio, 3),
                            "repair_requested": False,
                        }

        return actions

    def start_health_reconciler(
        self,
        health_probe: callable | None = None,
        interval_seconds: float = 10.0,
        rebalance_threshold: float = 0.75,
        disable_threshold: float = 0.5,
    ) -> threading.Thread:
        """
        Start a daemon thread that calls :py:meth:`reconcile_moe_health`
        every ``interval_seconds`` (paper §5.3 specifies a 10-second cadence
        for the expert health pings). Idempotent: a second call is ignored.
        """
        if getattr(self, "_reconciler_thread", None) and self._reconciler_thread.is_alive():
            return self._reconciler_thread

        stop_event = threading.Event()
        self._reconciler_stop = stop_event

        def _loop() -> None:
            while not stop_event.is_set():
                try:
                    self.reconcile_moe_health(
                        health_probe=health_probe,
                        rebalance_threshold=rebalance_threshold,
                        disable_threshold=disable_threshold,
                    )
                except Exception as exc:  # noqa: BLE001 — never crash the loop
                    logger.exception("FT-MoE reconciler tick failed: %s", exc)
                stop_event.wait(interval_seconds)

        thread = threading.Thread(target=_loop, name="ft-moe-reconciler", daemon=True)
        thread.start()
        self._reconciler_thread = thread
        logger.info("FT-MoE reconciler started (interval=%.1fs)", interval_seconds)
        return thread

    def stop_health_reconciler(self) -> None:
        stop_event = getattr(self, "_reconciler_stop", None)
        if stop_event is not None:
            stop_event.set()
        self._reconciler_thread = None
        self._reconciler_stop = None

    # ------------------------------------------------------------------
    # Regional Carbon Routing (Section 3.5.3)
    # ------------------------------------------------------------------

    def regional_score(
        self,
        model_entry: dict[str, Any],
        region_carbon_map: dict[str, float],
        w_carbon: float = 0.6,
        w_latency: float = 0.4,
        max_carbon: float = 600.0,
        max_latency: float = 300.0,
    ) -> float:
        """
        Score a model/region candidate by regional carbon and network latency.
        Lower score = better (minimisation objective per Section 3.5.3).
        """
        zone = model_entry.get("grid_zone", "local")
        ci = region_carbon_map.get(zone, model_entry.get("region_carbon_multiplier", 1.0) * 400.0)
        net_latency = float(model_entry.get("network_latency_ms", 0.0))

        norm_ci = min(ci / max(max_carbon, 1.0), 1.0)
        norm_lat = min(net_latency / max(max_latency, 1.0), 1.0)
        return round(w_carbon * norm_ci + w_latency * norm_lat, 4)

    # ------------------------------------------------------------------
    # Registry Management
    # ------------------------------------------------------------------

    def register_model(self, model_entry: dict[str, Any]) -> None:
        """Add or update a model in the live registry."""
        with self._lock:
            model_id = model_entry.get("id")
            self._models = [m for m in self._models if m.get("id") != model_id]
            self._models.append(model_entry)
            self._save_unlocked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            available = [m for m in self._models if m.get("available")]
            moe_models = [m for m in self._models if m.get("moe")]
            regions = sorted({m.get("region", "local") for m in self._models})
            return {
                "version": self._metadata.get("version", "unversioned"),
                "total_models": len(self._models),
                "available_models": len(available),
                "moe_models": len(moe_models),
                "regions": regions,
                "refresh_policy": self._metadata.get("refresh_policy", "quarterly"),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.zoo_path.exists():
            logger.warning("Model Zoo file not found at %s; using empty registry", self.zoo_path)
            self._models = []
            self._metadata = {}
            return
        try:
            payload = json.loads(self.zoo_path.read_text(encoding="utf-8"))
            self._models = payload.get("models", [])
            self._metadata = {k: v for k, v in payload.items() if k != "models"}
            logger.info("Model Zoo loaded: %d models, version=%s", len(self._models), self._metadata.get("version"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load Model Zoo: %s", exc)
            self._models = []
            self._metadata = {}

    def _save_unlocked(self) -> None:
        try:
            payload = dict(self._metadata)
            payload["models"] = self._models
            self.zoo_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to save Model Zoo: %s", exc)

    def reload(self) -> None:
        """Reload from disk (e.g., after CI/CD model push)."""
        with self._lock:
            self._load()


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Module-level singleton
_zoo_instance: ModelZooService | None = None
_zoo_lock = threading.Lock()


def get_model_zoo() -> ModelZooService:
    global _zoo_instance
    if _zoo_instance is None:
        with _zoo_lock:
            if _zoo_instance is None:
                _zoo_instance = ModelZooService()
    return _zoo_instance
