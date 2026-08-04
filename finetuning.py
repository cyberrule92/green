"""Carbon-aware LoRA fine-tuning: turn collected feedback into a new routing rung.

Why this exists
---------------
Measurement showed the router losing to always-full on both carbon and quality,
because the ladder has no genuinely cheaper rung. ``model_onboarding`` attacks
that by importing better models. This module attacks it from the other side: it
makes a *small* model better at this deployment's actual traffic, so the cheap
rung becomes good enough to pick.

That is the sustainability argument, and it is worth stating precisely because it
is easy to get backwards. Fine-tuning **spends** carbon — a training run is the
largest single compute event this system performs. It only pays off if the
resulting adapter lets the router serve requests on a smaller model that it would
otherwise have escalated. So every job here is metered, and
:func:`model_onboarding.estimate_payback` converts that one-off into a
requests-to-break-even figure against the rung it replaces. A fine-tune that
never pays back is a fine-tune that should not have run.

Five decisions are load-bearing:

1. **LoRA / QLoRA only, never full fine-tuning.** A full fine-tune of even a 1.5B
   model does not fit a shared 21.5 GB MIG slice alongside live inference, and it
   would cost orders of magnitude more energy for a domain adaptation that a
   rank-16 adapter captures. Parameter-efficient tuning is not a compromise here;
   it is the only version of this that is defensible on carbon grounds.

2. **Training is deferrable, so it defers.** Unlike a chat request, nobody is
   waiting on a training job. That makes it the single best candidate in the whole
   system for carbon-aware scheduling: the job is queued and started in the
   lowest-carbon window in the 48-hour forecast. Chat's deferral is advisory;
   this one is real, like the coding agent's.

3. **Carbon is integrated over the run, not snapshotted.** A multi-hour job spans
   a changing grid. Sampling CI periodically and accumulating energy x CI per
   interval is the honest figure; multiplying total energy by the intensity at
   kick-off is not, and would flatter a job that deliberately started in a clean
   window.

4. **The adapter is not routable until measured.** An adapter that declared its
   own accuracy would be exactly the fiction ``model_onboarding`` refuses: the
   shipped zoo declares one model at 0.92 that measured 0.793. A finished job
   registers ``available: false`` and a caller must post measured figures.

5. **It runs in its own container.** The API image cannot train: its torch is
   built for CUDA 13 against a 12.8 driver and fails to initialise CUDA at all,
   and peft/trl/datasets are not installed. The trainer image is operator-supplied
   (``MODEL_FINETUNE_IMAGE``); unset, the capability reports itself unavailable
   rather than pretending.

Serving: vLLM 0.21 supports ``--enable-lora`` / ``--lora-modules``, verified on the
running container, so an adapter costs a few MB of VRAM on top of a base model
rather than a whole second set of weights. That is what makes a fine-tuned rung
nearly free to *keep*, even though it was not free to *make*.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from model_onboarding import (
    CONTAINER_PREFIX,
    DockerClient,
    DockerError,
    HostResources,
    ServingManager,
    estimate_payback,
    measurement_patch,
    probe_resources,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

# Below this a run cannot produce a meaningful adapter, only an overfit one. The
# floor is deliberately conservative: burning GPU-hours to memorise forty examples
# is the worst possible carbon trade, and the failure is silent -- the loss curve
# looks fine and the model gets worse at everything else.
MIN_TRAINING_SAMPLES = 200

# Rank 16 is the default because it is the smallest that reliably captures a
# domain shift on 1-3B models. Higher ranks cost linearly more memory and time
# for diminishing returns; the cap exists so a caller cannot quietly request a
# rank that turns a cheap job into an expensive one.
DEFAULT_LORA_RANK = 16
MAX_LORA_RANK = 64

# vLLM's --max-lora-rank must be >= the adapter's rank, so a served base must be
# started with at least this.
SUPPORTED_METHODS = ("lora", "qlora")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetStats:
    samples: int
    source: str
    rejected: int
    reasons: dict[str, int]
    mean_prompt_chars: float
    mean_response_chars: float

    @property
    def usable(self) -> bool:
        return self.samples >= MIN_TRAINING_SAMPLES

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "source": self.source,
            "rejected": self.rejected,
            "reasons": dict(self.reasons),
            "mean_prompt_chars": round(self.mean_prompt_chars, 1),
            "mean_response_chars": round(self.mean_response_chars, 1),
            "usable": self.usable,
            "min_required": MIN_TRAINING_SAMPLES,
        }


def build_dataset(records: Iterable[dict[str, Any]], *, source: str) -> tuple[list[dict[str, str]], DatasetStats]:
    """Validate feedback records into supervised (prompt, response) pairs.

    Records come from ``/api/feedback/export`` — ``(prompt, response, rating,
    model_variant)`` tuples. Only up-voted pairs are training signal: a
    down-voted response tells you what *not* to say, which supervised
    fine-tuning cannot use (that needs DPO and a preference pair, which this
    module does not do).

    Every rejection is counted and reported, because "we trained on 40 of your
    900 rows" is something the operator has to be able to see before the GPU
    spins up.
    """
    out: list[dict[str, str]] = []
    reasons: dict[str, int] = {}
    p_chars = r_chars = 0

    def reject(why: str) -> None:
        reasons[why] = reasons.get(why, 0) + 1

    seen: set[str] = set()
    total = 0
    for rec in records:
        total += 1
        if not isinstance(rec, dict):
            reject("not_an_object")
            continue
        rating = rec.get("rating")
        if rating is not None and int(rating) <= 0:
            reject("not_upvoted")
            continue
        prompt = str(rec.get("prompt") or "").strip()
        response = str(rec.get("response") or rec.get("content") or "").strip()
        if not prompt:
            reject("empty_prompt")
            continue
        if not response:
            reject("empty_response")
            continue
        # The rule-based degradation notice is not a model answer. Training on it
        # would teach the model to apologise for being unavailable.
        if "inference backend is currently starting up" in response:
            reject("degradation_placeholder")
            continue
        key = f"{prompt}\x00{response}"
        if key in seen:
            reject("duplicate")
            continue
        seen.add(key)
        p_chars += len(prompt)
        r_chars += len(response)
        out.append({"prompt": prompt, "response": response})

    n = len(out)
    return out, DatasetStats(
        samples=n,
        source=source,
        rejected=total - n,
        reasons=reasons,
        mean_prompt_chars=(p_chars / n) if n else 0.0,
        mean_response_chars=(r_chars / n) if n else 0.0,
    )


def write_dataset(rows: Sequence[dict[str, str]], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FineTunePlan:
    fits: bool
    method: str
    base_model_id: str
    base_repo_id: str
    lora_rank: int
    epochs: int
    samples: int
    est_vram_mb: float
    est_duration_s: float
    est_carbon_g: float
    duration_basis: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fits": self.fits,
            "method": self.method,
            "base_model_id": self.base_model_id,
            "base_repo_id": self.base_repo_id,
            "lora_rank": self.lora_rank,
            "epochs": self.epochs,
            "samples": self.samples,
            "est_vram_mb": round(self.est_vram_mb, 1),
            "est_duration_s": round(self.est_duration_s),
            "est_duration_h": round(self.est_duration_s / 3600.0, 2),
            "est_carbon_g": round(self.est_carbon_g, 3),
            "duration_basis": self.duration_basis,
            "reason": self.reason,
        }


def plan_finetune(
    *,
    base_entry: dict[str, Any],
    samples: int,
    resources: HostResources,
    method: str = "qlora",
    lora_rank: int = DEFAULT_LORA_RANK,
    epochs: int = 3,
    grid_ci: float = 400.0,
) -> FineTunePlan:
    """Size a LoRA job against the headroom that exists, and price it.

    VRAM for PEFT training is dominated by the frozen base weights plus
    activations; the adapter and its optimiser state are small by construction
    (that is the point of LoRA). QLoRA loads the base in 4-bit, which is what
    makes a 1.5B trainable next to live inference on this slice.

    ``est_duration_s`` is an **estimate**, labelled as one. Real throughput
    depends on sequence length, batch size and what else is on the GPU, so the
    figure exists to answer "hours or days?" — never to be recorded as a
    measurement. The metered value from the actual run is what gets stored.
    """
    params_b = float(base_entry.get("parameter_count_b") or 0.0)
    base_repo = str(base_entry.get("vllm_model_id") or base_entry.get("serve_repo_id") or "")
    reasons: list[str] = []

    if method not in SUPPORTED_METHODS:
        method = "qlora"
    rank = max(1, min(int(lora_rank), MAX_LORA_RANK))

    if params_b <= 0:
        return FineTunePlan(
            False, method, str(base_entry.get("id")), base_repo, rank, epochs, samples,
            0.0, 0.0, 0.0, "unknown",
            "base model has no parameter count in the zoo, so the job cannot be sized",
        )
    if not base_repo:
        return FineTunePlan(
            False, method, str(base_entry.get("id")), base_repo, rank, epochs, samples,
            0.0, 0.0, 0.0, "unknown",
            "base model has no Hugging Face repo id in the zoo, so there is nothing to load",
        )

    # Base weights: 4-bit under QLoRA, fp16 under plain LoRA.
    bytes_per_param = 0.6 if method == "qlora" else 2.0
    weights_mb = params_b * 1e9 * bytes_per_param / 1024**2
    # Adapter + its optimiser state. Rank-scaled and small: a rank-16 adapter on
    # a 1.5B is single-digit MB of weights, and Adam keeps two moments per
    # trainable parameter.
    adapter_mb = params_b * 1e9 * (rank / 4096.0) * 2 / 1024**2 * 3
    # Activations and gradients for the backward pass. The dominant term at short
    # sequence lengths, and the reason a "small" model still needs headroom.
    activation_mb = 1600.0 if method == "qlora" else 2600.0
    est_vram = weights_mb + adapter_mb + activation_mb

    if not resources.vram_measured:
        return FineTunePlan(
            False, method, str(base_entry.get("id")), base_repo, rank, epochs, samples,
            est_vram, 0.0, 0.0, "unknown",
            f"GPU memory could not be measured (basis: {resources.vram_basis}); refusing to "
            "start a multi-hour job without knowing what is resident on the slice",
        )
    if est_vram > resources.vram_budget_mb:
        return FineTunePlan(
            False, method, str(base_entry.get("id")), base_repo, rank, epochs, samples,
            est_vram, 0.0, 0.0, "unknown",
            f"needs ~{est_vram:.0f} MB VRAM, only {resources.vram_budget_mb:.0f} MB free. Training "
            "competes with live inference for this slice; stop a vLLM container or wait for one to "
            "be evicted before starting.",
        )

    # Throughput assumption, stated so it can be argued with: ~1400 tokens/s of
    # training throughput per billion parameters on this class of slice, at an
    # assumed 512 tokens per sample.
    tokens = samples * epochs * 512
    tokens_per_s = max(50.0, 1400.0 / max(params_b, 0.1))
    est_duration = tokens / tokens_per_s

    tdp = _env_float("GPU_TDP", 300.0)
    pue = _env_float("DATACENTER_PUE", 1.3)
    he = max(0.05, _env_float("FINETUNE_HARDWARE_EFFICIENCY", 0.75))
    energy_wh = tdp * (est_duration / 3600.0) * pue / he
    est_carbon = energy_wh / 1000.0 * grid_ci

    return FineTunePlan(
        True, method, str(base_entry.get("id")), base_repo, rank, epochs, samples,
        est_vram, est_duration, est_carbon, "estimated_from_throughput_assumption",
        f"{method} rank {rank}, {epochs} epochs over {samples} samples",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Carbon-aware scheduling
# ─────────────────────────────────────────────────────────────────────────────


def choose_window(
    find_window: Callable[..., dict[str, Any] | None] | None,
    *,
    duration_s: float,
    current_ci: float,
    defer_above_ci: float,
) -> dict[str, Any]:
    """Pick when to run. This is the feature's actual sustainability mechanism.

    Nobody is waiting on a training job, which makes it the one workload in this
    system that can move hours without hurting anyone. If the grid is clean
    enough right now we start immediately; otherwise we take the cleanest window
    the forecast offers and report the saving so the deferral is justified rather
    than asserted.
    """
    if current_ci <= defer_above_ci:
        return {
            "defer": False,
            "reason": f"grid is at {current_ci:.0f} gCO2/kWh, at or below the {defer_above_ci:.0f} threshold",
            "start_ci": current_ci,
        }
    window = None
    if find_window is not None:
        try:
            window = find_window(duration_hours=max(1.0, duration_s / 3600.0))
        except Exception as exc:  # noqa: BLE001 - forecast is best-effort
            logger.warning("low-carbon window lookup failed: %s", exc)
    if not window:
        return {
            "defer": False,
            "reason": (
                f"grid is at {current_ci:.0f} gCO2/kWh (above the {defer_above_ci:.0f} threshold) but no "
                "forecast window is available, so deferring would be an open-ended wait"
            ),
            "start_ci": current_ci,
        }
    start_ci = float(window.get("carbon_intensity") or window.get("ci") or current_ci)
    saving = (current_ci - start_ci) / current_ci if current_ci > 0 else 0.0
    return {
        "defer": True,
        "start_at": window.get("start") or window.get("start_iso"),
        "start_ci": start_ci,
        "current_ci": current_ci,
        "expected_saving_pct": round(100 * saving, 1),
        "reason": (
            f"grid is at {current_ci:.0f} gCO2/kWh; waiting for a {start_ci:.0f} gCO2/kWh window "
            f"cuts this job's emissions by about {100 * saving:.0f}%"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────

JOB_STATES = ("queued", "waiting_for_window", "training", "registering", "succeeded", "failed", "cancelled")


@dataclass
class FineTuneJob:
    job_id: str
    base_model_id: str
    adapter_id: str
    state: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    steps: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    schedule: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float = 0.0
    energy_wh: float = 0.0
    carbon_g: float = 0.0
    ci_samples: list[dict[str, float]] = field(default_factory=list)
    adapter_path: str | None = None

    def note(self, step: str, detail: str, **extra: Any) -> None:
        self.steps.append({"step": step, "detail": detail, "at": utc_now_iso(), **extra})
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "base_model_id": self.base_model_id,
            "adapter_id": self.adapter_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": list(self.steps),
            "plan": self.plan,
            "dataset": self.dataset,
            "schedule": self.schedule,
            "error": self.error,
            "duration_s": round(self.duration_s, 1),
            "duration_h": round(self.duration_s / 3600.0, 2),
            "energy_wh": round(self.energy_wh, 2),
            "training_carbon_g": round(self.carbon_g, 3),
            "carbon_basis": "measured_wallclock_x_spec_tdp_x_sampled_ci",
            "ci_sample_count": len(self.ci_samples),
            "adapter_path": self.adapter_path,
        }


class FineTuningService:
    """Dataset → plan → carbon-scheduled LoRA run → adapter → registered rung.

    Capabilities are injected rather than imported, matching ``workflows``' old
    posture and ``model_onboarding``'s: this module never imports
    ``decision_engine``.
    """

    def __init__(
        self,
        zoo: Any,
        *,
        feedback_records: Callable[[], list[dict[str, Any]]] | None = None,
        grid_ci: Callable[[], float] | None = None,
        find_low_carbon_window: Callable[..., dict[str, Any] | None] | None = None,
        system_metrics: Callable[[], dict[str, Any]] | None = None,
        docker: DockerClient | None = None,
        serving: ServingManager | None = None,
        state_path: str | Path | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.zoo = zoo
        self.feedback_records = feedback_records
        self.grid_ci = grid_ci or (lambda: _env_float("GRID_CARBON_FALLBACK", 475.0))
        self.find_low_carbon_window = find_low_carbon_window
        self.system_metrics = system_metrics
        self.docker = docker or DockerClient()
        self.serving = serving or ServingManager()
        self.state_path = Path(state_path or os.getenv("FINETUNE_STATE_PATH", "data/finetuning.json"))
        self.work_dir = Path(work_dir or os.getenv("FINETUNE_WORK_DIR", "data/finetune"))
        self.image = os.getenv("MODEL_FINETUNE_IMAGE", "").strip()
        self.command_template = os.getenv("MODEL_FINETUNE_COMMAND", "").strip()
        self.hf_cache_host_path = os.getenv("HF_CACHE_HOST_PATH", "/opt/green/data/hf-cache")
        self.work_dir_host = os.getenv("FINETUNE_WORK_DIR_HOST", "/opt/green/data/finetune")
        self._jobs: dict[str, FineTuneJob] = {}
        self._lock = threading.RLock()
        self._cancelled: set[str] = set()
        self._load_jobs()

    # -- capability ----------------------------------------------------------

    def capability(self) -> dict[str, Any]:
        resources = probe_resources(self.hf_cache_host_path, self.system_metrics)
        trainer = self._trainer_capability()
        data = self.dataset_preview()
        return {
            "enabled": _env_flag("FINETUNE_ENABLED", False),
            "trainer": trainer,
            "resources": resources.to_dict(),
            "dataset": data["stats"],
            "defer_above_ci": _env_float("FINETUNE_DEFER_CI", 300.0),
            "current_ci": self._current_ci(),
            "methods": list(SUPPORTED_METHODS),
            "max_lora_rank": MAX_LORA_RANK,
        }

    def _trainer_capability(self) -> dict[str, Any]:
        if not self.docker.available:
            return {
                "enabled": False,
                "reason": (
                    "docker socket not mounted. Training runs in its own container because the API "
                    "image cannot train: its torch targets CUDA 13 against a 12.8 driver and fails to "
                    "initialise CUDA, and peft/trl/datasets are not installed."
                ),
            }
        if not self.image:
            return {
                "enabled": False,
                "reason": (
                    "MODEL_FINETUNE_IMAGE is unset. Provide an image carrying a PEFT toolchain "
                    "(transformers + peft + trl + datasets + bitsandbytes) built against this host's "
                    "CUDA 12.8 driver."
                ),
            }
        if not self.command_template:
            return {"enabled": False, "reason": "MODEL_FINETUNE_COMMAND is unset; nothing to run in the image"}
        return {"enabled": True, "reason": "trainer image configured", "image": self.image}

    def _current_ci(self) -> float:
        try:
            return float(self.grid_ci() or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    # -- dataset -------------------------------------------------------------

    def dataset_preview(self, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        source = "supplied"
        if records is None:
            source = "feedback"
            try:
                records = self.feedback_records() if self.feedback_records else []
            except Exception as exc:  # noqa: BLE001
                logger.warning("feedback export unavailable: %s", exc)
                records = []
        rows, stats = build_dataset(records or [], source=source)
        return {"rows": rows, "stats": stats.to_dict()}

    # -- planning ------------------------------------------------------------

    def preview(
        self,
        base_model_id: str,
        *,
        method: str = "qlora",
        lora_rank: int = DEFAULT_LORA_RANK,
        epochs: int = 3,
        records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        base = self.zoo.get_model(base_model_id)
        if not base:
            raise ValueError(f"base model {base_model_id!r} is not in the zoo")
        data = self.dataset_preview(records)
        resources = probe_resources(self.hf_cache_host_path, self.system_metrics)
        ci = self._current_ci()
        plan = plan_finetune(
            base_entry=base,
            samples=data["stats"]["samples"],
            resources=resources,
            method=method,
            lora_rank=lora_rank,
            epochs=epochs,
            grid_ci=ci,
        )
        schedule = choose_window(
            self.find_low_carbon_window,
            duration_s=plan.est_duration_s,
            current_ci=ci,
            defer_above_ci=_env_float("FINETUNE_DEFER_CI", 300.0),
        )
        return {
            "base_model": {
                "id": base.get("id"),
                "repo": base.get("vllm_model_id"),
                "parameter_count_b": base.get("parameter_count_b"),
                "accuracy_baseline": base.get("accuracy_baseline"),
            },
            "dataset": data["stats"],
            "resources": resources.to_dict(),
            "plan": plan.to_dict(),
            "schedule": schedule,
            "trainer": self._trainer_capability(),
        }

    # -- jobs ----------------------------------------------------------------

    def get_job(self, job_id: str) -> FineTuneJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"job {job_id} not found")
        if job.state in {"succeeded", "failed", "cancelled"}:
            return {"ok": False, "detail": f"job is already {job.state}"}
        self._cancelled.add(job_id)
        try:
            self.docker.stop(f"{CONTAINER_PREFIX}train-{job_id}")
        except DockerError:
            pass
        job.state = "cancelled"
        job.note("cancelled", "cancelled by operator")
        with self._lock:
            self._save_jobs()
        return {"ok": True, "detail": "cancelled"}

    def _load_jobs(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not load fine-tuning state: %s", exc)
            return
        for raw in payload.get("jobs", []):
            try:
                job = FineTuneJob(
                    job_id=raw["job_id"],
                    base_model_id=raw["base_model_id"],
                    adapter_id=raw["adapter_id"],
                    state=raw.get("state", "failed"),
                    created_at=raw.get("created_at", utc_now_iso()),
                    updated_at=raw.get("updated_at", utc_now_iso()),
                    steps=raw.get("steps", []),
                    plan=raw.get("plan"),
                    dataset=raw.get("dataset"),
                    schedule=raw.get("schedule"),
                    error=raw.get("error"),
                    duration_s=float(raw.get("duration_s") or 0.0),
                    energy_wh=float(raw.get("energy_wh") or 0.0),
                    carbon_g=float(raw.get("training_carbon_g") or 0.0),
                    adapter_path=raw.get("adapter_path"),
                )
            except (KeyError, TypeError, ValueError):
                continue
            # The thread died with the process. A job left looking like it is
            # still training would have an operator waiting on nothing.
            if job.state not in {"succeeded", "failed", "cancelled"}:
                job.state = "failed"
                job.error = "interrupted by an API restart"
            self._jobs[job.job_id] = job

    def _save_jobs(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:200]
            self.state_path.write_text(json.dumps({"jobs": [j.to_dict() for j in jobs]}, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not persist fine-tuning state: %s", exc)

    # -- submission ----------------------------------------------------------

    def submit(
        self,
        base_model_id: str,
        *,
        method: str = "qlora",
        lora_rank: int = DEFAULT_LORA_RANK,
        epochs: int = 3,
        records: list[dict[str, Any]] | None = None,
        adapter_id: str | None = None,
        force_now: bool = False,
    ) -> FineTuneJob:
        if not _env_flag("FINETUNE_ENABLED", False):
            raise PermissionError("fine-tuning is disabled; set FINETUNE_ENABLED=true to enable it")
        trainer = self._trainer_capability()
        if not trainer["enabled"]:
            raise RuntimeError(trainer["reason"])

        preview = self.preview(base_model_id, method=method, lora_rank=lora_rank, epochs=epochs, records=records)
        if not preview["dataset"]["usable"]:
            raise ValueError(
                f"only {preview['dataset']['samples']} usable training samples; {MIN_TRAINING_SAMPLES} is the "
                "floor. Below it a run burns GPU-hours to overfit, which is the worst carbon trade available."
            )
        if not preview["plan"]["fits"]:
            raise ValueError(preview["plan"]["reason"])

        aid = adapter_id or f"ft-{re.sub(r'[^a-z0-9]+', '-', base_model_id.lower()).strip('-')}-{uuid.uuid4().hex[:6]}"
        job = FineTuneJob(job_id=uuid.uuid4().hex[:12], base_model_id=base_model_id, adapter_id=aid)
        job.plan = preview["plan"]
        job.dataset = preview["dataset"]
        job.schedule = dict(preview["schedule"])
        if force_now:
            job.schedule = {"defer": False, "reason": "operator requested an immediate start",
                            "start_ci": preview["schedule"].get("current_ci") or self._current_ci()}
        with self._lock:
            self._jobs[job.job_id] = job
            self._save_jobs()
        threading.Thread(
            target=self._run_job, args=(job,), kwargs={"records": records}, daemon=True, name=f"finetune-{job.job_id}"
        ).start()
        return job

    def _run_job(self, job: FineTuneJob, *, records: list[dict[str, Any]] | None) -> None:
        try:
            # 1. Carbon-aware wait. This is the deferral that actually matters:
            #    nobody is blocked on a training job, so moving it hours is free
            #    in every sense except calendar time.
            if (job.schedule or {}).get("defer"):
                job.state = "waiting_for_window"
                job.note("schedule", job.schedule.get("reason", "waiting for a cleaner window"),
                         start_at=job.schedule.get("start_at"))
                if not self._wait_for_window(job):
                    job.state = "cancelled"
                    job.note("cancelled", "cancelled while waiting for a low-carbon window")
                    return

            # 2. Materialise the dataset next to the trainer's mount.
            data = self.dataset_preview(records)
            rows = data["rows"]
            ds_path = self.work_dir / job.job_id / "train.jsonl"
            write_dataset(rows, ds_path)
            job.note("dataset", f"wrote {len(rows)} supervised pairs", path=str(ds_path))

            # 3. Train, metered.
            job.state = "training"
            ok, detail = self._run_trainer(job, ds_path)
            job.note("train", detail, duration_s=round(job.duration_s, 1),
                     training_carbon_g=round(job.carbon_g, 3))
            if not ok:
                raise RuntimeError(detail)

            # 4. Register as a candidate the router may not yet select.
            job.state = "registering"
            entry = self.register_adapter(job)
            job.note("register",
                     f"registered {job.adapter_id} as unavailable — not routable until measured",
                     zoo_id=entry["id"])
            job.state = "succeeded"
        except Exception as exc:  # noqa: BLE001 - job boundary
            job.state = "failed"
            job.error = str(exc)
            job.note("failed", str(exc))
            logger.warning("fine-tune job %s failed: %s", job.job_id, exc)
        finally:
            job.updated_at = utc_now_iso()
            with self._lock:
                self._save_jobs()

    def _wait_for_window(self, job: FineTuneJob) -> bool:
        """Sleep until the chosen window, re-checking the grid as we go.

        Polls rather than sleeping blind, because a forecast is a forecast: if
        the grid cleans up early we start early, and if the job is cancelled we
        notice within a poll interval instead of hours later.
        """
        deadline = time.monotonic() + _env_float("FINETUNE_MAX_WAIT_S", 48 * 3600)
        poll = _env_float("FINETUNE_WINDOW_POLL_S", 300.0)
        threshold = _env_float("FINETUNE_DEFER_CI", 300.0)
        while time.monotonic() < deadline:
            if job.job_id in self._cancelled:
                return False
            ci = self._current_ci()
            if ci and ci <= threshold:
                job.note("schedule", f"grid reached {ci:.0f} gCO2/kWh; starting now", ci=ci)
                return True
            time.sleep(poll)
        job.note("schedule", "max wait elapsed; starting anyway rather than deferring indefinitely")
        return True

    def _run_trainer(self, job: FineTuneJob, dataset_path: Path) -> tuple[bool, str]:
        """Run the trainer container, sampling grid intensity throughout.

        Energy is spec TDP x measured wall-clock / hardware efficiency x PUE —
        the same form as ``model_zoo.compute_operational_carbon``, so a training
        gram and an inference gram mean the same thing. Carbon accumulates per
        sampling interval against the CI *at that moment*: a job that waited for
        a clean window should be credited for the clean hours it actually ran in,
        and a job whose window turned dirty should be charged for it.
        """
        name = f"{CONTAINER_PREFIX}train-{job.job_id}"
        out_dir = f"{self.work_dir_host}/{job.job_id}/adapter"
        rel_ds = f"{self.work_dir_host}/{job.job_id}/train.jsonl"
        plan = job.plan or {}
        cmd = self.command_template.format(
            base_repo=plan.get("base_repo_id", ""),
            dataset=rel_ds,
            out_dir=out_dir,
            method=plan.get("method", "qlora"),
            lora_rank=plan.get("lora_rank", DEFAULT_LORA_RANK),
            epochs=plan.get("epochs", 3),
        )
        env = ["HF_HOME=/root/.cache/huggingface"]
        token = os.getenv("HF_TOKEN", "").strip()
        if token:
            env.append(f"HF_TOKEN={token}")
        config = {
            "Image": self.image,
            "Cmd": ["sh", "-lc", cmd],
            "Env": env,
            "Labels": {"green.onboarded": "1", "green.role": "finetune", "green.job_id": job.job_id},
            "HostConfig": {
                "Binds": [
                    f"{self.hf_cache_host_path}:/root/.cache/huggingface",
                    f"{self.work_dir_host}:{self.work_dir_host}",
                ],
                "ShmSize": 4 * 1024**3,
                "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
            },
        }
        try:
            self.docker.create(name, config)
            self.docker.start(name)
        except DockerError as exc:
            return False, f"trainer container failed to start: {exc}"

        tdp = _env_float("GPU_TDP", 300.0)
        pue = _env_float("DATACENTER_PUE", 1.3)
        he = max(0.05, _env_float("FINETUNE_HARDWARE_EFFICIENCY", 0.75))
        sample_s = _env_float("FINETUNE_CI_SAMPLE_S", 300.0)
        timeout_s = _env_float("FINETUNE_TIMEOUT_S", 24 * 3600)

        started = time.monotonic()
        last = started
        exit_code: int | None = None
        while time.monotonic() - started < timeout_s:
            if job.job_id in self._cancelled:
                try:
                    self.docker.stop(name)
                except DockerError:
                    pass
                return False, "cancelled by operator"
            try:
                state = self.docker.inspect(name).get("State") or {}
            except DockerError as exc:
                return False, f"trainer container vanished: {exc}"
            now = time.monotonic()
            interval = now - last
            if interval >= sample_s:
                ci = self._current_ci()
                wh = tdp * (interval / 3600.0) * pue / he
                job.energy_wh += wh
                job.carbon_g += wh / 1000.0 * ci
                job.ci_samples.append({"at_s": round(now - started, 1), "ci": round(ci, 1)})
                last = now
            if str(state.get("Status") or "").lower() in {"exited", "dead"}:
                exit_code = state.get("ExitCode")
                break
            time.sleep(min(30.0, sample_s))

        # Charge the unsampled tail at the latest intensity.
        tail = time.monotonic() - last
        if tail > 0:
            ci = self._current_ci()
            wh = tdp * (tail / 3600.0) * pue / he
            job.energy_wh += wh
            job.carbon_g += wh / 1000.0 * ci
        job.duration_s = time.monotonic() - started
        logs = self.docker.logs_tail(name, 60)
        try:
            self.docker.remove(name, force=True)
        except DockerError:
            pass

        if exit_code is None:
            return False, f"training timed out after {timeout_s / 3600:.1f} h"
        if exit_code != 0:
            return False, f"training exited {exit_code}: {logs[-400:]}"
        job.adapter_path = out_dir
        return True, f"training completed in {job.duration_s / 3600:.2f} h for {job.carbon_g:.1f} gCO2e"

    # -- registration --------------------------------------------------------

    def register_adapter(self, job: FineTuneJob) -> dict[str, Any]:
        """Register the adapter as a zoo candidate the router may not yet select.

        The adapter inherits the base model's device and capability fields — it
        *is* the base model plus a small delta — but its accuracy is explicitly
        unmeasured. The whole point of fine-tuning is that quality changed; the
        direction and size of that change is exactly what has not been
        established yet.
        """
        base = self.zoo.get_model(job.base_model_id)
        if not base:
            raise ValueError(f"base model {job.base_model_id!r} disappeared from the zoo")
        plan = job.plan or {}
        entry = dict(base)
        entry.update({
            "id": job.adapter_id,
            "model_id": f"{base.get('model_id')}+lora",
            "model_variant": job.adapter_id,
            "available": False,
            "accuracy_baseline": 0.0,
            "accuracy_basis": "unmeasured",
            "latency_ms_p50": base.get("latency_ms_p50", 0),
            "latency_basis": "inherited_from_base_unmeasured",
            "lora_adapter": True,
            "lora_adapter_path": job.adapter_path,
            "lora_rank": plan.get("lora_rank"),
            "base_model_id": job.base_model_id,
            "finetuned_at": utc_now_iso(),
            "finetune_job_id": job.job_id,
            "training_carbon_g": round(job.carbon_g, 3),
            "training_duration_s": round(job.duration_s, 1),
            "training_samples": (job.dataset or {}).get("samples"),
            "training_carbon_basis": "measured_wallclock_x_spec_tdp_x_sampled_ci",
            "vllm_endpoint_env": f"VLLM_LORA_{re.sub(r'[^A-Za-z0-9]+', '_', job.adapter_id).upper()}_URL",
            "vllm_model_id": job.adapter_id,
            "onboarded": True,
        })
        for stale in ("expected_output_tokens", "expected_output_tokens_basis", "measured_at",
                      "measured_samples", "vllm_endpoint_url", "served_container"):
            entry.pop(stale, None)
        self.zoo.register_model(entry)
        return entry

    def apply_measurement(self, adapter_id: str, measurement: dict[str, Any], *, basis: str) -> dict[str, Any]:
        """Promote a measured adapter to routable, with its carbon payback.

        Payback is the question fine-tuning has to answer: the run spent real
        grams, and it is only justified if the adapter lets the router serve on a
        cheaper rung than it otherwise would. It is computed here rather than
        promised up front, because it needs both rungs measured.
        """
        entry = self.zoo.get_model(adapter_id)
        if not entry:
            raise ValueError(f"adapter {adapter_id!r} is not registered")
        if not entry.get("lora_adapter"):
            raise ValueError(f"{adapter_id!r} is not a fine-tuned adapter")
        if not entry.get("vllm_endpoint_url"):
            raise ValueError(
                f"{adapter_id!r} has no live endpoint. Serve it before measuring, or the figures "
                "describe nothing."
            )
        patched = dict(entry)
        patched.update(measurement_patch(measurement, basis))
        payback = estimate_payback(
            float(entry.get("training_carbon_g") or 0.0),
            measurement.get("carbon_g_per_request"),
            measurement.get("incumbent_carbon_g_per_request"),
        )
        if payback is not None:
            patched["training_payback"] = payback
        self.zoo.register_model(patched)
        return patched

    # -- serving -------------------------------------------------------------

    def serve_adapter(self, adapter_id: str, *, ready_timeout_s: float = 900.0) -> dict[str, Any]:
        """Serve base + adapter in one vLLM container via --enable-lora.

        This is what makes a fine-tuned rung cheap to *keep*: the adapter is a few
        MB on top of a base model that is already loaded, rather than a second set
        of weights. Verified against this deployment's vLLM 0.21, which supports
        --enable-lora / --lora-modules / --max-lora-rank.
        """
        entry = self.zoo.get_model(adapter_id)
        if not entry or not entry.get("lora_adapter"):
            raise ValueError(f"{adapter_id!r} is not a registered fine-tuned adapter")
        base = self.zoo.get_model(entry.get("base_model_id") or "")
        if not base:
            raise ValueError("the adapter's base model is no longer in the zoo")
        base_repo = base.get("vllm_model_id")
        path = entry.get("lora_adapter_path")
        if not base_repo or not path:
            raise ValueError("adapter is missing its base repo or adapter path")

        resources = probe_resources(self.hf_cache_host_path, self.system_metrics)
        if not resources.vram_measured:
            return {"ok": False, "detail": f"GPU memory unmeasured (basis: {resources.vram_basis}); refusing to serve"}

        name = f"{CONTAINER_PREFIX}lora-{re.sub(r'[^a-z0-9]+', '-', adapter_id.lower()).strip('-')}"
        port = _env_int("MODEL_SERVE_PORT_BASE", 8010) + 40
        rank = int(entry.get("lora_rank") or DEFAULT_LORA_RANK)
        util = min(0.95, max(0.05, (float(entry.get("est_vram_mb") or 4000.0) * 1.12) / max(resources.vram_total_mb, 1.0)))
        cmd = [
            f"--model={base_repo}", "--host=0.0.0.0", f"--port={port}",
            f"--served-model-name={base_repo}",
            "--enable-lora", f"--lora-modules={adapter_id}={path}", f"--max-lora-rank={rank}",
            f"--gpu-memory-utilization={util:.3f}",
            f"--max-model-len={int(entry.get('max_model_len') or 2048)}",
        ]
        config = {
            "Image": self.serving.image,
            "Cmd": cmd,
            "Env": [f"HF_HOME=/root/.cache/huggingface"] + (
                [f"HF_TOKEN={os.getenv('HF_TOKEN', '').strip()}"] if os.getenv("HF_TOKEN", "").strip() else []
            ),
            "Labels": {"green.onboarded": "1", "green.role": "lora", "green.adapter_id": adapter_id},
            "HostConfig": {
                "Binds": [
                    f"{self.hf_cache_host_path}:/root/.cache/huggingface",
                    f"{self.work_dir_host}:{self.work_dir_host}",
                ],
                "NetworkMode": self.serving.network,
                "ShmSize": 4 * 1024**3,
                "RestartPolicy": {"Name": "unless-stopped"},
                "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
            },
        }
        try:
            try:
                if self.docker.inspect(name):
                    self.docker.remove(name, force=True)
            except DockerError:
                pass
            self.docker.create(name, config)
            self.docker.start(name)
        except DockerError as exc:
            return {"ok": False, "detail": f"container start failed: {exc}"}

        # Readiness is a real generation against the adapter, not /health --
        # /health returns 200 from a wedged engine (see model_onboarding).
        ok, detail = self.serving._wait_ready(name, port, ready_timeout_s, 5.0, model_name=adapter_id)
        if not ok:
            return {"ok": False, "detail": detail, "logs": self.docker.logs_tail(name, 60)}
        url = f"http://{name}:{port}/v1"
        os.environ[str(entry["vllm_endpoint_env"])] = url
        patched = dict(entry)
        patched["vllm_endpoint_url"] = url
        patched["served_container"] = name
        self.zoo.register_model(patched)
        return {"ok": True, "detail": detail, "url": url}
