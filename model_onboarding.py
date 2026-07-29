"""Carbon-aware model onboarding: browse Hugging Face, size a quantization plan
against real headroom, download, serve, and register.

Why this module exists
----------------------
A three-arm routing comparison (1350 requests, frozen router) returned a
negative result: CSS lost to
always-full on *both* carbon and quality. The cause was not the routing policy —
it was the menu. Four zoo candidates (``ultra-light`` / ``small`` / ``medium`` /
``local-cpu-fallback``) all dispatch to the same vllm-medium container serving
TinyLlama-1.1B, with hand-written differing TDPs (70/95/145 W) and accuracies
(0.60/0.66/0.81). Worse, TinyLlama measured *more* verbose than Qwen2.5-1.5B
(108 vs 94 output tokens), so the "cheap" rung burned more carbon for worse
answers. A router can only be as good as the candidates it ranks, and those rungs
are synthetic.

This module builds real rungs. Quantization is the best rung-generator available:
same model family, genuinely lower VRAM and power draw, and a quality delta that
can be *measured* rather than declared.

Three rules are load-bearing, and each exists because of something that
measurement showed:

1. **Measurement gates availability, not declaration.** Every zoo entry carries
   ``accuracy_baseline`` and ``latency_ms_p50``, and CSS ranks on them. The
   shipped zoo declares ``local-vgpu-full`` at 0.92; it measured 0.793 in
   practice. If an onboarded model could self-declare accuracy from an HF
   model card, this module would be a machine for injecting fiction into the
   router. So a newly registered model lands ``available: false`` with
   ``accuracy_basis: "unmeasured"`` and CSS cannot select it. Only
   :meth:`ModelOnboardingService.apply_measurement` flips it live, and it stamps
   the basis so the provenance survives. Producing those figures is the caller's
   job: post them to ``/api/models/{id}/measure`` from whatever evaluation you
   trust, against a live endpoint.

2. **VRAM is the binding constraint, not disk.** The deployment target is a 24 GB
   H100L vGPU *slice* already hosting several vLLM containers. Sizing must happen
   against free VRAM at serve time, and admitting a model may mean evicting one.
   Disk matters too (the box runs hot) but is checked as a simple preflight.

3. **Prefer a pre-quantized checkpoint over quantizing locally.** An AWQ/GPTQ
   calibration pass is the most carbon-expensive operation this system can
   perform, and it competes for the same slice that is serving live traffic. HF
   already hosts thousands of pre-quantized repos. So the planner searches for one
   first, falls back to vLLM's in-flight bitsandbytes path (which needs no
   calibration at all), and only runs a local pass when explicitly asked. When it
   does, the pass runs in its own container and its cost is metered from measured
   wall-clock, recorded on the zoo entry as ``quantization_carbon_g``. Once the
   rung is measured, :func:`estimate_payback` turns that one-off into a
   requests-to-break-even figure against the rung it would replace — a quantizer
   that hides its own footprint has no business in a carbon system.

Dependency posture matches ``workflows.py`` and ``coding_agent.py``: no import of
``decision_engine`` (that would cycle), no new third-party dependency. The Docker
control plane is raw HTTP over the unix socket via stdlib ``http.client``, and
grid carbon intensity is injected as a callable rather than imported.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"

# Bytes per parameter by weight format, derived arithmetically — not measured.
# 4-bit group quantization at the conventional group_size=128 stores, per group:
# 128 x 4 bits of packed weight (64 B) + one fp16 scale (2 B) + one packed 4-bit
# zero-point (0.5 B) = 66.5 B, i.e. 0.52 B/param. The figures below round up from
# that to 0.56/0.60 to cover the layers such checkpoints conventionally leave in
# fp16 (embeddings and lm_head), which are a larger share on small models. These
# are sizing inputs for an admission decision, so erring high is the safe error.
QUANT_BYTES_PER_PARAM: dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "awq": 0.56,
    "gptq": 0.56,
    "compressed-tensors": 0.56,
    "bitsandbytes": 0.60,
}

# Quantization formats this pipeline will select, mapped to the --quantization
# value. Deliberately a subset of what vLLM supports: the planner only picks a
# format it can also *size*, and the set is intersected at runtime with the
# serving image's own QUANTIZATION_METHODS (see ServingManager.supported_quant).
VLLM_QUANT_ARG: dict[str, str] = {
    "awq": "awq",
    "gptq": "gptq",
    "fp8": "fp8",
    "compressed-tensors": "compressed-tensors",
    "bitsandbytes": "bitsandbytes",
}

# Repo-name patterns that mark an already-quantized checkpoint.
_QUANT_REPO_RE = re.compile(r"(?:^|[-_.])(awq|gptq|int4|int8|4bit|8bit|w4a16|gguf)(?:$|[-_.])", re.I)
_PARAM_IN_NAME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:$|[-_.])")

# Headroom for CUDA graphs, activations and allocator slack, on top of weights
# and KV cache. A conservative constant, not a measurement: vLLM's true non-KV
# overhead varies with batch shape and attention backend, and under-reserving
# fails at warm-up rather than at load. Override with MODEL_ONBOARD_OVERHEAD_MB
# once a given deployment has its own numbers.
VLLM_RUNTIME_OVERHEAD_MB = 1024.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
# Host resource probing
#
# "Quantize to fit the system's resources" is only meaningful against a number
# that reflects what is actually free *right now*, on a slice that is already
# serving. nvidia-smi reports the whole slice, so free VRAM is total minus what
# the resident containers hold — not a static config figure.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HostResources:
    gpu_name: str | None
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    vram_reserve_mb: float
    disk_free_gb: float
    disk_total_gb: float
    disk_reserve_gb: float
    vram_basis: str = "gpu"
    mig_mode: str | None = None
    probed_at: str = field(default_factory=utc_now_iso)

    # Bases that reflect an actual reading of *used* VRAM. Anything else means we
    # know the card's capacity but not what is currently resident on it, which is
    # not enough to admit a model — see `vram_measured`.
    MEASURED_BASES = ("mig_instance", "gpu", "gpu_board_mig_unparsed", "metrics_sidecar")

    @property
    def vram_measured(self) -> bool:
        """True when free VRAM is a reading rather than an assumption.

        This matters more than it looks. The API container has no `nvidia-smi`
        and the metrics sidecar ships with `DISABLE_GPU_METRICS=1`, so on a
        default deployment there is no in-container source of GPU memory at all.
        Sizing against a config-declared *total* while assuming the card is empty
        would admit a model onto a slice that is already full — which is exactly
        how every engine on this host wedged at once. So an unmeasured basis
        fails closed.
        """
        return self.vram_basis in self.MEASURED_BASES and self.vram_total_mb > 0

    @property
    def vram_budget_mb(self) -> float:
        """Free VRAM minus the safety reserve — what a new model may claim."""
        if not self.vram_measured:
            return 0.0
        return max(0.0, self.vram_free_mb - self.vram_reserve_mb)

    @property
    def disk_budget_gb(self) -> float:
        return max(0.0, self.disk_free_gb - self.disk_reserve_gb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "vram_total_mb": round(self.vram_total_mb, 1),
            "vram_used_mb": round(self.vram_used_mb, 1),
            "vram_free_mb": round(self.vram_free_mb, 1),
            "vram_basis": self.vram_basis,
            "vram_measured": self.vram_measured,
            "mig_mode": self.mig_mode,
            "vram_reserve_mb": round(self.vram_reserve_mb, 1),
            "vram_budget_mb": round(self.vram_budget_mb, 1),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "disk_reserve_gb": round(self.disk_reserve_gb, 2),
            "disk_budget_gb": round(self.disk_budget_gb, 2),
            "probed_at": self.probed_at,
        }


def _nvidia_smi(args: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi unavailable: %s", exc)
        return None
    if out.returncode != 0:
        return None
    return out.stdout


# The MIG device table row carries "<used>MiB / <total>MiB" for the instance.
_MIG_MEM_RE = re.compile(r"(\d+)MiB\s*/\s*(\d+)MiB")


def _probe_mig_memory() -> tuple[float, float] | None:
    """Parse the MIG instance's own capacity out of nvidia-smi's table.

    Necessary because on a MIG-partitioned board ``--query-gpu=memory.total``
    reports the *whole board* — 24576 MiB on this host — while the instance a
    container can actually use is 21547 MiB. Sizing an admission decision
    against the board figure over-promises by ~3 GB, and over-promising VRAM is
    how you wedge every engine on the slice at once.

    Querying ``--id=<MIG-UUID>`` would be cleaner but returns "No devices were
    found" on this driver (570.172.08), so the text table is the only source
    that works here. Returns None when it cannot be parsed, and the caller
    degrades to the board figure with the basis recorded rather than silently
    substituting a number that means something different.
    """
    text = _nvidia_smi([])
    if not text or "MIG devices:" not in text:
        return None
    section = text.split("MIG devices:", 1)[1]
    for line in section.splitlines():
        match = _MIG_MEM_RE.search(line)
        if match:
            used, total = float(match.group(1)), float(match.group(2))
            # The BAR1 row matches the same shape; a real instance has non-zero
            # capacity well above BAR1's typical 4 GB aperture, and BAR1 always
            # follows the memory row, so taking the first match is correct.
            if total > 0:
                return total, used
    return None


def probe_gpu() -> tuple[str | None, float, float, float, str, str | None]:
    """Return ``(name, total_mb, used_mb, free_mb, basis, mig_mode)``.

    ``free`` is returned separately rather than derived as ``total - used``
    because on a MIG board those disagree: nvidia-smi reports the *board* total
    (24576 MiB here) alongside a MIG-aware ``memory.free`` (6660 MiB), so the
    subtraction overstates free by ~3 GB. Free is the number admission depends
    on, so it is read, not computed.

    Returns zeros when there is no GPU or nvidia-smi is absent, which keeps this
    importable on a CPU-only box (the test host) instead of raising at import.
    """
    csv = _nvidia_smi(
        [
            "--query-gpu=name,memory.total,memory.used,memory.free,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    )
    if not csv or not csv.strip():
        return None, 0.0, 0.0, 0.0, "unavailable", None
    parts = [p.strip() for p in csv.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return None, 0.0, 0.0, 0.0, "unavailable", None
    name = parts[0] or None
    mig_mode = parts[4] if len(parts) > 4 else None
    try:
        total, used, free = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        # "[Insufficient Permissions]" lands here: a MIG board denies memory
        # queries to unprivileged containers, and a partial reading is worse
        # than none because it would be sized against.
        return name, 0.0, 0.0, 0.0, "unavailable", mig_mode

    if (mig_mode or "").strip().lower() == "enabled":
        mig = _probe_mig_memory()
        if mig:
            # The MIG table gives the instance's own capacity and occupancy;
            # free follows from those two consistently.
            return name, mig[0], mig[1], max(0.0, mig[0] - mig[1]), "mig_instance", mig_mode
        # Board total, but memory.free is still MIG-aware, so keep it as read.
        return name, total, used, free, "gpu_board_mig_unparsed", mig_mode
    return name, total, used, free, "gpu", mig_mode


def probe_resources(
    cache_dir: str | Path | None = None,
    system_metrics: Callable[[], dict[str, Any]] | None = None,
) -> HostResources:
    """Probe headroom, preferring the most direct source of truth available.

    Source order, because no single one works everywhere in this deployment:

    1. ``nvidia-smi`` in-process. Correct and MIG-aware, but the API container
       does not ship it — this is the path that works on the host and in the
       vLLM containers.
    2. The metrics sidecar's ``system_gpu_*`` fields, which is the seam the rest
       of the system already uses for GPU telemetry. Note it ships with
       ``DISABLE_GPU_METRICS=1``, so it reports zeros until enabled.
    3. Nothing. Recorded as ``unavailable``, and :meth:`HostResources.vram_measured`
       then makes the planner refuse rather than guess.

    ``GPU_VRAM_GB`` is deliberately *not* used as a fallback. It gives capacity,
    not occupancy, and admitting a model against capacity alone would ignore the
    containers already resident on the slice.
    """
    name, total_mb, used_mb, free_mb, basis, mig_mode = probe_gpu()

    if basis == "unavailable" and system_metrics is not None:
        try:
            metrics = system_metrics() or {}
            total = float(metrics.get("system_gpu_TotalMemory") or 0.0)
            free = float(metrics.get("system_gpu_MemoryFree") or 0.0)
            used = float(metrics.get("system_gpu_UsedMemory") or 0.0)
            if total > 0 and free > 0:
                # Take free as reported. On this MIG board `total - used` would
                # say 9688 MB where thetrue figure is 6660 — the sidecar's
                # memory.free is MIG-aware and the board total is not.
                total_mb, used_mb, free_mb = total, used or max(0.0, total - free), free
                basis = "metrics_sidecar"
        except Exception as exc:  # noqa: BLE001 - telemetry must not break planning
            logger.debug("metrics sidecar unusable for VRAM probing: %s", exc)

    path = Path(cache_dir or os.getenv("HF_HOME", "/app/data/hf-cache"))
    probe_path = path if path.exists() else Path("/")
    usage = shutil.disk_usage(probe_path)
    return HostResources(
        gpu_name=name,
        vram_total_mb=total_mb,
        vram_used_mb=used_mb,
        vram_free_mb=free_mb,
        vram_reserve_mb=_env_float("MODEL_ONBOARD_VRAM_RESERVE_MB", 1024.0),
        disk_free_gb=usage.free / 1024**3,
        disk_total_gb=usage.total / 1024**3,
        disk_reserve_gb=_env_float("MODEL_ONBOARD_DISK_RESERVE_GB", 20.0),
        vram_basis=basis,
        mig_mode=mig_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face catalog
# ─────────────────────────────────────────────────────────────────────────────


class HFError(RuntimeError):
    """Hugging Face Hub API returned an error or could not be reached."""


class HFCatalog:
    """Read-only browse/search against the HF Hub API.

    The token is optional for public repos and required for gated ones. It is
    passed in rather than read from the environment so ``decision_engine`` can
    hand over a value decrypted from the workflow secret box instead of keeping
    it in plaintext config.
    """

    def __init__(self, token: str | None = None, timeout_s: float = 20.0) -> None:
        self.token = (token or "").strip() or None
        self.timeout_s = timeout_s

    # -- transport -----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "adaptive-green-ai/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        import requests  # already a project dependency; imported lazily to keep this module light

        try:
            resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - network failure surface
            raise HFError(f"could not reach Hugging Face: {exc}") from exc
        if resp.status_code == 401:
            raise HFError("Hugging Face rejected the token (401). Set a valid HF token.")
        if resp.status_code == 403:
            raise HFError("access denied (403) — the repo is gated and the token lacks access.")
        if resp.status_code == 404:
            raise HFError("not found (404) on Hugging Face.")
        if resp.status_code >= 400:
            raise HFError(f"Hugging Face API error {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise HFError("Hugging Face returned a non-JSON response") from exc

    # -- queries -------------------------------------------------------------

    def search(
        self,
        query: str = "",
        *,
        limit: int = 25,
        task: str | None = "text-generation",
        sort: str = "downloads",
        author: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the hub. Returns summaries shaped for the browse UI."""
        params: dict[str, Any] = {
            "limit": max(1, min(int(limit), 100)),
            "sort": sort,
            "direction": -1,
            "full": "true",
        }
        if query:
            params["search"] = query
        if task:
            params["filter"] = task
        if author:
            params["author"] = author
        payload = self._get(f"{HF_API}/models", params)
        if not isinstance(payload, list):
            return []
        return [self._summarize(item) for item in payload]

    def detail(self, repo_id: str) -> dict[str, Any]:
        payload = self._get(f"{HF_API}/models/{repo_id}", {"blobs": "true"})
        summary = self._summarize(payload)
        summary["siblings"] = [
            {"name": s.get("rfilename"), "size": s.get("size")}
            for s in (payload.get("siblings") or [])
            if isinstance(s, dict)
        ]
        summary["config"] = payload.get("config") or {}
        summary["gated"] = payload.get("gated", False)
        summary["card_data"] = payload.get("cardData") or {}
        return summary

    def architecture_config(self, repo_id: str) -> dict[str, Any]:
        """Fetch the repo's ``config.json`` — the source for KV-cache sizing."""
        import requests

        url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise HFError(f"could not fetch config.json: {exc}") from exc
        if resp.status_code >= 400:
            return {}
        try:
            return resp.json() or {}
        except ValueError:
            return {}

    def find_quantized_siblings(self, repo_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Look for community pre-quantized builds of the same base model.

        This is the cheapest possible path to a new rung: someone else already
        paid the calibration carbon. Matching is by model name rather than by a
        declared lineage field because HF has no reliable "quantized from" link —
        so results are candidates for the operator to confirm, not proof.
        """
        base = repo_id.split("/")[-1]
        # Drop an existing quant suffix so "Qwen2.5-7B-AWQ" still finds siblings.
        stem = _QUANT_REPO_RE.sub("", base).strip("-_. ")
        if not stem:
            return []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for term in (f"{stem} AWQ", f"{stem} GPTQ"):
            try:
                hits = self.search(term, limit=limit, task=None)
            except HFError as exc:
                logger.debug("sibling search failed for %r: %s", term, exc)
                continue
            for hit in hits:
                rid = hit.get("repo_id") or ""
                if rid in seen or rid == repo_id:
                    continue
                fmt = detect_quant_format(rid, hit.get("config") or {})
                if fmt is None:
                    continue
                # Guard against "Qwen2.5-7B" matching "Qwen2.5-72B".
                if stem.lower() not in rid.lower().replace("_", "-"):
                    continue
                seen.add(rid)
                hit["quant_format"] = fmt
                out.append(hit)
        return out

    # -- shaping -------------------------------------------------------------

    @staticmethod
    def _summarize(item: dict[str, Any]) -> dict[str, Any]:
        repo_id = item.get("modelId") or item.get("id") or ""
        safet = item.get("safetensors") or {}
        params = safet.get("total")
        config = item.get("config") or {}
        return {
            "repo_id": repo_id,
            "author": item.get("author") or (repo_id.split("/")[0] if "/" in repo_id else None),
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "pipeline_tag": item.get("pipeline_tag"),
            "tags": item.get("tags") or [],
            "last_modified": item.get("lastModified"),
            "gated": item.get("gated", False),
            "private": item.get("private", False),
            "parameter_count": params,
            "parameter_count_b": round(params / 1e9, 3) if isinstance(params, (int, float)) else None,
            "architectures": config.get("architectures") or [],
            "quant_format": detect_quant_format(repo_id, config),
            "config": config,
        }


def detect_quant_format(repo_id: str, config: dict[str, Any] | None = None) -> str | None:
    """Identify an already-quantized checkpoint from its config or its name.

    Config first: ``quantization_config.quant_method`` is authoritative when
    present. The name pattern is the fallback, because a great many community
    repos carry the format only in the repo name.
    """
    cfg = config or {}
    qc = cfg.get("quantization_config") or {}
    method = str(qc.get("quant_method") or "").lower().strip()
    if method:
        # Whatever the checkpoint declares is authoritative, including methods
        # this planner will not itself select — the caller decides what to do
        # with a format it cannot size.
        return method
    match = _QUANT_REPO_RE.search(repo_id.split("/")[-1])
    if not match:
        return None
    token = match.group(1).lower()
    # Name tokens are a weak signal: "int4"/"4bit" say the bit width but not the
    # method, and AWQ and GPTQ checkpoints are not interchangeable. Report the
    # ambiguity instead of inventing a method — the planner treats an
    # unrecognised format as unsizable and falls through to a path it can size.
    if token in {"awq", "gptq", "gguf"}:
        return token
    return f"unspecified-{token}"


def infer_parameter_count_b(summary: dict[str, Any]) -> float | None:
    """Best-effort parameter count in billions.

    ``safetensors.total`` is exact when the hub has indexed the repo. Otherwise
    fall back to the size in the repo name, which is a convention rather than a
    guarantee — callers surface the basis so the operator can see which it was.
    """
    exact = summary.get("parameter_count_b")
    if isinstance(exact, (int, float)) and exact > 0:
        return float(exact)
    match = _PARAM_IN_NAME_RE.search((summary.get("repo_id") or "").split("/")[-1])
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Quantization planning
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuantPlan:
    """A concrete, servable plan — or an explained refusal.

    ``rejected`` carries every option considered and why it lost, so an operator
    can see that (say) fp16 was skipped for VRAM rather than silently ignored.
    The router's audit trail follows the same principle.
    """

    fits: bool
    quant_format: str
    source: str  # native | prequantized | inflight | local_quantize
    serve_repo_id: str
    download_repo_id: str
    est_vram_mb: float
    est_weights_mb: float
    est_kv_cache_mb: float
    est_disk_gb: float
    gpu_memory_utilization: float
    max_model_len: int
    vllm_args: list[str]
    reason: str
    parameter_count_b: float | None
    parameter_basis: str
    rejected: list[dict[str, Any]] = field(default_factory=list)
    local_quantize_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fits": self.fits,
            "quant_format": self.quant_format,
            "source": self.source,
            "serve_repo_id": self.serve_repo_id,
            "download_repo_id": self.download_repo_id,
            "est_vram_mb": round(self.est_vram_mb, 1),
            "est_weights_mb": round(self.est_weights_mb, 1),
            "est_kv_cache_mb": round(self.est_kv_cache_mb, 1),
            "est_disk_gb": round(self.est_disk_gb, 2),
            "gpu_memory_utilization": round(self.gpu_memory_utilization, 4),
            "max_model_len": self.max_model_len,
            "vllm_args": list(self.vllm_args),
            "reason": self.reason,
            "parameter_count_b": self.parameter_count_b,
            "parameter_basis": self.parameter_basis,
            "local_quantize_required": self.local_quantize_required,
            "rejected": list(self.rejected),
        }


def estimate_kv_cache_mb(arch: dict[str, Any], max_model_len: int, concurrency: int = 4) -> float:
    """KV cache bytes for one sequence set, from the model's own config.

    ``2`` for K and V, ``2`` bytes per fp16 element. Grouped-query attention
    models report ``num_key_value_heads`` below ``num_attention_heads``, and
    honouring that is the difference between a 4x overestimate and a usable
    number on modern checkpoints.
    """
    layers = arch.get("num_hidden_layers") or arch.get("n_layer") or 0
    hidden = arch.get("hidden_size") or arch.get("n_embd") or 0
    heads = arch.get("num_attention_heads") or arch.get("n_head") or 0
    kv_heads = arch.get("num_key_value_heads") or heads
    try:
        layers, hidden, heads, kv_heads = int(layers), int(hidden), int(heads), int(kv_heads)
    except (TypeError, ValueError):
        return 0.0
    if layers <= 0 or hidden <= 0 or heads <= 0:
        return 0.0
    head_dim = int(arch.get("head_dim") or (hidden // heads))
    per_token_bytes = 2 * layers * max(1, kv_heads) * head_dim * 2
    return per_token_bytes * max(1, max_model_len) * max(1, concurrency) / 1024**2


def estimate_vram_mb(
    param_count_b: float,
    quant_format: str,
    arch: dict[str, Any],
    max_model_len: int,
    concurrency: int = 4,
) -> tuple[float, float, float]:
    """Return ``(total_mb, weights_mb, kv_mb)`` for serving this checkpoint."""
    bpp = QUANT_BYTES_PER_PARAM.get(quant_format, 2.0)
    weights_mb = param_count_b * 1e9 * bpp / 1024**2
    kv_mb = estimate_kv_cache_mb(arch, max_model_len, concurrency)
    total = weights_mb * 1.05 + kv_mb + VLLM_RUNTIME_OVERHEAD_MB
    return total, weights_mb, kv_mb


def plan_quantization(
    *,
    summary: dict[str, Any],
    arch: dict[str, Any],
    resources: HostResources,
    prequantized: Iterable[dict[str, Any]] = (),
    max_model_len: int = 2048,
    concurrency: int = 4,
    prefer: str | None = None,
    allow_local_quantize: bool = False,
    local_quantizer_available: bool = False,
) -> QuantPlan:
    """Choose the cheapest servable format that fits the *current* headroom.

    Preference order, and the reasoning behind it:

    1. ``prequantized`` — someone already paid the calibration carbon, so this
       costs a download and nothing else. Always tried first.
    2. ``native`` fp16 — no quality loss at all, chosen when it comfortably fits.
       Preferred over 4-bit for small models where the VRAM saving is marginal
       but the quality risk is not.
    3. ``inflight`` bitsandbytes — vLLM quantizes at load time. No calibration
       pass, hence no calibration carbon; costs throughput, not emissions.
    4. ``local_quantize`` — a real offline AWQ pass. Only on explicit request,
       because it is the single most carbon-expensive operation here and it
       contends with live traffic for the same vGPU slice.

    An unfittable plan is returned with ``fits=False`` rather than raised, so the
    caller can show the operator exactly how far short the box is.
    """
    repo_id = summary.get("repo_id") or ""
    params = infer_parameter_count_b(summary)
    basis = "safetensors_index" if summary.get("parameter_count_b") else "repo_name"
    rejected: list[dict[str, Any]] = []

    if params is None:
        return QuantPlan(
            fits=False,
            quant_format="unknown",
            source="none",
            serve_repo_id=repo_id,
            download_repo_id=repo_id,
            est_vram_mb=0.0,
            est_weights_mb=0.0,
            est_kv_cache_mb=0.0,
            est_disk_gb=0.0,
            gpu_memory_utilization=0.0,
            max_model_len=max_model_len,
            vllm_args=[],
            reason=(
                "parameter count unknown: the hub has no safetensors index for this repo and "
                "the name carries no size. Sizing would be a guess, so onboarding is refused."
            ),
            parameter_count_b=None,
            parameter_basis="unknown",
        )

    if not resources.vram_measured:
        # Failing closed, but with the actual remedy rather than "0 MB free",
        # which would read as "the GPU is full" and send the operator hunting
        # for a problem that does not exist.
        return QuantPlan(
            fits=False,
            quant_format="unknown",
            source="none",
            serve_repo_id=repo_id,
            download_repo_id=repo_id,
            est_vram_mb=0.0,
            est_weights_mb=0.0,
            est_kv_cache_mb=0.0,
            est_disk_gb=0.0,
            gpu_memory_utilization=0.0,
            max_model_len=max_model_len,
            vllm_args=[],
            reason=(
                f"GPU memory could not be measured (basis: {resources.vram_basis}). Admitting a model "
                "without knowing what is already resident on the slice risks wedging the engines that "
                "are serving now. Fix by enabling GPU metrics on the host metrics sidecar "
                "(DISABLE_GPU_METRICS=0 for system_metrics.sh) — note that giving the API container "
                "GPU access does not help on a MIG-partitioned board, where in-container memory "
                "queries return '[Insufficient Permissions]' and only the host can read occupancy."
            ),
            parameter_count_b=params,
            parameter_basis=basis,
        )

    budget = resources.vram_budget_mb
    disk_budget = resources.disk_budget_gb

    def _candidate(
        fmt: str, source: str, serve_repo: str, download_repo: str, note: str
    ) -> QuantPlan | None:
        total, weights, kv = estimate_vram_mb(params, fmt, arch, max_model_len, concurrency)
        disk_gb = params * 1e9 * QUANT_BYTES_PER_PARAM.get(fmt, 2.0) / 1024**3
        # A local pass needs the fp16 source on disk *as well as* the output.
        if source == "local_quantize":
            disk_gb += params * 1e9 * 2.0 / 1024**3
        if total > budget:
            rejected.append(
                {
                    "quant_format": fmt,
                    "source": source,
                    "est_vram_mb": round(total, 1),
                    "reason": f"needs {total:.0f} MB VRAM, only {budget:.0f} MB free after reserve",
                }
            )
            return None
        if disk_gb > disk_budget:
            rejected.append(
                {
                    "quant_format": fmt,
                    "source": source,
                    "est_disk_gb": round(disk_gb, 2),
                    "reason": f"needs {disk_gb:.1f} GB disk, only {disk_budget:.1f} GB free after reserve",
                }
            )
            return None
        util = 0.0
        if resources.vram_total_mb > 0:
            # vLLM reads this as a fraction of the WHOLE device, not of what is
            # free — on a shared slice a default 0.9 would try to claim memory
            # the resident containers already hold. Pad 12% for fragmentation.
            #
            # On the `metrics_sidecar` basis the total is the board figure
            # (24576 MiB here) while vLLM resolves the fraction against the MIG
            # instance it actually sees (21547 MiB), so the computed fraction is
            # ~12% conservative. That errs toward a smaller KV cache rather than
            # an OOM, which is the right direction to be wrong in.
            util = min(0.95, max(0.05, (total * 1.12) / resources.vram_total_mb))
        args = [f"--max-model-len={max_model_len}", f"--gpu-memory-utilization={util:.3f}"]
        qarg = VLLM_QUANT_ARG.get(fmt)
        if qarg and fmt not in {"fp16", "bf16"}:
            # No --load-format here: vLLM 0.21 resolves the loader from the
            # checkpoint and the quantization method, and the flag takes a free
            # string rather than a validated choice set, so passing one would be
            # a guess that fails at container start instead of at plan time.
            args.append(f"--quantization={qarg}")
        return QuantPlan(
            fits=True,
            quant_format=fmt,
            source=source,
            serve_repo_id=serve_repo,
            download_repo_id=download_repo,
            est_vram_mb=total,
            est_weights_mb=weights,
            est_kv_cache_mb=kv,
            est_disk_gb=disk_gb,
            gpu_memory_utilization=util,
            max_model_len=max_model_len,
            vllm_args=args,
            reason=note,
            parameter_count_b=params,
            parameter_basis=basis,
            rejected=rejected,
            local_quantize_required=(source == "local_quantize"),
        )

    native_fmt = detect_quant_format(repo_id, summary.get("config") or arch) or "fp16"
    if native_fmt not in QUANT_BYTES_PER_PARAM:
        # vLLM 0.21 serves more methods than this planner selects (gguf, torchao,
        # modelopt and others are in its QUANTIZATION_METHODS). They are not
        # rejected as unservable — they are simply not auto-selected, because
        # sizing them from a parameter count is not something this module can do
        # honestly. An operator who wants one passes `prefer` explicitly.
        rejected.append(
            {
                "quant_format": native_fmt,
                "source": "native",
                "reason": (
                    f"checkpoint declares '{native_fmt}', which this planner cannot size from a "
                    "parameter count. vLLM may well serve it; select it explicitly to bypass sizing."
                ),
            }
        )
        native_fmt = "fp16"

    # An explicit operator override short-circuits the preference order, but is
    # still sized — "I want 4-bit" does not mean "and I accept an OOM".
    if prefer:
        plan = _candidate(prefer, "native" if prefer == native_fmt else "inflight", repo_id, repo_id, f"operator requested {prefer}")
        if plan:
            return plan

    # 1. Already quantized upstream — nothing to pay.
    if native_fmt in VLLM_QUANT_ARG:
        plan = _candidate(
            native_fmt,
            "prequantized",
            repo_id,
            repo_id,
            f"repo is already {native_fmt.upper()}; no calibration carbon is spent",
        )
        if plan:
            return plan

    for sib in prequantized:
        fmt = sib.get("quant_format")
        rid = sib.get("repo_id")
        if not fmt or not rid or fmt not in VLLM_QUANT_ARG:
            continue
        plan = _candidate(
            fmt,
            "prequantized",
            rid,
            rid,
            f"community {fmt.upper()} build of {repo_id}; the calibration carbon was already paid upstream",
        )
        if plan:
            return plan

    # 2. fp16 as published — no quality loss. Only when it fits with real slack,
    #    otherwise a 4-bit plan is the better trade.
    if native_fmt in {"fp16", "bf16"}:
        total, _, _ = estimate_vram_mb(params, "fp16", arch, max_model_len, concurrency)
        if total <= budget * 0.8:
            plan = _candidate("fp16", "native", repo_id, repo_id, "fp16 fits with headroom; no quality loss")
            if plan:
                return plan

    # 3. In-flight 4-bit: load-time, so it costs throughput rather than carbon.
    plan = _candidate(
        "bitsandbytes",
        "inflight",
        repo_id,
        repo_id,
        "no pre-quantized build found; vLLM quantizes to 4-bit at load time, "
        "which needs no calibration pass and therefore spends no extra carbon",
    )
    if plan:
        return plan

    # 4. Offline AWQ. Explicit opt-in only.
    if allow_local_quantize:
        if not local_quantizer_available:
            rejected.append(
                {
                    "quant_format": "awq",
                    "source": "local_quantize",
                    "reason": "no local AWQ toolchain installed (autoawq/llm-compressor absent)",
                }
            )
        else:
            plan = _candidate(
                "awq",
                "local_quantize",
                repo_id,
                repo_id,
                "no upstream build fits; running a local AWQ calibration pass (metered, "
                "booked as embodied carbon with a payback estimate)",
            )
            if plan:
                return plan

    return QuantPlan(
        fits=False,
        quant_format=native_fmt,
        source="none",
        serve_repo_id=repo_id,
        download_repo_id=repo_id,
        est_vram_mb=estimate_vram_mb(params, native_fmt, arch, max_model_len, concurrency)[0],
        est_weights_mb=params * 1e9 * QUANT_BYTES_PER_PARAM.get(native_fmt, 2.0) / 1024**2,
        est_kv_cache_mb=estimate_kv_cache_mb(arch, max_model_len, concurrency),
        est_disk_gb=params * 1e9 * QUANT_BYTES_PER_PARAM.get(native_fmt, 2.0) / 1024**3,
        gpu_memory_utilization=0.0,
        max_model_len=max_model_len,
        vllm_args=[],
        reason=(
            f"no servable format fits: {budget:.0f} MB VRAM and {disk_budget:.1f} GB disk free. "
            "Free a resident model, lower max_model_len, or pick a smaller checkpoint."
        ),
        parameter_count_b=params,
        parameter_basis=basis,
        rejected=rejected,
    )


def local_quantizer_available() -> bool:
    """True when an offline AWQ/GPTQ toolchain is importable."""
    import importlib.util

    return any(importlib.util.find_spec(m) is not None for m in ("awq", "llmcompressor", "gptqmodel"))


# ─────────────────────────────────────────────────────────────────────────────
# Docker control plane (stdlib HTTP over the unix socket)
#
# The API container needs to launch vLLM containers to serve newly onboarded
# models. The `docker` SDK is not a project dependency and the socket is not
# mounted by default — both are deliberate, and both are documented as the
# operator's explicit opt-in, because mounting the socket is equivalent to
# granting root on the host.
# ─────────────────────────────────────────────────────────────────────────────


class DockerError(RuntimeError):
    pass


class _UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 60.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:  # noqa: D102 - stdlib override
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


class DockerClient:
    """Minimal Docker Engine API client over the unix socket.

    Only the verbs onboarding needs: list, create, start, stop, remove, inspect,
    pull. Raw stdlib so no dependency is added for what is ultimately a handful
    of JSON calls.
    """

    def __init__(self, socket_path: str | None = None, timeout_s: float = 120.0) -> None:
        self.socket_path = socket_path or os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return os.path.exists(self.socket_path)

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        if not self.available:
            raise DockerError(
                f"docker socket not present at {self.socket_path}. Dynamic serving requires "
                "mounting it into the API container (see MODEL_SERVE_* in .env.example)."
            )
        conn = _UnixHTTPConnection(self.socket_path, timeout=self.timeout_s)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json", "Host": "localhost"}
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        except OSError as exc:
            raise DockerError(f"docker socket error: {exc}") from exc
        finally:
            conn.close()
        if status >= 400:
            detail = raw.decode("utf-8", "replace")[:300]
            raise DockerError(f"docker API {status} on {method} {path}: {detail}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return raw.decode("utf-8", "replace")

    def ping(self) -> bool:
        try:
            self._request("GET", "/_ping")
            return True
        except DockerError:
            return False

    def list_containers(self, *, all_states: bool = True, label: str | None = None) -> list[dict[str, Any]]:
        path = f"/containers/json?all={'1' if all_states else '0'}"
        if label:
            filt = json.dumps({"label": [label]})
            path += f"&filters={filt}"
        return self._request("GET", path) or []

    def inspect(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/containers/{name}/json") or {}

    def create(self, name: str, config: dict[str, Any]) -> str:
        out = self._request("POST", f"/containers/create?name={name}", config) or {}
        cid = out.get("Id")
        if not cid:
            raise DockerError(f"container create returned no id: {out}")
        return cid

    def start(self, name: str) -> None:
        self._request("POST", f"/containers/{name}/start")

    def stop(self, name: str, timeout_s: int = 30) -> None:
        self._request("POST", f"/containers/{name}/stop?t={timeout_s}")

    def remove(self, name: str, force: bool = True) -> None:
        self._request("DELETE", f"/containers/{name}?force={'1' if force else '0'}&v=0")

    def exec_run(self, name: str, cmd: list[str], timeout_s: float = 30.0) -> str:
        """Run a command in a container and return its combined output.

        ``Tty: true`` is deliberate: without it the Engine API returns a
        multiplexed stream with 8-byte frame headers, and demultiplexing that is
        more machinery than a one-shot probe warrants. With a TTY the body is
        plain text. The API image has no ``docker`` CLI, so the Engine API is the
        only way to reach into a container from here.
        """
        exec_id = (self._request(
            "POST",
            f"/containers/{name}/exec",
            {"AttachStdout": True, "AttachStderr": True, "Tty": True, "Cmd": cmd},
        ) or {}).get("Id")
        if not exec_id:
            raise DockerError("exec create returned no id")
        conn = _UnixHTTPConnection(self.socket_path, timeout=timeout_s)
        body = json.dumps({"Detach": False, "Tty": True}).encode("utf-8")
        try:
            conn.request(
                "POST",
                f"/exec/{exec_id}/start",
                body=body,
                headers={"Content-Type": "application/json", "Host": "localhost"},
            )
            resp = conn.getresponse()
            return resp.read().decode("utf-8", "replace")
        except OSError as exc:
            raise DockerError(f"exec start failed: {exc}") from exc
        finally:
            conn.close()

    def logs_tail(self, name: str, lines: int = 40) -> str:
        try:
            out = self._request("GET", f"/containers/{name}/logs?stdout=1&stderr=1&tail={lines}")
        except DockerError:
            return ""
        if isinstance(out, bytes):
            return out.decode("utf-8", "replace")
        return str(out or "")


# ─────────────────────────────────────────────────────────────────────────────
# Serving
#
# A registered model is only useful if traffic can reach it. `resolve_vllm_endpoint`
# in decision_engine reads a zoo entry's `vllm_endpoint_env` with os.getenv at
# *lookup* time, explicitly so a deployment can rotate an endpoint without
# restarting the API. Onboarding uses that same seam: serving a model exports
# `VLLM_ONBOARD_<SLUG>_URL` into the API process and the zoo entry points at it.
# The URL is also persisted on the entry so `restore_endpoints` can re-export it
# after a restart — an env var set in-process does not survive one.
# ─────────────────────────────────────────────────────────────────────────────

CONTAINER_PREFIX = "green-onboard-"
ONBOARD_LABEL = "green.onboarded=1"


def endpoint_env_name(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    return f"VLLM_ONBOARD_{slug}_URL"


def container_name_for(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return f"{CONTAINER_PREFIX}{slug}"


@dataclass
class ServeResult:
    ok: bool
    container: str
    url: str | None
    port: int
    detail: str
    logs: str = ""


class ServingManager:
    """Launch, health-gate and stop vLLM containers for onboarded models.

    Containers join the compose network and are addressed by container name, so
    no host port is published and there is no port-collision surface with the
    statically declared services.

    Admission is against measured free VRAM. Only containers this manager
    created are eligible for eviction — the statically declared vllm-* services
    carry live traffic and are never touched, because a router that can silently
    kill its own backends is worse than one that refuses to admit a model.
    """

    def __init__(
        self,
        docker: DockerClient | None = None,
        *,
        image: str | None = None,
        network: str | None = None,
        hf_cache_host_path: str | None = None,
        hf_token: str | None = None,
        port_base: int | None = None,
    ) -> None:
        self.docker = docker or DockerClient()
        self.image = image or os.getenv("MODEL_SERVE_IMAGE", "vllm/vllm-openai:latest")
        self.network = network or os.getenv("MODEL_SERVE_NETWORK", "green_default")
        self.hf_cache_host_path = hf_cache_host_path or os.getenv(
            "HF_CACHE_HOST_PATH", "/opt/green/data/hf-cache"
        )
        self.hf_token = (hf_token or "").strip()
        self.port_base = port_base or _env_int("MODEL_SERVE_PORT_BASE", 8010)
        self._lock = threading.RLock()

    # -- capability reporting -------------------------------------------------

    def capability(self) -> dict[str, Any]:
        """What this manager can actually do here, and why not when it cannot."""
        sock = self.docker.socket_path
        if not self.docker.available:
            return {
                "enabled": False,
                "reason": (
                    f"docker socket not mounted at {sock}. Dynamic serving needs it bind-mounted "
                    "into the API container. That grants root-equivalent control of the host, so "
                    "it is opt-in and off by default."
                ),
                "socket_path": sock,
            }
        if not self.docker.ping():
            return {"enabled": False, "reason": "docker socket present but not responding", "socket_path": sock}
        return {"enabled": True, "reason": "docker socket reachable", "socket_path": sock, "image": self.image}

    def supported_quant(self) -> dict[str, Any]:
        """Quantization methods the *running* vLLM actually supports.

        Read from a live container rather than assumed, because the supported
        set is version-dependent. When no vLLM container is up, the planner's own
        table is returned with the basis marked so the caller knows it is a
        default rather than an observation.
        """
        try:
            containers = self.docker.list_containers(all_states=False)
        except DockerError:
            containers = []
        for c in containers:
            names = [n.lstrip("/") for n in (c.get("Names") or [])]
            image = str(c.get("Image") or "")
            if not any("vllm" in n for n in names) and "vllm" not in image:
                continue
            name = names[0] if names else None
            if not name:
                continue
            try:
                out = self.docker.exec_run(
                    name,
                    [
                        "python3", "-c",
                        "from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as Q;"
                        "import json;print(json.dumps(sorted(Q)))",
                    ],
                )
            except DockerError as exc:
                logger.debug("could not probe %s for quantization methods: %s", name, exc)
                break
            # A TTY-attached exec echoes progress and warnings too; the payload is
            # the last line that parses as a JSON list.
            for line in reversed(out.strip().splitlines()):
                try:
                    methods = json.loads(line.strip())
                except ValueError:
                    continue
                if isinstance(methods, list):
                    return {"methods": methods, "basis": f"observed:{name}"}
            break
        return {"methods": sorted(VLLM_QUANT_ARG), "basis": "planner_default"}

    # -- admission ------------------------------------------------------------

    def onboarded_containers(self) -> list[dict[str, Any]]:
        try:
            return [
                c for c in self.docker.list_containers(all_states=True, label=ONBOARD_LABEL)
            ]
        except DockerError:
            return []

    def admit(self, plan: QuantPlan, resources: HostResources) -> tuple[bool, str]:
        """Can this plan be served right now without oversubscribing the slice?"""
        if plan.est_vram_mb <= resources.vram_budget_mb:
            return True, "fits in current free VRAM"
        evictable = [
            c for c in self.onboarded_containers()
            if str(c.get("State") or "").lower() == "running"
        ]
        if not evictable:
            return False, (
                f"needs {plan.est_vram_mb:.0f} MB VRAM, {resources.vram_budget_mb:.0f} MB free, and no "
                "onboarded container is running to evict. The statically declared vllm-* services are "
                "never evicted automatically."
            )
        return False, (
            f"needs {plan.est_vram_mb:.0f} MB VRAM, {resources.vram_budget_mb:.0f} MB free. Stop one of "
            f"{[ (c.get('Names') or ['?'])[0].lstrip('/') for c in evictable ]} first."
        )

    # -- lifecycle ------------------------------------------------------------

    def serve(
        self,
        model_id: str,
        plan: QuantPlan,
        *,
        trust_remote_code: bool = False,
        ready_timeout_s: float = 900.0,
        poll_s: float = 5.0,
    ) -> ServeResult:
        """Start a vLLM container and block until /health passes or it fails.

        The readiness gate is the point: a container that started is not a
        backend. On failure the container's own log tail is returned, because
        "it did not come up" without the reason is useless to an operator.
        """
        name = container_name_for(model_id)
        port = self._pick_port()
        cmd = [
            f"--model={plan.serve_repo_id}",
            "--host=0.0.0.0",
            f"--port={port}",
            f"--served-model-name={plan.serve_repo_id}",
            *plan.vllm_args,
        ]
        if trust_remote_code:
            # Arbitrary code execution from a third-party repo. Never defaulted on.
            cmd.append("--trust-remote-code")

        env = [f"HF_HOME=/root/.cache/huggingface"]
        if self.hf_token:
            env.append(f"HF_TOKEN={self.hf_token}")

        config: dict[str, Any] = {
            "Image": self.image,
            "Cmd": cmd,
            "Env": env,
            "Labels": {
                "green.onboarded": "1",
                "green.model_id": model_id,
                "green.repo_id": plan.serve_repo_id,
                "green.quant_format": plan.quant_format,
            },
            "HostConfig": {
                "Binds": [f"{self.hf_cache_host_path}:/root/.cache/huggingface"],
                "NetworkMode": self.network,
                "ShmSize": 4 * 1024**3,
                "RestartPolicy": {"Name": "unless-stopped"},
                "DeviceRequests": [
                    {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
                ],
            },
        }

        with self._lock:
            try:
                existing = self.docker.inspect(name)
            except DockerError:
                existing = {}
            if existing:
                try:
                    self.docker.remove(name, force=True)
                except DockerError as exc:
                    return ServeResult(False, name, None, port, f"could not replace existing container: {exc}")
            try:
                self.docker.create(name, config)
                self.docker.start(name)
            except DockerError as exc:
                return ServeResult(False, name, None, port, f"container start failed: {exc}",
                                   logs=self.docker.logs_tail(name))

        url = f"http://{name}:{port}/v1"
        ok, detail = self._wait_ready(
            name, port, ready_timeout_s, poll_s, model_name=plan.serve_repo_id
        )
        if not ok:
            return ServeResult(False, name, None, port, detail, logs=self.docker.logs_tail(name, 60))
        return ServeResult(True, name, url, port, detail)

    def stop(self, model_id: str, *, remove: bool = True) -> tuple[bool, str]:
        name = container_name_for(model_id)
        try:
            self.docker.stop(name)
            if remove:
                self.docker.remove(name, force=True)
        except DockerError as exc:
            return False, str(exc)
        return True, f"stopped {name}"

    def _pick_port(self) -> int:
        """Pick a container-internal port not already used by an onboarded peer.

        Ports are internal to the compose network (nothing is published to the
        host), so this only needs to avoid collisions among onboarded containers.
        """
        used: set[int] = set()
        for c in self.onboarded_containers():
            for p in c.get("Ports") or []:
                try:
                    used.add(int(p.get("PrivatePort")))
                except (TypeError, ValueError):
                    continue
        port = self.port_base
        while port in used and port < self.port_base + 200:
            port += 1
        return port

    def _wait_ready(
        self,
        name: str,
        port: int,
        timeout_s: float,
        poll_s: float,
        *,
        model_name: str,
        generate_timeout_s: float = 60.0,
    ) -> tuple[bool, str]:
        """Block until the container can actually *generate*, not merely answer /health.

        Gating on /health is not sufficient and this is not hypothetical: on
        2026-07-29 all three vLLM containers on this host returned 200 from
        /health for hours while their EngineCore processes were wedged and every
        completion hung until the client timed out. A readiness check that the
        wedged state passes is worse than none, because it certifies a dead
        backend as live and the router then sends real traffic to it.

        So the gate is a one-token completion. /health is still polled first, but
        only as the cheap precondition for attempting the real probe.
        """
        import requests

        deadline = time.monotonic() + timeout_s
        started = time.monotonic()
        health = f"http://{name}:{port}/health"
        completions = f"http://{name}:{port}/v1/completions"
        last = "no probe attempted"
        while time.monotonic() < deadline:
            try:
                state = self.docker.inspect(name).get("State") or {}
            except DockerError as exc:
                return False, f"container vanished during startup: {exc}"
            if str(state.get("Status") or "").lower() in {"exited", "dead"}:
                return False, f"container exited during startup (code {state.get('ExitCode')})"

            healthy = False
            try:
                healthy = requests.get(health, timeout=5).status_code == 200
            except Exception as exc:  # noqa: BLE001 - not up yet is the normal case
                last = f"health not reachable: {type(exc).__name__}"

            if healthy:
                try:
                    resp = requests.post(
                        completions,
                        json={"model": model_name, "prompt": "ping", "max_tokens": 1, "temperature": 0},
                        timeout=generate_timeout_s,
                    )
                    if resp.status_code == 200:
                        return True, f"generated a token after {time.monotonic() - started:.0f}s"
                    last = f"health ok but generation returned {resp.status_code}: {resp.text[:120]}"
                except Exception as exc:  # noqa: BLE001
                    last = f"health ok but generation failed: {type(exc).__name__}"
            time.sleep(poll_s)
        return False, f"not ready within {timeout_s:.0f}s ({last})"


# ─────────────────────────────────────────────────────────────────────────────
# Local quantization (opt-in, metered)
#
# Running an AWQ/GPTQ calibration pass is the most carbon-expensive thing this
# system can do, and on this deployment it contends with live traffic for the
# same vGPU slice. So it is not the default path, it never runs in-process, and
# its cost is measured rather than assumed.
#
# There is no standard image that ships a calibration toolchain, so the runner
# is generic: the operator names an image and a command template. With
# MODEL_QUANT_IMAGE unset the capability reports itself unavailable instead of
# pretending to work.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuantizationResult:
    ok: bool
    detail: str
    duration_s: float = 0.0
    energy_wh: float = 0.0
    carbon_g: float = 0.0
    grid_ci: float = 0.0
    output_dir: str | None = None
    logs: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "detail": self.detail,
            "duration_s": round(self.duration_s, 1),
            "energy_wh": round(self.energy_wh, 3),
            "carbon_g": round(self.carbon_g, 4),
            "grid_ci_g_per_kwh": round(self.grid_ci, 1),
            "output_dir": self.output_dir,
        }


class LocalQuantizer:
    """One-shot containerised quantization pass, metered for carbon.

    Energy uses the same power x duration form as ``model_zoo`` operational
    carbon — spec TDP against *measured* wall-clock, divided by hardware
    efficiency and multiplied by PUE. The TDP is an upper bound on real draw
    (this vGPU slice reports ``[N/A]`` for ``power.draw``), so the figure is an
    upper bound too, and it is labelled as such wherever it surfaces.
    """

    def __init__(
        self,
        docker: DockerClient | None = None,
        *,
        grid_ci: Callable[[], float] | None = None,
        image: str | None = None,
        command_template: str | None = None,
        hf_cache_host_path: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.docker = docker or DockerClient()
        self.grid_ci = grid_ci or (lambda: _env_float("GRID_CARBON_FALLBACK", 475.0))
        self.image = (image if image is not None else os.getenv("MODEL_QUANT_IMAGE", "")).strip()
        self.command_template = (
            command_template
            if command_template is not None
            else os.getenv("MODEL_QUANT_COMMAND", "")
        ).strip()
        self.hf_cache_host_path = hf_cache_host_path or os.getenv(
            "HF_CACHE_HOST_PATH", "/opt/green/data/hf-cache"
        )
        self.hf_token = (hf_token or "").strip()

    def capability(self) -> dict[str, Any]:
        if not self.docker.available:
            return {"enabled": False, "reason": "docker socket not mounted; local quantization runs in a container"}
        if not self.image:
            return {
                "enabled": False,
                "reason": (
                    "MODEL_QUANT_IMAGE is unset. Local AWQ/GPTQ needs an image carrying a calibration "
                    "toolchain (e.g. autoawq or llm-compressor); none ships with this stack. Prefer a "
                    "pre-quantized checkpoint from the hub, which costs no calibration carbon at all."
                ),
            }
        if not self.command_template:
            return {"enabled": False, "reason": "MODEL_QUANT_COMMAND is unset; nothing to run in MODEL_QUANT_IMAGE"}
        return {"enabled": True, "reason": "quantization image configured", "image": self.image}

    def run(self, repo_id: str, out_dir: str, *, timeout_s: float = 14400.0) -> QuantizationResult:
        cap = self.capability()
        if not cap["enabled"]:
            return QuantizationResult(False, cap["reason"])

        name = f"{CONTAINER_PREFIX}quant-{uuid.uuid4().hex[:8]}"
        cmd = self.command_template.format(repo_id=repo_id, out_dir=out_dir)
        env = ["HF_HOME=/root/.cache/huggingface"]
        if self.hf_token:
            env.append(f"HF_TOKEN={self.hf_token}")
        config = {
            "Image": self.image,
            "Cmd": ["sh", "-lc", cmd],
            "Env": env,
            "Labels": {"green.onboarded": "1", "green.role": "quantize", "green.repo_id": repo_id},
            "HostConfig": {
                "Binds": [f"{self.hf_cache_host_path}:/root/.cache/huggingface"],
                "ShmSize": 4 * 1024**3,
                "DeviceRequests": [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
            },
        }

        started = time.monotonic()
        try:
            self.docker.create(name, config)
            self.docker.start(name)
        except DockerError as exc:
            return QuantizationResult(False, f"quantization container failed to start: {exc}")

        exit_code: int | None = None
        deadline = started + timeout_s
        while time.monotonic() < deadline:
            try:
                state = self.docker.inspect(name).get("State") or {}
            except DockerError as exc:
                return QuantizationResult(False, f"quantization container vanished: {exc}")
            if str(state.get("Status") or "").lower() in {"exited", "dead"}:
                exit_code = state.get("ExitCode")
                break
            time.sleep(10)

        duration_s = time.monotonic() - started
        logs = self.docker.logs_tail(name, 80)

        tdp = _env_float("GPU_TDP", 300.0)
        pue = _env_float("DATACENTER_PUE", 1.3)
        he = max(0.05, _env_float("QUANT_HARDWARE_EFFICIENCY", 0.75))
        energy_wh = tdp * (duration_s / 3600.0) * pue / he
        ci = float(self.grid_ci() or 0.0)
        carbon_g = energy_wh / 1000.0 * ci

        try:
            self.docker.remove(name, force=True)
        except DockerError:
            pass

        if exit_code is None:
            return QuantizationResult(
                False, f"quantization timed out after {timeout_s:.0f}s", duration_s, energy_wh, carbon_g, ci, logs=logs
            )
        if exit_code != 0:
            return QuantizationResult(
                False, f"quantization exited {exit_code}", duration_s, energy_wh, carbon_g, ci, logs=logs
            )
        return QuantizationResult(
            True, "quantization completed", duration_s, energy_wh, carbon_g, ci, output_dir=out_dir, logs=logs
        )


# ─────────────────────────────────────────────────────────────────────────────
# Zoo registration
#
# The hard part is not writing JSON — it is being honest about which numbers are
# measured and which are inherited. Device-level properties (TDP, embodied
# manufacturing carbon, lifetime, slice share) genuinely are shared with the
# existing entries: it is the same physical board. Those are inherited from a
# donor entry and labelled. Model-level properties (accuracy, latency, output
# verbosity) are NOT shared and cannot be guessed, so they are left explicitly
# unmeasured and the entry stays unavailable until a caller supplies real ones.
# ─────────────────────────────────────────────────────────────────────────────

INHERITED_DEVICE_FIELDS = (
    "power_tdp_w",
    "hardware_efficiency",
    "pue",
    "mfg_carbon_kg",
    "device_lifetime_years",
    "device_utilization",
    "device_share",
    "annual_inference_volume",
    "region",
    "region_label",
    "region_carbon_multiplier",
    "grid_zone",
    "hardware",
    "hardware_class",
    "hardware_affinity",
)


def build_zoo_entry(
    model_id: str,
    summary: dict[str, Any],
    plan: QuantPlan,
    *,
    donor: dict[str, Any],
    endpoint_env: str,
    endpoint_url: str | None,
    max_output_tokens: int = 512,
    cost_units: float | None = None,
) -> dict[str, Any]:
    """Build a draft zoo entry: routable-shaped, but not yet routable."""
    params = plan.parameter_count_b or 0.0
    entry: dict[str, Any] = {
        "id": model_id,
        "model_id": plan.serve_repo_id.split("/")[-1],
        "model_variant": model_id,
        "architecture": "dense",
        "parameter_count_b": params,
        # Standard dense forward cost: ~2 FLOPs (one multiply, one add) per
        # parameter per token. CLAUDE.md: FLOP counts are reporting metadata and
        # do not feed any carbon number, so this cannot skew routing.
        "flop_count_per_token": int(params * 1e9 * 2),
        "quantization": plan.quant_format,
        "supports_batching": True,
        "moe": False,
        "all_to_all_overhead_ratio": 0.0,
        "active_expert_ratio": 1.0,
        "modality": "text",
        # --- the gate -------------------------------------------------------
        "available": False,
        "accuracy_baseline": 0.0,
        "latency_ms_p50": 0,
        "latency_ms_p95": 0,
        "accuracy_basis": "unmeasured",
        "latency_basis": "unmeasured",
        "max_output_tokens": int(max_output_tokens),
        # --- provenance -----------------------------------------------------
        "onboarded": True,
        "onboarded_at": utc_now_iso(),
        "source_repo_id": summary.get("repo_id"),
        "serve_repo_id": plan.serve_repo_id,
        "quant_source": plan.source,
        "quant_plan_reason": plan.reason,
        "parameter_basis": plan.parameter_basis,
        "device_fields_basis": f"inherited:{donor.get('id')}",
        "est_vram_mb": round(plan.est_vram_mb, 1),
        # --- dispatch -------------------------------------------------------
        "vllm_endpoint_env": endpoint_env,
        "vllm_endpoint_url": endpoint_url,
        "vllm_model_id": plan.serve_repo_id,
    }
    for key in INHERITED_DEVICE_FIELDS:
        if key in donor:
            entry[key] = donor[key]
    entry["cost_units"] = cost_units if cost_units is not None else donor.get("cost_units", 0.5)
    return entry


def estimate_payback(
    quantization_carbon_g: float,
    new_carbon_g_per_request: float | None,
    incumbent_carbon_g_per_request: float | None,
) -> dict[str, Any] | None:
    """Requests needed before a local quantization pass has paid for itself.

    Only computable once *both* rungs have measured per-request carbon — before
    that there is no saving to divide into, and a projected payback would be the
    same kind of declared-not-measured figure this module exists to avoid. A
    non-positive saving is reported as such rather than as an infinite payback,
    because "this rung is not cheaper" is the useful answer.
    """
    if not quantization_carbon_g or new_carbon_g_per_request is None or incumbent_carbon_g_per_request is None:
        return None
    saving = float(incumbent_carbon_g_per_request) - float(new_carbon_g_per_request)
    if saving <= 0:
        return {
            "pays_back": False,
            "saving_g_per_request": round(saving, 6),
            "detail": "the onboarded rung is not cheaper per request, so the quantization carbon is never recovered",
        }
    return {
        "pays_back": True,
        "saving_g_per_request": round(saving, 6),
        "requests_to_payback": int(round(float(quantization_carbon_g) / saving)),
        "quantization_carbon_g": round(float(quantization_carbon_g), 4),
    }


def measurement_patch(measurement: dict[str, Any], basis: str) -> dict[str, Any]:
    """Turn a measurement summary into the fields that make an entry routable.

    Requires accuracy and p50 latency at minimum. ``expected_output_tokens`` is
    applied when the run observed it — the same field, and the same
    measured-vs-estimated discipline, that ``expected_output_tokens_basis``
    already carries for the shipped models.
    """
    accuracy = measurement.get("accuracy")
    p50 = measurement.get("latency_ms_p50")
    if accuracy is None or p50 is None:
        raise ValueError("measurement must carry at least 'accuracy' and 'latency_ms_p50'")
    acc = float(accuracy)
    if not 0.0 <= acc <= 1.0:
        raise ValueError(f"accuracy must be a 0..1 pass rate, got {accuracy!r}")
    patch: dict[str, Any] = {
        "accuracy_baseline": round(acc, 4),
        "latency_ms_p50": int(round(float(p50))),
        "accuracy_basis": basis,
        "latency_basis": basis,
        "measured_at": utc_now_iso(),
        "measured_samples": measurement.get("samples"),
        "available": True,
    }
    if measurement.get("latency_ms_p95") is not None:
        patch["latency_ms_p95"] = int(round(float(measurement["latency_ms_p95"])))
    if measurement.get("expected_output_tokens") is not None:
        patch["expected_output_tokens"] = int(round(float(measurement["expected_output_tokens"])))
        patch["expected_output_tokens_basis"] = basis
    return patch


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

JOB_STATES = ("queued", "planning", "downloading", "quantizing", "registering", "serving", "succeeded", "failed")


@dataclass
class OnboardJob:
    job_id: str
    repo_id: str
    model_id: str
    state: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    steps: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    error: str | None = None
    carbon_g: float = 0.0
    auto_serve: bool = False

    def note(self, step: str, detail: str, **extra: Any) -> None:
        self.steps.append({"step": step, "detail": detail, "at": utc_now_iso(), **extra})
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "repo_id": self.repo_id,
            "model_id": self.model_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": list(self.steps),
            "plan": self.plan,
            "error": self.error,
            "quantization_carbon_g": round(self.carbon_g, 4),
            "auto_serve": self.auto_serve,
        }


class ModelOnboardingService:
    """Browse → plan → download → (quantize) → register → serve, as one pipeline.

    Zoo writes go through the injected ``ModelZooService`` so the live registry
    and the on-disk config stay in step. Grid carbon intensity is a callable for
    the same reason ``WorkflowServices`` takes callables: importing
    ``decision_engine`` from here would cycle.
    """

    def __init__(
        self,
        zoo: Any,
        *,
        hf_token: str | None = None,
        grid_ci: Callable[[], float] | None = None,
        state_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        serving: ServingManager | None = None,
        quantizer: LocalQuantizer | None = None,
        system_metrics: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.zoo = zoo
        # Injected like grid_ci: the sidecar is the system's existing GPU
        # telemetry seam, and importing decision_engine here would cycle.
        self.system_metrics = system_metrics
        self.hf_token = (hf_token or os.getenv("HF_TOKEN", "")).strip()
        self.catalog = HFCatalog(self.hf_token)
        self.grid_ci = grid_ci or (lambda: _env_float("GRID_CARBON_FALLBACK", 475.0))
        self.cache_dir = str(cache_dir or os.getenv("HF_HOME", "/app/data/hf-cache"))
        self.state_path = Path(state_path or os.getenv("MODEL_ONBOARD_STATE_PATH", "data/model_onboarding.json"))
        self.serving = serving or ServingManager(hf_token=self.hf_token)
        self.quantizer = quantizer or LocalQuantizer(grid_ci=self.grid_ci, hf_token=self.hf_token)
        self._jobs: dict[str, OnboardJob] = {}
        self._lock = threading.RLock()
        self._load_jobs()

    def _probe(self) -> HostResources:
        return probe_resources(self.cache_dir, self.system_metrics)

    # -- capability ----------------------------------------------------------

    def capability(self) -> dict[str, Any]:
        return {
            "enabled": _env_flag("MODEL_ONBOARD_ENABLED", False),
            "hf_token_present": bool(self.hf_token),
            "serving": self.serving.capability(),
            "local_quantization": self.quantizer.capability(),
            "supported_quantization": self.serving.supported_quant(),
            "resources": self._probe().to_dict(),
        }

    # -- browse --------------------------------------------------------------

    def search(self, query: str = "", **kwargs: Any) -> list[dict[str, Any]]:
        return self.catalog.search(query, **kwargs)

    def preview(
        self,
        repo_id: str,
        *,
        max_model_len: int = 2048,
        prefer: str | None = None,
        allow_local_quantize: bool = False,
        find_siblings: bool = True,
    ) -> dict[str, Any]:
        """Everything an operator needs to decide, without committing anything."""
        summary = self.catalog.detail(repo_id)
        arch = self.catalog.architecture_config(repo_id)
        resources = self._probe()
        siblings = self.catalog.find_quantized_siblings(repo_id) if find_siblings else []
        plan = plan_quantization(
            summary=summary,
            arch=arch,
            resources=resources,
            prequantized=siblings,
            max_model_len=max_model_len,
            prefer=prefer,
            allow_local_quantize=allow_local_quantize,
            local_quantizer_available=self.quantizer.capability()["enabled"],
        )
        admit_ok, admit_detail = self.serving.admit(plan, resources) if plan.fits else (False, plan.reason)
        return {
            "model": {k: v for k, v in summary.items() if k != "siblings"},
            "architecture": {
                k: arch.get(k)
                for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "max_position_embeddings")
                if arch.get(k) is not None
            },
            "resources": resources.to_dict(),
            "prequantized_candidates": siblings,
            "plan": plan.to_dict(),
            "admission": {"ok": admit_ok, "detail": admit_detail},
        }

    # -- onboarding ----------------------------------------------------------

    def start(
        self,
        repo_id: str,
        *,
        model_id: str | None = None,
        max_model_len: int = 2048,
        prefer: str | None = None,
        allow_local_quantize: bool = False,
        auto_serve: bool = False,
        trust_remote_code: bool = False,
        donor_id: str | None = None,
        max_output_tokens: int = 512,
    ) -> OnboardJob:
        if not _env_flag("MODEL_ONBOARD_ENABLED", False):
            raise PermissionError("model onboarding is disabled; set MODEL_ONBOARD_ENABLED=true to enable it")
        if trust_remote_code and not _env_flag("MODEL_ONBOARD_ALLOW_TRUST_REMOTE_CODE", False):
            raise PermissionError(
                "trust_remote_code executes arbitrary code from the model repo. It is refused unless "
                "MODEL_ONBOARD_ALLOW_TRUST_REMOTE_CODE=true is set deliberately."
            )
        mid = model_id or f"onboard-{re.sub(r'[^a-z0-9]+', '-', repo_id.lower()).strip('-')}"
        job = OnboardJob(job_id=uuid.uuid4().hex[:12], repo_id=repo_id, model_id=mid, auto_serve=auto_serve)
        with self._lock:
            self._jobs[job.job_id] = job
            self._save_jobs()
        thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            kwargs={
                "max_model_len": max_model_len,
                "prefer": prefer,
                "allow_local_quantize": allow_local_quantize,
                "trust_remote_code": trust_remote_code,
                "donor_id": donor_id,
                "max_output_tokens": max_output_tokens,
            },
            daemon=True,
            name=f"onboard-{job.job_id}",
        )
        thread.start()
        return job

    def _run_job(
        self,
        job: OnboardJob,
        *,
        max_model_len: int,
        prefer: str | None,
        allow_local_quantize: bool,
        trust_remote_code: bool,
        donor_id: str | None,
        max_output_tokens: int,
    ) -> None:
        try:
            job.state = "planning"
            preview = self.preview(
                job.repo_id,
                max_model_len=max_model_len,
                prefer=prefer,
                allow_local_quantize=allow_local_quantize,
            )
            job.plan = preview["plan"]
            if not preview["plan"]["fits"]:
                raise RuntimeError(preview["plan"]["reason"])
            job.note("plan", preview["plan"]["reason"], quant_format=preview["plan"]["quant_format"])

            plan = _plan_from_dict(preview["plan"])

            job.state = "downloading"
            path = self._download(plan.download_repo_id, job)
            job.note("download", f"checkpoint present at {path}")

            if plan.source == "local_quantize":
                job.state = "quantizing"
                out_dir = os.path.join(self.cache_dir, "quantized", job.model_id)
                result = self.quantizer.run(plan.download_repo_id, out_dir)
                job.carbon_g = result.carbon_g
                job.note("quantize", result.detail, **result.to_dict())
                if not result.ok:
                    raise RuntimeError(result.detail)
                plan.serve_repo_id = out_dir

            job.state = "registering"
            donor = self._pick_donor(donor_id)
            entry = build_zoo_entry(
                job.model_id,
                preview["model"],
                plan,
                donor=donor,
                endpoint_env=endpoint_env_name(job.model_id),
                endpoint_url=None,
                max_output_tokens=max_output_tokens,
            )
            if job.carbon_g:
                entry["quantization_carbon_g"] = round(job.carbon_g, 4)
                entry["quantization_carbon_basis"] = "measured_wallclock_x_spec_tdp"
            self.zoo.register_model(entry)
            job.note(
                "register",
                f"registered {job.model_id} as unavailable — CSS cannot route to it until measured",
                device_fields_basis=entry["device_fields_basis"],
            )

            if job.auto_serve:
                job.state = "serving"
                served = self.serve(job.model_id, trust_remote_code=trust_remote_code)
                job.note("serve", served["detail"], url=served.get("url"))
                if not served["ok"]:
                    raise RuntimeError(served["detail"])

            job.state = "succeeded"
        except Exception as exc:  # noqa: BLE001 - job boundary
            job.state = "failed"
            job.error = str(exc)
            job.note("failed", str(exc))
            logger.warning("onboarding job %s failed: %s", job.job_id, exc)
        finally:
            job.updated_at = utc_now_iso()
            with self._lock:
                self._save_jobs()

    def _download(self, repo_id: str, job: OnboardJob) -> str:
        """Fetch the checkpoint into the shared HF cache, after a disk preflight."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - dependency is present in the API image
            raise RuntimeError("huggingface_hub is not installed in this image") from exc

        resources = self._probe()
        need_gb = float((job.plan or {}).get("est_disk_gb") or 0.0)
        if need_gb and need_gb > resources.disk_budget_gb:
            raise RuntimeError(
                f"download needs ~{need_gb:.1f} GB but only {resources.disk_budget_gb:.1f} GB is free "
                f"after the {resources.disk_reserve_gb:.0f} GB reserve"
            )
        job.note("download", f"fetching {repo_id} (~{need_gb:.1f} GB)")
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=self.cache_dir,
            token=self.hf_token or None,
            # Weights and the metadata vLLM needs. Excluding the duplicate
            # PyTorch .bin copies that most repos ship alongside safetensors
            # roughly halves the download for no loss.
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.py"],
        )

    def _pick_donor(self, donor_id: str | None) -> dict[str, Any]:
        """Choose the entry whose *device* properties the new model inherits.

        Device fields describe the physical board, so the donor must be a model
        on the same hardware. Defaults to the highest-parameter local vGPU entry,
        which is the one most likely to have been maintained.
        """
        models = [m for m in self.zoo.list_models() if not m.get("onboarded")]
        if donor_id:
            for m in models:
                if m.get("id") == donor_id:
                    return m
            raise ValueError(f"donor model {donor_id!r} not found in the zoo")
        candidates = [m for m in models if m.get("hardware") == "vgpu" and m.get("region") == "local"]
        if not candidates:
            raise RuntimeError("no local vGPU model in the zoo to inherit device properties from")
        return max(candidates, key=lambda m: float(m.get("parameter_count_b") or 0))

    # -- serving -------------------------------------------------------------

    def serve(self, model_id: str, *, trust_remote_code: bool = False) -> dict[str, Any]:
        entry = self.zoo.get_model(model_id)
        if not entry:
            raise ValueError(f"model {model_id!r} is not registered")
        if not entry.get("onboarded"):
            raise ValueError(
                f"{model_id!r} is a statically declared model; it is served by compose, not by this pipeline"
            )
        resources = self._probe()
        plan = _plan_from_entry(entry, resources)
        ok, detail = self.serving.admit(plan, resources)
        if not ok:
            return {"ok": False, "detail": detail}
        result = self.serving.serve(model_id, plan, trust_remote_code=trust_remote_code)
        if result.ok and result.url:
            # This is the seam resolve_vllm_endpoint reads with os.getenv at
            # lookup time. Persisting it too so restore_endpoints can re-export
            # after a restart, which an in-process env var does not survive.
            os.environ[endpoint_env_name(model_id)] = result.url
            entry = dict(entry)
            entry["vllm_endpoint_url"] = result.url
            entry["served_container"] = result.container
            self.zoo.register_model(entry)
        return {"ok": result.ok, "detail": result.detail, "url": result.url, "logs": result.logs}

    def unserve(self, model_id: str) -> dict[str, Any]:
        """Stop the container and take the model out of routing.

        Availability is cleared first: a model whose backend is gone must not
        remain a CSS candidate, or the router will keep selecting a dead endpoint.
        """
        entry = self.zoo.get_model(model_id)
        if entry and entry.get("onboarded"):
            patched = dict(entry)
            patched["available"] = False
            patched["vllm_endpoint_url"] = None
            self.zoo.register_model(patched)
        os.environ.pop(endpoint_env_name(model_id), None)
        ok, detail = self.serving.stop(model_id)
        return {"ok": ok, "detail": detail}

    def restore_endpoints(self) -> list[str]:
        """Re-export endpoint env vars for onboarded models after a restart."""
        restored: list[str] = []
        for entry in self.zoo.list_models():
            if not entry.get("onboarded"):
                continue
            url = entry.get("vllm_endpoint_url")
            if url:
                os.environ[endpoint_env_name(entry["id"])] = str(url)
                restored.append(entry["id"])
        return restored

    # -- measurement ---------------------------------------------------------

    def apply_measurement(self, model_id: str, measurement: dict[str, Any], *, basis: str) -> dict[str, Any]:
        """Promote an onboarded model to routable using measured figures.

        This is the only path that sets ``available: true``. ``basis`` records
        where the numbers came from (e.g. ``"measured:onboard-bench-03"``) so the
        provenance is auditable later, matching how
        ``expected_output_tokens_basis`` works for the shipped models.
        """
        entry = self.zoo.get_model(model_id)
        if not entry:
            raise ValueError(f"model {model_id!r} is not registered")
        if not entry.get("onboarded"):
            raise ValueError(f"{model_id!r} was not onboarded by this pipeline; edit the zoo config directly")
        if not entry.get("vllm_endpoint_url"):
            raise ValueError(
                f"{model_id!r} has no live endpoint. Serve it before measuring, or the figures describe nothing."
            )
        patched = dict(entry)
        patched.update(measurement_patch(measurement, basis))
        payback = estimate_payback(
            float(entry.get("quantization_carbon_g") or 0.0),
            measurement.get("carbon_g_per_request"),
            measurement.get("incumbent_carbon_g_per_request"),
        )
        if payback is not None:
            patched["quantization_payback"] = payback
        self.zoo.register_model(patched)
        return patched

    # -- jobs ----------------------------------------------------------------

    def get_job(self, job_id: str) -> OnboardJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def registry(self) -> list[dict[str, Any]]:
        """Onboarded models with their gate status spelled out."""
        out = []
        for entry in self.zoo.list_models():
            if not entry.get("onboarded"):
                continue
            out.append(
                {
                    "id": entry.get("id"),
                    "source_repo_id": entry.get("source_repo_id"),
                    "serve_repo_id": entry.get("serve_repo_id"),
                    "quantization": entry.get("quantization"),
                    "quant_source": entry.get("quant_source"),
                    "parameter_count_b": entry.get("parameter_count_b"),
                    "est_vram_mb": entry.get("est_vram_mb"),
                    "available": entry.get("available", False),
                    "accuracy_basis": entry.get("accuracy_basis"),
                    "accuracy_baseline": entry.get("accuracy_baseline"),
                    "latency_ms_p50": entry.get("latency_ms_p50"),
                    "endpoint_url": entry.get("vllm_endpoint_url"),
                    "onboarded_at": entry.get("onboarded_at"),
                    "quantization_carbon_g": entry.get("quantization_carbon_g"),
                    "routable": bool(entry.get("available")) and bool(entry.get("vllm_endpoint_url")),
                }
            )
        return out

    def _load_jobs(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not load onboarding job state: %s", exc)
            return
        for raw in payload.get("jobs", []):
            try:
                job = OnboardJob(
                    job_id=raw["job_id"],
                    repo_id=raw["repo_id"],
                    model_id=raw["model_id"],
                    state=raw.get("state", "failed"),
                    created_at=raw.get("created_at", utc_now_iso()),
                    updated_at=raw.get("updated_at", utc_now_iso()),
                    steps=raw.get("steps", []),
                    plan=raw.get("plan"),
                    error=raw.get("error"),
                    carbon_g=float(raw.get("quantization_carbon_g") or 0.0),
                    auto_serve=bool(raw.get("auto_serve")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            # A job cannot survive the process that was running it; the thread is
            # gone, so anything mid-flight is reported as interrupted rather than
            # left looking active forever.
            if job.state not in {"succeeded", "failed"}:
                job.state = "failed"
                job.error = "interrupted by an API restart"
            self._jobs[job.job_id] = job

    def _save_jobs(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:200]
            self.state_path.write_text(
                json.dumps({"jobs": [j.to_dict() for j in jobs]}, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("could not persist onboarding job state: %s", exc)


def _plan_from_dict(data: dict[str, Any]) -> QuantPlan:
    return QuantPlan(
        fits=bool(data.get("fits")),
        quant_format=str(data.get("quant_format") or "fp16"),
        source=str(data.get("source") or "native"),
        serve_repo_id=str(data.get("serve_repo_id") or ""),
        download_repo_id=str(data.get("download_repo_id") or ""),
        est_vram_mb=float(data.get("est_vram_mb") or 0.0),
        est_weights_mb=float(data.get("est_weights_mb") or 0.0),
        est_kv_cache_mb=float(data.get("est_kv_cache_mb") or 0.0),
        est_disk_gb=float(data.get("est_disk_gb") or 0.0),
        gpu_memory_utilization=float(data.get("gpu_memory_utilization") or 0.0),
        max_model_len=int(data.get("max_model_len") or 2048),
        vllm_args=list(data.get("vllm_args") or []),
        reason=str(data.get("reason") or ""),
        parameter_count_b=data.get("parameter_count_b"),
        parameter_basis=str(data.get("parameter_basis") or "unknown"),
        rejected=list(data.get("rejected") or []),
        local_quantize_required=bool(data.get("local_quantize_required")),
    )


def _plan_from_entry(entry: dict[str, Any], resources: HostResources | None = None) -> QuantPlan:
    """Rebuild the serving plan from a registered entry, re-sized against now.

    Re-deriving the vLLM args rather than storing them means a model served
    months later is sized against the headroom that exists at that moment, not
    the headroom that existed when it was onboarded.
    """
    params = float(entry.get("parameter_count_b") or 0.0)
    fmt = str(entry.get("quantization") or "fp16")
    resources = resources or probe_resources()
    max_len = int(entry.get("max_model_len") or 2048)
    total, weights, kv = estimate_vram_mb(params, fmt, {}, max_len)
    total = float(entry.get("est_vram_mb") or total)
    util = 0.0
    if resources.vram_total_mb > 0:
        util = min(0.95, max(0.05, (total * 1.12) / resources.vram_total_mb))
    args = [f"--max-model-len={max_len}", f"--gpu-memory-utilization={util:.3f}"]
    qarg = VLLM_QUANT_ARG.get(fmt)
    if qarg and fmt not in {"fp16", "bf16"}:
        args.append(f"--quantization={qarg}")
    return QuantPlan(
        fits=True,
        quant_format=fmt,
        source=str(entry.get("quant_source") or "native"),
        serve_repo_id=str(entry.get("serve_repo_id") or entry.get("vllm_model_id") or ""),
        download_repo_id=str(entry.get("source_repo_id") or ""),
        est_vram_mb=total,
        est_weights_mb=weights,
        est_kv_cache_mb=kv,
        est_disk_gb=0.0,
        gpu_memory_utilization=util,
        max_model_len=max_len,
        vllm_args=args,
        reason="rebuilt from the registered entry",
        parameter_count_b=params,
        parameter_basis=str(entry.get("parameter_basis") or "unknown"),
    )
