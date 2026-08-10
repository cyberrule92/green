#!/usr/bin/env python3
"""
Adaptive Green AI API  v4.0
Per-request carbon-aware inference orchestration integrating:
- LLMCarbon operational + embodied carbon accounting (Section 3.4.2, 4.1)
- Model Zoo versioned registry (Section 3.3)
- MoE sparse routing with all-to-all overhead (Section 5.1)
- EcoServe deferred queue with low-carbon window scheduling (Section 3.5.2)
- Multi-region grid carbon signals (Section 3.5.3)
- HMAC-SHA256 signed audit log entries (Section 3.6.1)
- RL policy outcome recording foundation (Section 3.6.2)
- Hybrid RAG, semantic profiler, quality guardrails, conversation persistence
"""

from __future__ import annotations

import hashlib
import base64
import hmac
import io
import json
import asyncio
import logging
import mimetypes
import os
import re
import time
import threading
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree

import math as _math_module

import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
# urllib3 compatibility: <1.26 uses method_whitelist, >=1.26/v2 uses allowed_methods
try:
    from urllib3.util.retry import Retry
except ModuleNotFoundError:
    try:
        from requests.packages.urllib3.util.retry import Retry  # type: ignore[no-redef]
    except (ModuleNotFoundError, AttributeError):
        class Retry:  # type: ignore[no-redef]
            def __init__(self, *_, **__): pass

from advanced_rag import AdvancedRAGService
from conversation_store import ConversationStore
from deferred_queue import get_deferred_queue
from secret_box import secret_decrypt, secret_encrypt
from model_zoo import get_model_zoo
from model_zoo_updater import get_zoo_updater
from model_onboarding import HFError, ModelOnboardingService
from finetuning import FineTuningService
from monitoring_layer import (
    fetch_all_zone_signals,
    fetch_grid_signal,
    fetch_system_metrics,
    find_low_carbon_window,
    get_zone_carbon_map,
    get_zone_forecast,
    infer_task_profile,
)
from routing_policies import (
    build_request_context,
    evaluate_ecoserve_actions,
    infer_prompt_profile,
    load_policy_config,
    load_routing_targets,
    rank_routing_candidates,
    CSS_REFERENCE_OUTPUT_TOKENS,
    safe_float,
    variant_capability_tier,
)
from rl_controller import get_rl_controller
from quality_latency_estimator import get_quality_latency_estimator
from multimodal import run_image_generation, run_vlm_inference
from nemo_guardrails import apply_guardrails, GUARDRAILS_ENABLED

import budgets as budgets_mod
import csrd_reporting
import semantic_cache
from tenancy import (
    DEFAULT_TENANT_ID,
    normalise_tenant_id,
    require_admin,
    resolve_tenant,
    tenant_metadata,
)
from fastapi import Depends, Header
from fastapi.responses import Response, StreamingResponse

try:
    from simpleeval import EvalWithCompoundTypes as _SimpleEvalCls
    _SIMPLEEVAL_AVAILABLE = True
except ImportError:
    _SIMPLEEVAL_AVAILABLE = False

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


BASE_DIR = Path(__file__).resolve().parent
PRIMARY_DATA_DIR = BASE_DIR / "data"
FALLBACK_DATA_DIR = Path(os.getenv("FALLBACK_DATA_DIR", "/tmp/green-ai"))
DATA_DIR = PRIMARY_DATA_DIR
CONFIG_DIR = BASE_DIR / "config"
POLICY_CONFIG_PATH = Path(os.getenv("POLICY_CONFIG_PATH", CONFIG_DIR / "policies.json"))
ROUTING_TARGETS_PATH = Path(os.getenv("ROUTING_TARGETS_PATH", CONFIG_DIR / "routing_targets.json"))

VLLM_MEDIUM_URL = os.getenv("VLLM_MEDIUM_URL", "http://127.0.0.1:8001/v1")
VLLM_FULL_URL = os.getenv("VLLM_FULL_URL", "http://127.0.0.1:8002/v1")
# Optional dedicated endpoints; default to VLLM_FULL_URL so an operator running
# only the 2-container default stack continues to work without changes.
VLLM_MOE_URL = os.getenv("VLLM_MOE_URL", VLLM_FULL_URL)
VLLM_STEM_MATH_URL = os.getenv("VLLM_STEM_MATH_URL", VLLM_FULL_URL)
VLLM_STEM_SCIENCE_URL = os.getenv("VLLM_STEM_SCIENCE_URL", VLLM_FULL_URL)
VLLM_STEM_CODING_URL = os.getenv("VLLM_STEM_CODING_URL", VLLM_FULL_URL)
# Llama2-7B always-available fallback (paper §4.3). Defaults to VLLM_FULL_URL —
# operators that spin up the dedicated CPU container should set VLLM_FALLBACK_URL.
VLLM_FALLBACK_URL = os.getenv("VLLM_FALLBACK_URL", VLLM_FULL_URL)
VLLM_TIMEOUT_SECONDS = int(os.getenv("VLLM_TIMEOUT_SECONDS", "45"))

# Multimodal NIM endpoints (pluggable; unset → graceful fallback in multimodal.py).
# Image analysis (VLM) and image generation (diffusion) NVIDIA NIM microservices.
NIM_VLM_URL = os.getenv("NIM_VLM_URL", "")
NIM_SDXL_URL = os.getenv("NIM_SDXL_URL", "")
NIM_FLUX_URL = os.getenv("NIM_FLUX_URL", "")

# Map env-var name → resolved URL for zoo-driven endpoint lookup.
_VLLM_ENDPOINTS_BY_ENV: dict[str, str] = {
    "VLLM_MEDIUM_URL":       VLLM_MEDIUM_URL,
    "VLLM_FULL_URL":         VLLM_FULL_URL,
    "VLLM_MOE_URL":          VLLM_MOE_URL,
    "VLLM_STEM_MATH_URL":    VLLM_STEM_MATH_URL,
    "VLLM_STEM_SCIENCE_URL": VLLM_STEM_SCIENCE_URL,
    "VLLM_STEM_CODING_URL":  VLLM_STEM_CODING_URL,
    "VLLM_FALLBACK_URL":     VLLM_FALLBACK_URL,
}

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5200"))   # ~1600 input tokens at 3.2 chars/token; fits in TinyLlama's 2048 ctx with output reserve. Operators can raise this when running with the full Qwen2.5/MoE container — but doing so will push longer chats onto the heaviest available model.
MAX_ATTACHMENT_BYTES = int(os.getenv("MAX_ATTACHMENT_BYTES", str(8 * 1024 * 1024)))
MAX_ATTACHMENT_EXTRACT_CHARS = int(os.getenv("MAX_ATTACHMENT_EXTRACT_CHARS", "12000"))
MAX_ATTACHMENTS_PER_MESSAGE = int(os.getenv("MAX_ATTACHMENTS_PER_MESSAGE", "6"))
DEFAULT_RAG_TOP_K = int(os.getenv("DEFAULT_RAG_TOP_K", "6"))

MODEL_NAME_MAP = {
    # ultra-light dispatches to VLLM_MEDIUM_URL (TinyLlama) in the default
    # 2-container stack — the historic "DialoGPT-medium" label was misleading.
    "ultra-light":  "TinyLlama-1.1B-Chat-v1.0",
    "medium":       "TinyLlama-1.1B-Chat-v1.0",
    "full":         "Qwen2.5-1.5B-Instruct",
    "moe":          "Qwen3-30B-A3B-MoE",
    "stem-math":    "Qwen2.5-Math-1.5B-Instruct",
    "stem-science": "Qwen2.5-1.5B-Instruct",
    "stem-coding":  "Qwen2.5-Coder-1.5B-Instruct",
}

# Default per-variant HF model IDs. The zoo entry's `vllm_model_id` overrides
# this when present (e.g., to swap MoE between Qwen3-30B-A3B and a smaller
# placeholder during dev).
VLLM_MODEL_MAP: dict[str, str] = {
    "ultra-light": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "medium":      "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "full":        "Qwen/Qwen2.5-1.5B-Instruct",
    "moe":         "Qwen/Qwen3-30B-A3B",
    "stem-math":   "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "stem-science":"Qwen/Qwen2.5-1.5B-Instruct",
    "stem-coding": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}

# Default per-variant endpoint. Each variant points at its dedicated env-var
# URL; if the operator hasn't set the dedicated env var, that var defaults to
# VLLM_FULL_URL (see resolution above), so single-container deploys still work.
VLLM_URL_MAP: dict[str, str] = {
    "ultra-light": VLLM_MEDIUM_URL,
    "medium":      VLLM_MEDIUM_URL,
    "full":        VLLM_FULL_URL,
    "moe":         VLLM_MOE_URL,
    "stem-math":   VLLM_STEM_MATH_URL,
    "stem-science":VLLM_STEM_SCIENCE_URL,
    "stem-coding": VLLM_STEM_CODING_URL,
}

STEM_VARIANT_FOR_DOMAIN: dict[str, str] = {
    "math":    "stem-math",
    "science": "stem-science",
    "coding":  "stem-coding",
}


# HMAC audit trail signing key (Section 3.6.1).
# Must be supplied via env in production; in dev a randomised per-process key is
# generated so audit signatures are still verifiable within the running process.
def _resolve_audit_hmac_key() -> bytes:
    raw = os.getenv("AUDIT_HMAC_KEY", "").strip()
    env_label = os.getenv("ENV", os.getenv("ENVIRONMENT", "dev")).strip().lower()
    if raw:
        if raw == "green-ai-audit-key-change-in-production":
            raise RuntimeError(
                "AUDIT_HMAC_KEY is set to the documented placeholder value. "
                "Replace it with a real secret (HSM-backed in production)."
            )
        return raw.encode("utf-8")
    if env_label in {"prod", "production"}:
        raise RuntimeError(
            "AUDIT_HMAC_KEY is required in production (paper §3.6.1: HSM-stored "
            "key). Refusing to start with an unsigned audit trail."
        )
    # Dev/test fallback: ephemeral key (per-process). Stable across requests in
    # this process; not durable across restarts.
    import secrets as _secrets
    return _secrets.token_bytes(32)


AUDIT_HMAC_KEY = _resolve_audit_hmac_key()


# Encryption key for secrets at rest — currently HF_TOKEN_ENC. Prefer a
# dedicated SECRET_KEY; WF_SECRET_KEY is still read so deployments that set it
# for the old workflow credentials keep working. With neither set (dev), the
# audit key is ephemeral per process, so encrypted values will not decrypt after
# a restart -- set SECRET_KEY in production.
def _resolve_secret_key() -> bytes:
    raw = os.getenv("SECRET_KEY", "").strip() or os.getenv("WF_SECRET_KEY", "").strip()
    return raw.encode("utf-8") if raw else AUDIT_HMAC_KEY


SECRET_KEY = _resolve_secret_key()

# Multi-region: fetch zone signals at startup + cache
MULTI_REGION_ENABLED = os.getenv("MULTI_REGION_ENABLED", "false").lower() in {"1", "true", "yes"}

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SUMMARY_KEYWORDS = {
    "summarize",
    "summary",
    "analyze",
    "analysis",
    "compare",
    "steps",
    "architecture",
    "design",
    "review",
    "root cause",
    "migration",
    "policy",
    "compliance",
    "walkthrough",
}
DOCUMENT_ANALYSIS_KEYWORDS = {
    "summarize",
    "summary",
    "analyze",
    "analysis",
    "compare",
    "architecture",
    "design",
    "review",
    "root cause",
    "migration",
    "policy",
    "compliance",
    "walkthrough",
}
STEP_KEYWORDS = {
    "step",
    "steps",
    "procedure",
    "runbook",
    "checklist",
    "action items",
    "implementation",
    "migration",
}
COMMON_QUERY_WORDS = {
    "a",
    "an",
    "and",
    "architecture",
    "be",
    "by",
    "can",
    "compare",
    "doc",
    "document",
    "documents",
    "explain",
    "file",
    "files",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "review",
    "show",
    "steps",
    "summarize",
    "summary",
    "tell",
    "the",
    "these",
    "this",
    "to",
    "use",
    "what",
    "with",
}
PROMPT_CACHE_MAX_ENTRIES = int(os.getenv("PROMPT_CACHE_MAX_ENTRIES", "128"))
HISTORY_SUMMARY_TRIGGER_MESSAGES = int(os.getenv("HISTORY_SUMMARY_TRIGGER_MESSAGES", "6"))
HISTORY_SUMMARY_KEEP_RECENT = int(os.getenv("HISTORY_SUMMARY_KEEP_RECENT", "4"))
HISTORY_SUMMARY_MAX_CHARS = int(os.getenv("HISTORY_SUMMARY_MAX_CHARS", "1400"))
PROMPT_CACHE: dict[str, dict[str, Any]] = {}
HISTORY_SUMMARY_CACHE: dict[str, dict[str, Any]] = {}
CACHE_LOCK = threading.Lock()


def prepare_writable_file(target: Path, fallback_dir: Path) -> Path:
    for candidate in (target, fallback_dir / target.name):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch(exist_ok=True)
            return candidate
        except OSError:
            continue
    return target


LOG_FILE = prepare_writable_file(
    Path(os.getenv("DECISION_LOG_PATH", PRIMARY_DATA_DIR / "decision_logs.jsonl")),
    FALLBACK_DATA_DIR,
)
STORE_PATH = prepare_writable_file(
    Path(os.getenv("CONVERSATION_STORE_PATH", PRIMARY_DATA_DIR / "green_ai.db")),
    FALLBACK_DATA_DIR,
)
RAG_STORE_PATH = prepare_writable_file(
    Path(os.getenv("RAG_STORE_PATH", PRIMARY_DATA_DIR / "rag_store.json")),
    FALLBACK_DATA_DIR,
)
DECISION_ENGINE_LOG = prepare_writable_file(
    Path(os.getenv("DECISION_ENGINE_LOG", PRIMARY_DATA_DIR / "decision_engine.log")),
    FALLBACK_DATA_DIR,
)
DATA_DIR = STORE_PATH.parent

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(DECISION_ENGINE_LOG),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

store = ConversationStore(STORE_PATH)
rag_service = AdvancedRAGService(
    RAG_STORE_PATH,
    chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
    chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "180")),
    dense_top_k=int(os.getenv("RAG_DENSE_TOP_K", "14")),
    rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "8")),
    context_char_limit=int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "5200")),
    embedding_dim=int(os.getenv("RAG_HASH_EMBED_DIM", "256")),
    embedding_model_name=os.getenv(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
    reranker_model_name=os.getenv(
        "RAG_RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
)


def _requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


http_session = _requests_session()
app = FastAPI(title="Adaptive Green AI API", version="3.0.0")

from tracing import setup_tracing
setup_tracing(app)

# FT-MoE reconciler tunables (Section 5.3)
MOE_RECONCILER_INTERVAL_S = float(os.getenv("MOE_RECONCILER_INTERVAL_S", "10"))
MOE_RECONCILER_REBALANCE_RATIO = float(os.getenv("MOE_RECONCILER_REBALANCE_RATIO", "0.75"))
MOE_RECONCILER_DISABLE_RATIO = float(os.getenv("MOE_RECONCILER_DISABLE_RATIO", "0.5"))
MOE_RECONCILER_ENABLED = os.getenv("MOE_RECONCILER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _moe_health_probe(model_id: str) -> tuple[int, int]:
    """
    Default health probe: returns the registry's last-known health. In a
    distributed deployment this is replaced with a real ping against expert
    replicas (Kubernetes endpoint scrape, vLLM /health on each shard, etc.).
    Falling back to last-known here means an externally-published health
    update via POST /api/model-zoo/{id}/expert-health remains the source of
    truth and the reconciler simply re-evaluates it on its cadence.
    """
    zoo = get_model_zoo()
    health = zoo.get_expert_health(model_id)
    model = zoo.get_model(model_id) or {}
    total = int(health.get("total_experts") or model.get("num_experts", 0))
    healthy = int(health.get("healthy_experts", total))
    return healthy, total


@app.on_event("startup")
async def _start_background_workers() -> None:
    if MOE_RECONCILER_ENABLED:
        get_model_zoo().start_health_reconciler(
            health_probe=_moe_health_probe,
            interval_seconds=MOE_RECONCILER_INTERVAL_S,
            rebalance_threshold=MOE_RECONCILER_REBALANCE_RATIO,
            disable_threshold=MOE_RECONCILER_DISABLE_RATIO,
        )
    # Model-zoo auto-updater. Off by default; only starts when
    # MODEL_ZOO_UPDATE_ENABLED=true and MODEL_ZOO_UPDATE_SOURCE is set.
    get_zoo_updater().start()
    # Re-export endpoint env vars for previously onboarded models. resolve_vllm_endpoint
    # reads them with os.getenv at lookup time, and an in-process env var does not
    # survive a restart — without this an onboarded model stays registered but
    # unreachable, which is the worst of both states.
    if os.getenv("MODEL_ONBOARD_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            restored = _get_onboarding_service().restore_endpoints()
            if restored:
                logger.info("Restored onboarded model endpoints: %s", ", ".join(restored))
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            logger.warning("Could not restore onboarded model endpoints: %s", exc)


@app.on_event("shutdown")
async def _stop_background_workers() -> None:
    get_model_zoo().stop_health_reconciler()
    get_zoo_updater().stop()


allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    query: str
    priority: str = ""
    mode: str = ""
    conversation_id: str | None = None
    task_profile: dict[str, Any] | None = None
    user_tier: str = "standard"
    accuracy_floor: float | None = None
    sla_ms: int | None = None
    deferral_tolerance_ms: int | None = None
    region_preference: str | None = None
    model_preference: str | None = None
    persist_attachments: bool = False
    top_k: int = DEFAULT_RAG_TOP_K


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trim_text(text: str, limit: int) -> str:
    cleaned = re.sub(r"\r\n?", "\n", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def calculate_system_co2(
    system_power_w: float,
    grid_carbon_intensity: float,
    duration_seconds: float = 1.0,
) -> float:
    energy_kwh = (system_power_w * duration_seconds) / (1000 * 3600)
    return energy_kwh * grid_carbon_intensity


#  Token counting
# Model max context windows (tokens)  used to clamp estimates.
# NOTE: ultra-light dispatches to the same TinyLlama container as medium
# (both share VLLM_MEDIUM_URL with a 2048 ctx). Keep their caps equal so the
# pre-dispatch escalation chain (line ~2768 _VARIANT_CAPS) doesn't bounce
# ultra-light → medium pointlessly when prompt size grows — both back-ends
# have the same headroom; if input overflows ultra-light it also overflows
# medium, and we must jump straight to full.
_MODEL_MAX_TOKENS: dict[str, int] = {
    "ultra-light": 2048,
    "medium":      2048,
    "full":        4096,
    "moe":         8192,
}
# Conservative bias: the 4 chars/token rule of thumb under-counts for
# markdown / code / multi-byte content by 10–20%, and a single under-count
# pushed by an accumulating conversation history triggers vLLM's 400
# "exceeds context length" error and the auto-escalate chain — which is the
# dominant reason long B2B conversations drift onto the heaviest model. Use
# 3.2 chars/token so the estimate over-counts slightly and any necessary
# trimming / variant escalation happens *before* dispatch, not after a 400.
_CHARS_PER_TOKEN = 3.2


def estimate_token_count(text: str, model_variant: str = "medium", clamp_to_cap: bool = True) -> int:
    """Conservative token estimate (~3.2 chars/token).

    By default the returned value is clamped at the model's context cap (used
    for token accounting / billing). Pass ``clamp_to_cap=False`` to get the
    raw count — needed by the overflow-escalation check, otherwise a 3000-
    token prompt looks like exactly 2048 to a 2048-ctx model and the gate
    never fires.
    """
    raw = max(1, int(len(text) / _CHARS_PER_TOKEN))
    if not clamp_to_cap:
        return raw
    cap = _MODEL_MAX_TOKENS.get(model_variant, 4096)
    return min(raw, cap)


#  GPU metrics helpers 

# Constants for GPU utilisation estimation formula
_TRANSFER_OVERHEAD_MS: float = 25.0   # memory transfer + kernel launch overhead (ms)

# VRAM footprints (GB) per model variant  weights + activations at fp16
_MODEL_VRAM_GB: dict[str, float] = {
    "ultra-light": 1.0,
    "medium":      4.0,
    "full":        8.0,
    "moe":        18.0,
}
_ASSUMED_GPU_VRAM_GB: float = float(os.getenv("GPU_VRAM_GB", "16"))

# Module-level cache for the inference-derived GPU util estimate
# Used by /api/system/metrics to show estimated GPU util even between requests
_gpu_util_estimate_lock = threading.Lock()
_gpu_util_estimate: float = 0.0
_gpu_util_estimate_ts: float = 0.0
_GPU_UTIL_ESTIMATE_TTL: float = 60.0   # seconds before estimate is considered stale


def _cache_gpu_util_estimate(value: float) -> None:
    global _gpu_util_estimate, _gpu_util_estimate_ts
    with _gpu_util_estimate_lock:
        _gpu_util_estimate = value
        _gpu_util_estimate_ts = time.monotonic()


def _get_cached_gpu_util_estimate() -> float | None:
    with _gpu_util_estimate_lock:
        if time.monotonic() - _gpu_util_estimate_ts > _GPU_UTIL_ESTIMATE_TTL:
            return None   # stale  don't show outdated data
        return _gpu_util_estimate


def estimate_gpu_utilization(
    actual_latency_ms: float,
    estimated_latency_ms: float,
    model_variant: str,
    queue_size: int = 0,
    used_memory_mb: float = 0.0,
    total_memory_mb: float = 0.0,
) -> float:
    """Estimate GPU utilisation when hardware counters return 0 or are unavailable.

    Formula
    -------
    GPU_UTIL_ESTIMATE = compute_ratio " transfer_factor   (Component 1)
                      + queue_factor                       (Component 2)
                      + memory_factor                      (Component 3)

    Component 1  Compute ratio
        slowdown_ratio = actual_latency_ms / estimated_latency_ms
        A slowdown of 1" means the model ran at expected GPU speed  high util.
        A slowdown of 10" means the model ran on CPU  low GPU util.
        compute_ratio = min(0.85 / slowdown_ratio, 0.85)

        transfer_factor = 1.0 - (TRANSFER_OVERHEAD_MS / actual_latency_ms)
        Represents the fraction of wall time NOT spent in host-GPU data transfers.

    Component 2  Queue delay factor
        queue_factor = 0.05 " min(queue_depth / 4, 1.0)
        Full queue (+4 pending) contributes +5% to estimated GPU pressure.

    Component 3  Memory pressure factor
        If measured VRAM data is available: mem_fraction = used_mb / total_mb
        Else: use model variant VRAM footprint relative to assumed GPU capacity.
        memory_factor = mem_fraction " 0.15   (up to +15%)

    Returns
    -------
    float  GPU utilisation percentage in [0.0, 100.0].
    """
    wall_ms  = max(actual_latency_ms,   1.0)
    est_ms   = max(estimated_latency_ms, 1.0)

    #  Component 1: inference compute ratio 
    slowdown = wall_ms / est_ms            # 1.0 = on-GPU speed; >1 = slower
    compute_ratio = min(0.85 / max(slowdown, 1.0), 0.85)
    transfer_factor = max(0.0, 1.0 - (_TRANSFER_OVERHEAD_MS / wall_ms))
    component1 = compute_ratio * transfer_factor

    #  Component 2: queue saturation pressure 
    component2 = 0.05 * min(queue_size / 4.0, 1.0)

    #  Component 3: VRAM memory pressure 
    if total_memory_mb > 0 and used_memory_mb > 0:
        mem_fraction = used_memory_mb / total_memory_mb
    else:
        model_vram_gb = _MODEL_VRAM_GB.get(model_variant, 4.0)
        mem_fraction  = min(model_vram_gb / _ASSUMED_GPU_VRAM_GB, 1.0)
    component3 = mem_fraction * 0.15

    raw = component1 + component2 + component3
    result = round(min(max(raw, 0.0), 1.0) * 100.0, 1)

    logger.debug(
        "GPU util estimate: %.1f%% "
        "(C1=%.3fx%.3f + C2=%.3f + C3=%.3f | slowdown=%.2fx queue=%d)",
        result, compute_ratio, transfer_factor, component2, component3,
        slowdown, queue_size,
    )
    return result


def extract_gpu_metrics(system_metrics: dict[str, Any]) -> dict[str, Any]:
    """Pull normalised GPU fields from the system-metrics sidecar payload."""
    util   = safe_float(system_metrics.get("system_gpu_utilization"), 0.0)
    power  = safe_float(system_metrics.get("system_gpu_PowerDraw"), 0.0)
    total_mem   = safe_float(system_metrics.get("system_gpu_TotalMemory"), 0.0)
    used_mem    = safe_float(system_metrics.get("system_gpu_UsedMemory"), 0.0)
    free_mem    = safe_float(system_metrics.get("system_gpu_MemoryFree"), 0.0)
    temperature = safe_float(system_metrics.get("system_gpu_CoreTemperature"), 0.0)
    perf_state  = system_metrics.get("system_gpu_PerformanceState", "unknown")
    mem_util_pct = round((used_mem / total_mem * 100) if total_mem > 0 else 0.0, 1)
    return {
        "utilization_pct":        round(util, 1),
        "utilization_source":     "hardware" if util > 0 else "none",
        "power_w":                round(power, 2),
        "total_memory_mb":        round(total_mem, 0),
        "used_memory_mb":         round(used_mem, 0),
        "free_memory_mb":         round(free_mem, 0),
        "memory_utilization_pct": mem_util_pct,
        "temperature_c":          round(temperature, 1),
        "performance_state":      perf_state,
        "gpu_available":          power > 0 or util > 0,
        "constrained":            util > 80.0,
    }


def compute_gpu_co2(
    gpu_power_w: float,
    grid_carbon_intensity: float,
    duration_seconds: float,
) -> float:
    """CO2 attributed specifically to GPU power during this inference call.

    Formula: E_kWh = (P_gpu_W " t_s) / 3_600_000
             CO2_g = E_kWh " CI (gCO/kWh)
    """
    if gpu_power_w <= 0 or duration_seconds <= 0:
        return 0.0
    energy_kwh = (gpu_power_w * duration_seconds) / (1000 * 3600)
    return energy_kwh * grid_carbon_intensity


def resolve_zoo_target(
    variant: str,
    hint_target_id: str | None = None,
    hardware: str | None = None,
) -> str | None:
    """Map a *served* model_variant back to a model-zoo target id.

    run_vllm_inference escalates internally and reports only the served variant;
    selected_candidate["target_id"] keeps pointing at the RANKED model. Ex-post
    carbon must be billed to the model that actually ran, so it needs the id.

    model_variant is not unique — "ultra-light" matches both local-vgpu-small and
    local-cpu-fallback, "full" matches both local-vgpu-full and the CPU Llama-2
    fallback — so hardware is used to disambiguate before falling back to the hint.
    """
    if not variant:
        return hint_target_id

    models = get_model_zoo().list_models()
    matches = [
        m for m in models
        if str(m.get("model_variant", "")).lower() == variant.lower()
        and str(m.get("region", "")).lower() == "local"
    ]
    if not matches:
        return hint_target_id
    if len(matches) > 1 and hardware:
        narrowed = [
            m for m in matches
            if str(m.get("hardware", "")).lower() == str(hardware).lower()
        ]
        if narrowed:
            matches = narrowed
    return str(matches[0].get("id")) or hint_target_id


def apply_gpu_routing_adjustment(
    ranked_candidates: list[dict[str, Any]],
    gpu_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """When GPU util >80 %, prefer lighter models by boosting their CSS score.

    Avoids scheduling heavy model inference onto a GPU that is already thermally
    or compute-saturated  which would degrade latency *and* efficiency.
    Heavier models (full, moe) get a CSS penalty; lighter ones get a bonus.
    """
    if not gpu_metrics.get("constrained"):
        return ranked_candidates   # no change needed

    util = gpu_metrics["utilization_pct"]
    pressure = (util - 80.0) / 20.0   # 0.0 at 80%, 1.0 at 100%
    LIGHT = {"ultra-light", "medium"}
    HEAVY = {"full", "moe"}

    adjusted: list[dict[str, Any]] = []
    for c in ranked_candidates:
        variant = c.get("model_variant", "")
        c = dict(c)
        if variant in LIGHT:
            c["css_score"] = min(1.0, c.get("css_score", 0.5) + 0.15 * pressure)
            c["gpu_routing_bonus"] = round(0.15 * pressure, 3)
        elif variant in HEAVY:
            c["css_score"] = max(0.0, c.get("css_score", 0.5) - 0.10 * pressure)
            c["gpu_routing_penalty"] = round(0.10 * pressure, 3)
        adjusted.append(c)

    # Re-sort after adjustment
    adjusted.sort(key=lambda x: x.get("css_score", 0.0), reverse=True)
    return adjusted


def build_conversation_title(prompt: str) -> str:
    compact = re.sub(r"\s+", " ", prompt).strip()
    if not compact:
        return "Adaptive Green AI chat"
    title = " ".join(compact.split()[:8]).strip(" .,:;")
    if len(compact) > len(title):
        title = f"{title}..."
    return title[:72]


def extract_text_from_bytes(
    filename: str,
    content_type: str,
    raw_bytes: bytes,
) -> tuple[str, str]:
    extension = Path(filename).suffix.lower()
    normalized_type = (content_type or mimetypes.guess_type(filename)[0] or "").lower()

    if normalized_type.startswith("text/") or extension in {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".csv",
        ".log",
        ".yaml",
        ".yml",
    }:
        try:
            return raw_bytes.decode("utf-8"), "text"
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="ignore"), "text"

    if extension == ".pdf" and PdfReader is not None:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(filter(None, pages)), "pdf"

    if extension == ".docx":
        return extract_text_from_docx(raw_bytes), "docx"

    return "", "metadata-only"


def extract_docx_paragraph(paragraph: ElementTree.Element) -> str:
    fragments: list[str] = []
    for node in paragraph.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            fragments.append(node.text)
        elif tag == "tab":
            fragments.append("\t")
        elif tag in {"br", "cr"}:
            fragments.append("\n")

    text = "".join(fragments)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_docx(raw_bytes: bytes) -> str:
    blocks: list[str] = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_bytes)
    body = root.find("w:body", WORD_NS)
    if body is None:
        return ""

    for child in list(body):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = extract_docx_paragraph(child)
            if paragraph:
                blocks.append(paragraph)
            continue

        if tag == "tbl":
            for row in child.findall("w:tr", WORD_NS):
                cells: list[str] = []
                for cell in row.findall("w:tc", WORD_NS):
                    cell_parts = [
                        extract_docx_paragraph(paragraph)
                        for paragraph in cell.findall("w:p", WORD_NS)
                    ]
                    cell_text = " ".join(part for part in cell_parts if part)
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    blocks.append(" | ".join(cells))

    return "\n\n".join(blocks).strip()


async def read_attachments(files: list[UploadFile] | None) -> list[dict[str, Any]]:
    if not files:
        return []

    if len(files) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise HTTPException(
            status_code=400,
            detail=f"Attach up to {MAX_ATTACHMENTS_PER_MESSAGE} files per request.",
        )

    attachments: list[dict[str, Any]] = []
    for upload in files:
        if not upload.filename:
            continue

        try:
            raw_bytes = await upload.read()
            size_bytes = len(raw_bytes)
            if size_bytes > MAX_ATTACHMENT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename} is too large. Limit is {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB.",
                )

            try:
                extracted_text, extraction_status = extract_text_from_bytes(
                    upload.filename,
                    upload.content_type or "",
                    raw_bytes,
                )
            except Exception as exc:
                logger.warning("Attachment extraction failed for %s: %s", upload.filename, exc)
        # Trigger async reload of the offline model
                extracted_text, extraction_status = "", "metadata-only"

            retrieval_text = trim_text(extracted_text, MAX_ATTACHMENT_EXTRACT_CHARS) if extracted_text else ""
            excerpt = trim_text(extracted_text, min(1400, MAX_ATTACHMENT_EXTRACT_CHARS)) if extracted_text else ""

            resolved_ctype = (
                upload.content_type
                or mimetypes.guess_type(upload.filename)[0]
                or "application/octet-stream"
            )
            # Images carry no extractable text — keep the bytes as a data URI so
            # the VLM dispatch can reason over them and the UI can re-display the
            # upload. (Non-image attachments keep the existing text path.)
            _is_image = resolved_ctype.lower().startswith("image/") or Path(
                upload.filename
            ).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            image_data_uri = None
            if _is_image:
                image_data_uri = (
                    f"data:{resolved_ctype};base64,"
                    + base64.b64encode(raw_bytes).decode("ascii")
                )

            attachments.append(
                {
                    "id": str(uuid4()),
                    "name": upload.filename,
                    "content_type": resolved_ctype,
                    "size_bytes": size_bytes,
                    "excerpt": excerpt,
                    "context_text": retrieval_text,
                    "extraction_status": extraction_status,
                    "is_image": _is_image,
                    "image_data_uri": image_data_uri,
                }
            )
        finally:
            await upload.close()

    return attachments


def attachment_to_rag_document(
    attachment: dict[str, Any],
    conversation_id: str | None,
) -> dict[str, Any] | None:
    text = (attachment.get("context_text") or attachment.get("excerpt") or "").strip()
    if not text:
        return None

    return {
        "name": attachment.get("name", "upload"),
        "text": text,
        "metadata": {
            "source_type": "attachment",
            "conversation_id": conversation_id,
            "content_type": attachment.get("content_type"),
            "tags": ["attachment", "chat-upload"],
        },
    }


def format_history_message(message: dict[str, Any]) -> str:
    role_label = "Assistant" if message.get("role") == "assistant" else "User"
    blocks = [f"{role_label}: {message.get('content', '').strip()}"]

    for attachment in message.get("attachments", []):
        if attachment.get("excerpt"):
            blocks.append(
                f"[Attachment: {attachment.get('name', 'upload')}]\n{attachment.get('excerpt')}"
            )

    return "\n".join(block for block in blocks if block.strip())


def prune_memory_cache(cache: dict[str, dict[str, Any]], max_entries: int) -> None:
    if len(cache) <= max_entries:
        return

    sorted_items = sorted(
        cache.items(),
        key=lambda item: safe_float(item[1].get("last_access"), 0.0),
    )
    for key, _value in sorted_items[: len(cache) - max_entries]:
        cache.pop(key, None)


def build_history_digest(messages: list[dict[str, Any]]) -> str:
    digest_payload = [
        {
            "id": message.get("id"),
            "role": message.get("role"),
            "content": trim_text(message.get("content", ""), 240),
            "attachments": [
                attachment.get("name")
                for attachment in message.get("attachments", [])
                if isinstance(attachment, dict)
            ],
        }
        for message in messages
    ]
    encoded = json.dumps(digest_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize_history_for_context(
    conversation_id: str,
    history: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if len(history) <= HISTORY_SUMMARY_TRIGGER_MESSAGES:
        return "", {
            "summary_used": False,
            "summary_strategy": "recent-window",
            "summary_cache_hit": False,
            "summarized_messages": 0,
            "recent_messages_kept": len(history),
        }

    older_messages = history[:-HISTORY_SUMMARY_KEEP_RECENT]
    if not older_messages:
        return "", {
            "summary_used": False,
            "summary_strategy": "recent-window",
            "summary_cache_hit": False,
            "summarized_messages": 0,
            "recent_messages_kept": len(history),
        }

    history_digest = build_history_digest(older_messages)
    cache_key = f"{conversation_id}:{history_digest}"

    with CACHE_LOCK:
        cached = HISTORY_SUMMARY_CACHE.get(cache_key)
        if cached:
            cached["last_access"] = time.time()
            return cached["summary"], {
                **cached["metadata"],
                "summary_cache_hit": True,
            }

    summary_lines = ["Compressed conversation context:"]
    for message in older_messages[-8:]:
        prefix = "User focus" if message.get("role") == "user" else "Assistant reply"
        content = trim_text(normalize_response_text(message.get("content", "")), 220)
        attachment_names = ", ".join(
            attachment.get("name", "upload")
            for attachment in message.get("attachments", [])
            if isinstance(attachment, dict) and attachment.get("name")
        )
        if attachment_names:
            content = f"{content} [files: {attachment_names}]"
        if content:
            summary_lines.append(f"- {prefix}: {content}")

    summary = trim_text("\n".join(summary_lines), HISTORY_SUMMARY_MAX_CHARS)
    metadata = {
        "summary_used": True,
        "summary_strategy": "heuristic-slm-style",
        "summary_cache_hit": False,
        "summarized_messages": len(older_messages),
        "recent_messages_kept": min(len(history), HISTORY_SUMMARY_KEEP_RECENT),
        "summary_chars": len(summary),
    }

    with CACHE_LOCK:
        HISTORY_SUMMARY_CACHE[cache_key] = {
            "summary": summary,
            "metadata": metadata,
            "last_access": time.time(),
        }
        prune_memory_cache(HISTORY_SUMMARY_CACHE, PROMPT_CACHE_MAX_ENTRIES)

    return summary, metadata


def build_prompt_cache_key(
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
    history_summary: str,
    recent_history: list[dict[str, Any]],
) -> str:
    payload = {
        "prompt": normalize_response_text(prompt),
        "history_summary": history_summary,
        "recent_history": [
            {
                "id": message.get("id"),
                "role": message.get("role"),
                "content": trim_text(message.get("content", ""), 180),
            }
            for message in recent_history[-HISTORY_SUMMARY_KEEP_RECENT:]
        ],
        "attachments": [
            {
                "name": attachment.get("name"),
                "excerpt": trim_text(attachment.get("excerpt", ""), 240),
                "content_type": attachment.get("content_type"),
            }
            for attachment in attachments
        ],
        "rag": {
            "search_mode": rag_result.get("search_mode"),
            "sources": [
                {
                    "chunk_id": source.get("chunk_id"),
                    "document_name": source.get("document_name"),
                    "score": source.get("score"),
                }
                for source in rag_result.get("sources", [])[:6]
            ],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_or_build_prompt_payload(
    history: list[dict[str, Any]],
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
    history_summary: str,
    stem_domain: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_key = build_prompt_cache_key(
        prompt,
        attachments,
        rag_result,
        history_summary,
        history,
    )

    with CACHE_LOCK:
        cached = PROMPT_CACHE.get(cache_key)
        if cached:
            cached["last_access"] = time.time()
            return dict(cached["payload"]), {
                "prompt_cache_hit": True,
                "prompt_cache_key": cache_key,
            }

    payload = build_prompt(
        history,
        prompt,
        attachments,
        rag_result,
        history_summary=history_summary,
        stem_domain=stem_domain,
    )

    with CACHE_LOCK:
        PROMPT_CACHE[cache_key] = {
            "payload": payload,
            "last_access": time.time(),
        }
        prune_memory_cache(PROMPT_CACHE, PROMPT_CACHE_MAX_ENTRIES)

    return payload, {
        "prompt_cache_hit": False,
        "prompt_cache_key": cache_key,
    }


_STEM_SYSTEM_ADDENDUM: dict[str, str] = {
    "math": (
        " You are a precise mathematics and engineering-math assistant. "
        "For every problem: (1) restate the given quantities with units, "
        "(2) state the governing equation or identity you will use, "
        "(3) show the substitution and algebra step-by-step, "
        "(4) carry units through the calculation, "
        "(5) box or label the final numeric answer with correct units and significant figures. "
        "Use LaTeX where it aids clarity (e.g. $\\sigma = F/A$). "
        "Re-check your arithmetic before finalising. If the problem is under-specified, "
        "say what assumption you are making."
    ),
    "science": (
        " You are a knowledgeable science assistant covering physics, chemistry, and biology. "
        "Cite relevant laws, equations, or mechanisms. Be precise with units and terminology."
    ),
    "coding": (
        " You are an expert in scientific computing and algorithms. "
        "Provide correct, efficient code with complexity analysis where relevant. "
        "Prefer Python with numpy/scipy for numerical tasks."
    ),
}


def build_prompt(
    history: list[dict[str, Any]],
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
    history_summary: str = "",
    stem_domain: str | None = None,
) -> dict[str, Any]:
    normalized_prompt = (prompt or "").strip().lower()
    summary_like_request = any(keyword in normalized_prompt for keyword in SUMMARY_KEYWORDS)
    grounded_request = bool(attachments) or bool(rag_result.get("retrieved_count"))

    system_prompt = (
        "You are a helpful assistant. Use the retrieved context when relevant. "
        "Answer directly and concisely."
    )
    if stem_domain:
        system_prompt += _STEM_SYSTEM_ADDENDUM.get(stem_domain, "")
    if grounded_request:
        system_prompt += (
            " Answer only from the retrieved context and attachments. "
            "Say so clearly if the evidence does not support a claim."
        )
    if attachments and summary_like_request:
        system_prompt += (
            " For document summaries, produce 4 to 7 grounded bullet points or numbered steps."
        )

    rag_context = rag_result.get("context") or ""

    # -- RAG relevance gate -----------------------------------------------
    # Only inject RAG context when the retrieved chunks are actually relevant
    # to the current query. Without this gate, asking "arithmetic table of 12"
    # after a docx upload injects k3s chunks and causes total context poisoning.
    #
    # Rule: if ALL sources have rerank_score < RAG_MIN_RELEVANCE_SCORE AND
    # there is no attachment in the CURRENT turn, suppress rag_context.
    _RAG_MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.13"))
    if rag_context and not attachments:
        _rag_sources = rag_result.get("sources", [])
        if _rag_sources:
            _max_score = max(
                (s.get("score", 0.0) for s in _rag_sources),
                default=0.0,
            )
            if _max_score < _RAG_MIN_RELEVANCE_SCORE:
                rag_context = ""   # suppress low-relevance context
    attachment_manifest = "\n".join(
        f"- {attachment.get('name')} ({attachment.get('extraction_status')})"
        for attachment in attachments
    )
    attachment_excerpt_budget = min(
        max(1800, MAX_CONTEXT_CHARS // 4),
        max(2400, MAX_ATTACHMENT_EXTRACT_CHARS * max(len(attachments), 1)),
    )
    attachment_excerpt_blocks: list[str] = []
    attachment_excerpt_chars = 0
    for attachment in attachments:
        attachment_text = (
            attachment.get("context_text")
            or attachment.get("excerpt")
            or ""
        ).strip()
        if not attachment_text:
            continue
        block = (
            f"[Attachment: {attachment.get('name', 'upload')}]\n"
            f"{attachment_text}"
        )
        if (
            attachment_excerpt_blocks
            and attachment_excerpt_chars + len(block) > attachment_excerpt_budget
        ):
            break
        attachment_excerpt_blocks.append(block)
        attachment_excerpt_chars += len(block)

    reserved = (
        len(system_prompt)
        + len(prompt)
        + len(rag_context)
        + len(attachment_manifest)
        + len(history_summary)
        + attachment_excerpt_chars
        + 1500
    )
    history_budget = max(2400, MAX_CONTEXT_CHARS - reserved)
    selected_blocks: list[str] = []
    consumed_chars = 0

    for message in reversed(history):
        block = format_history_message(message)
        if not block:
            continue
        if selected_blocks and consumed_chars + len(block) > history_budget:
            break
        selected_blocks.append(block)
        consumed_chars += len(block)

    selected_blocks.reverse()

    sections = [system_prompt]
    if rag_context:
        sections.append(f"Retrieved context:\n{rag_context}")
    if attachment_manifest:
        sections.append(f"Current attachments:\n{attachment_manifest}")
    if attachment_excerpt_blocks:
        sections.append(
            "Current attachment excerpts:\n" + "\n\n".join(attachment_excerpt_blocks)
        )
    if history_summary:
        sections.append(history_summary)
    if selected_blocks:
        sections.append("Conversation so far:\n" + "\n\n".join(selected_blocks))
    sections.append(f"Current user request:\n{prompt.strip()}\n\nAssistant:")

    return {
        "prompt": "\n\n".join(sections),
        "context_messages_used": len(selected_blocks),
        "context_attachments_used": len(attachments),
        "retrieved_chunks_used": rag_result.get("retrieved_count", 0),
        "rag_context_characters": rag_result.get("context_characters", 0),
    }


def unwrap_output_data(output_data: Any) -> str:
    if output_data is None:
        return ""
    if isinstance(output_data, list):
        if not output_data:
            return ""
        return unwrap_output_data(output_data[0])
    if isinstance(output_data, bytes):
        return output_data.decode("utf-8", errors="ignore")
    return str(output_data)


def prompt_requests_high_quality_response(prompt: str, attachments: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    normalized_prompt = (prompt or "").strip().lower()
    reasons: list[str] = []

    if attachments:
        reasons.append("attachments_present")
    if len(normalized_prompt) > 280 or len(normalized_prompt.split()) > 40:
        reasons.append("long_prompt")
    if any(keyword in normalized_prompt for keyword in SUMMARY_KEYWORDS):
        reasons.append("summary_or_analysis_request")
    if sum(len(attachment.get("context_text") or "") for attachment in attachments) > 1800:
        reasons.append("large_attachment_context")

    return bool(reasons), reasons


def looks_low_quality_response(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True

    words = re.findall(r"[A-Za-z][A-Za-z0-9/\-]*", stripped)
    # Brevity check: only flag responses that are *truly* too short to convey
    # an answer. The old < 18 words / < 120 chars threshold rejected perfectly
    # fine sentences like "The capital of France is Paris." (6 words / 32
    # chars) and forced the auto-escalate-to-full retry path to fire, sending
    # every short-answer prompt to Qwen2.5-1.5B regardless of CSS routing.
    # A real low-quality response is empty, a single token, or a fragment —
    # not a complete one-sentence answer.
    if len(words) < 4 and len(stripped) < 20:
        return True

    short_word_ratio = sum(1 for word in words if len(word) <= 3) / max(len(words), 1)
    upper_word_ratio = sum(1 for word in words if word.isupper() and len(word) > 1) / max(len(words), 1)
    unique_ratio = len({word.lower() for word in words}) / max(len(words), 1)
    punctuation_clusters = len(re.findall(r"[<>]{2,}|[/|]{2,}", stripped))

    # Ratio checks are only meaningful on substantial responses — applying
    # them to one-sentence answers produces false positives (a 5-word answer
    # has near-100% unique_ratio but trivially clears 0.45, while a 7-word
    # acronym-heavy reply trips upper_word_ratio for no good reason).
    if len(words) >= 18 and (
        short_word_ratio > 0.58
        or upper_word_ratio > 0.22
        or unique_ratio < 0.45
    ):
        return True
    if punctuation_clusters > 0:
        return True

    # Hallucination detector — patterns live at module scope (see
    # _HALLUCINATION_PATTERNS below) so response_has_quality_red_flag can
    # share the same list without duplication.
    _lower = stripped.lower()
    _hallucination_hits = sum(
        1 for pat in _HALLUCINATION_PATTERNS
        if re.search(pat, _lower)
    )
    if _hallucination_hits >= 1:
        logger.info(
            "Hallucination detected (%d pattern hits) in response (first 120 chars): %s",
            _hallucination_hits, stripped[:120],
        )
        return True

    return False


def response_has_quality_red_flag(text: str) -> bool:
    """
    Stricter, narrower companion to :py:func:`looks_low_quality_response` used
    by the auto-escalate-to-full retry path. Returns True only when the
    response shows *positive* signs of being broken (empty, single fragment,
    garbled punctuation, hallucination markers) — not merely short.

    Why a separate function: ``looks_low_quality_response`` also drives the
    final safety net + extractive fallback, both of which want to catch
    moderately suspicious output. The retry path is far more expensive
    (re-runs inference on the largest available model, contradicts green
    routing) and must only fire when the small-model output is clearly
    unusable. Anti-green escalation on a one-sentence answer that happens to
    be < 18 words is exactly the failure mode the routing memo flags.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True

    words = re.findall(r"[A-Za-z][A-Za-z0-9/\-]*", stripped)
    if len(words) < 3:
        return True

    if re.search(r"[<>]{2,}|[/|]{2,}", stripped):
        return True

    # Unique-ratio repetition check: only fires for *severely* degenerate
    # output ("the the the…" loops, unique_ratio ≈ 0.05). A small chat model
    # answering a comparison prompt ("compare B2B vs B2C") naturally reiterates
    # the comparands and produces unique_ratio ≈ 0.35–0.45 — that is verbose
    # style, not breakage. The previous 0.45 threshold treated those as broken
    # and auto-escalated every analysis-style prompt to the full (Qwen2.5-1.5B)
    # model, then the cache pinned the served-model label so subsequent B2B
    # questions appeared "stuck on Qwen2.5". 0.25 still catches real loops.
    unique_ratio = len({w.lower() for w in words}) / max(len(words), 1)
    if len(words) >= 24 and unique_ratio < 0.25:
        return True

    _lower = stripped.lower()
    for pat in _HALLUCINATION_PATTERNS:
        if re.search(pat, _lower):
            return True

    return False


_HALLUCINATION_PATTERNS = [
    r"click\s+(?:on\s+)?the\s+.{0,30}\b(?:button|tab|menu|bar|icon|link)",
    r"open\s+the\s+(?:adaptive|green|ai)\b",
    r"adaptivegreen\s+ai",
    r"to handle the current(?: user\S*)? request",
    r"retrieved knowledge\s*(?:base|context)",
    r"clean factually? (?:accurate )?(?:summary|answer)",
    r"fact based answer",
    r"adapt[a-z]*\s*green\s*(?:ai)?",
    r"common mistakes to (?:avoid|prevent)",
    r"clean factual summary",
    r"not raw notes",
    r"kamasutras?",
    r"to handle the current request",
    r"carbon.?aware retri[a-z]*\s+and routing",
    r"^response:\s+(?:sure|here'?s|of\s+course|certainly|absolutely)[!,]",
    r"user\s*\(user\)\s*:",
    r"assistant\s*\(adaptive",
    r"user\s*\(assistant\)",
    r"\|user\|assistant",
    r"assistantuser|assistant,user",
    r"\|assistant,\|user,\|",
    r"type\s+.{0,30}into\s+the\s+search\s+bar",
    r"adaptive\s+gree?ne?s?\s+ai\b",
    r"adaptive\s+gree?ne?s?\s+ai.{0,30}(?:carbon.efficient|most\s+sustainable)",
    r"agga'?s?\s+(?:retrieval|interface|platform|ai)",
    r"the\s+trit(?:on|e|a)\.?\s+the\s+trit",
    r"theater\s+generation\s+backend",
    r"transter",
    r"model_repositories",
    r"select\s+.{0,40}from\s+the\s+drop.?down",
    r"\bsearch\s+(?:bar|results)\s+tab",
    r"in\s+the\s+top\s+(?:right|left)\s+corner\s+of\s+your\s+(?:screen|computer)",
    r"\bai\s+button\b",
    r"press\s+enter\s+to\s+see\s+all\s+available\s+commands",
]


def normalize_response_text(text: str) -> str:
    normalized = re.sub(r"\r\n?", "\n", text or "")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


_SAFE_MATH_NAMES = {
    "sin": _math_module.sin, "cos": _math_module.cos, "tan": _math_module.tan,
    "asin": _math_module.asin, "acos": _math_module.acos, "atan": _math_module.atan,
    "sqrt": _math_module.sqrt, "log": _math_module.log, "log2": _math_module.log2,
    "log10": _math_module.log10, "exp": _math_module.exp, "abs": abs,
    "ceil": _math_module.ceil, "floor": _math_module.floor, "round": round,
    "pi": _math_module.pi, "e": _math_module.e,
    "factorial": _math_module.factorial, "gcd": _math_module.gcd,
    "pow": pow,
}
_ARITH_SAFE_RE = re.compile(
    r"^[\d\s\+\-\*\/\(\)\.\%\^]+$"
    r"|^(?:(?:sin|cos|tan|asin|acos|atan|sqrt|log|log2|log10|exp|abs|ceil|floor|round|factorial|gcd|pow)"
    r"\s*\([\d\s\+\-\*\/\.\,]+\)\s*[\+\-\*\/\^]?\s*)+[\d\s\+\-\*\/\.\^]*$"
)


def _normalize_natural_language_math(expr: str) -> str:
    """Convert common natural-language math phrasings to function form.

    Bridges the gap between how users actually phrase math questions and the
    functional grammar :py:func:`_try_arithmetic` expects. Without this,
    "square root of 12794587" falls through to the LLM, which then fumbles a
    7-digit sqrt — sending a deterministically-answerable arithmetic question
    through a 1.5B-parameter model is both wasteful (carbon) and unreliable
    (math models routinely answer 7+ digit sqrts wrong).

    Transformations are applied in order; each leaves the expression in a
    form _try_arithmetic can parse. Operates on lowercased text; the caller
    re-evaluates after normalisation.
    """
    s = expr.lower().strip()
    # "N to the power of M" / "N raised to (the power of) M" — must run
    # before the bare "the" strip below, otherwise "to the power of" becomes
    # "to power of" and the pattern no longer matches.
    s = re.sub(
        r"\b([\d.\-]+)\s+(?:to\s+the\s+power\s+of|raised\s+to(?:\s+the\s+power\s+of)?)\s+([\d.\-]+)",
        r"(\1)**(\2)",
        s,
    )
    # "the X of Y" framing (e.g. "the square root of 25")
    s = re.sub(r"\bthe\s+", "", s)
    # Unary-prefix forms: "<func> of N" → "<func>(N)". Only matches when N
    # is a numeric literal — alphabetic tails would push us into the LLM
    # path anyway. Order matters: longer keywords first ("cube root" before
    # "cube"), and "square root" must precede the bare "square" rule.
    _unary_prefix_map = [
        (r"\b(?:square\s+root|sqrt)\s+of\s+([\d.\-]+)", r"sqrt(\1)"),
        (r"\b(?:cube\s+root|cbrt)\s+of\s+([\d.\-]+)", r"(\1)**(1/3)"),
        (r"\b(?:natural\s+log|ln)\s+of\s+([\d.\-]+)", r"log(\1)"),
        (r"\blog(?:arithm)?\s+(?:base\s*10\s+)?of\s+([\d.\-]+)", r"log10(\1)"),
        (r"\blog2\s+of\s+([\d.\-]+)", r"log2(\1)"),
        (r"\b(?:absolute\s+value|abs)\s+of\s+([\d.\-]+)", r"abs(\1)"),
        (r"\b(?:factorial)\s+of\s+([\d]+)", r"factorial(\1)"),
        (r"\b(?:sin|sine)\s+of\s+([\d.\-]+)", r"sin(\1)"),
        (r"\b(?:cos|cosine)\s+of\s+([\d.\-]+)", r"cos(\1)"),
        (r"\b(?:tan|tangent)\s+of\s+([\d.\-]+)", r"tan(\1)"),
    ]
    for pat, repl in _unary_prefix_map:
        s = re.sub(pat, repl, s)
    # Postfix forms: "N squared", "N cubed", "N factorial"
    s = re.sub(r"\b([\d.\-]+)\s+squared\b", r"(\1)**2", s)
    s = re.sub(r"\b([\d.\-]+)\s+cubed\b", r"(\1)**3", s)
    s = re.sub(r"\b([\d]+)\s+factorial\b", r"factorial(\1)", s)
    # English operators
    s = re.sub(r"\bplus\b", "+", s)
    s = re.sub(r"\bminus\b", "-", s)
    s = re.sub(r"\b(?:times|multiplied\s+by)\b", "*", s)
    s = re.sub(r"\bdivided\s+by\b", "/", s)
    s = re.sub(r"\bmodulo\b", "%", s)
    return s.strip()


def _try_arithmetic(text: str) -> str | None:
    """Evaluate a simple arithmetic expression and return a formatted answer.

    Handles expressions like: 2+5, 10*7, sqrt(16), 2^8, sin(0), as well as
    natural-language phrasings ("square root of 16", "12 plus 7", "5
    factorial", "3 to the power of 4") via :py:func:`_normalize_natural_language_math`.
    Returns None if the expression is not recognisably arithmetic.
    """
    raw = (text or "").strip()
    # Strip trailing =? or = characters and whitespace
    expr = re.sub(r"[\s=?!]+$", "", raw).strip()
    if not expr:
        return None

    # Natural-language → functional form ("square root of N" → "sqrt(N)")
    expr = _normalize_natural_language_math(expr)
    # Normalise operators
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    # Reject if it still has alphabetic tokens that aren't known math functions
    unknown_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)
    if any(t not in _SAFE_MATH_NAMES for t in unknown_tokens):
        return None
    # Must have at least one digit
    if not re.search(r"\d", expr):
        return None

    try:
        if _SIMPLEEVAL_AVAILABLE:
            _evaluator = _SimpleEvalCls(
                names=_SAFE_MATH_NAMES,
                functions=_SAFE_MATH_NAMES,
            )
            result = _evaluator.eval(expr)
        else:
            result = eval(  # noqa: S307
                compile(expr, "<expr>", "eval"),
                {"__builtins__": {}},
                _SAFE_MATH_NAMES,
            )
        if not isinstance(result, (int, float)):
            return None
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            result = int(result)
        display_expr = re.sub(r"\*\*", "^", expr)
        if isinstance(result, float):
            # 12 significant figures — enough for any reasonable arithmetic
            # answer without dumping floating-point noise. sqrt(12794587)
            # rendered at .6g was "3576.95"; users expect 3576.95219426.
            formatted = f"{result:.12g}"
        else:
            formatted = str(result)
        return f"{display_expr} = {formatted}"
    except Exception:
        return None


def extract_query_terms(prompt: str) -> set[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9/\-]{2,}", (prompt or "").lower())
    return {
        token
        for token in candidates
        if token not in COMMON_QUERY_WORDS and not token.isdigit()
    }


def is_informative_segment(segment: str) -> bool:
    if len(segment) < 42:
        return False
    alpha_chars = re.findall(r"[A-Za-z]", segment)
    if len(alpha_chars) < 24:
        return False
    if re.fullmatch(r"[-=:_./\\| ]+", segment):
        return False
    if segment.count(" ") < 6:
        return False
    return True


def collect_attachment_segments(
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_segments(text: str, source_name: str) -> None:
        nonlocal segments
        for raw_segment in re.split(r"\n{2,}", text or ""):
            cleaned = normalize_response_text(raw_segment)
            cleaned = trim_text(cleaned, 280)
            if not is_informative_segment(cleaned):
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            segments.append({"text": cleaned, "source_name": source_name})

    for attachment in attachments:
        source_name = attachment.get("name", "upload")
        add_segments(attachment.get("context_text") or "", source_name)
        add_segments(attachment.get("excerpt") or "", source_name)

    for source in rag_result.get("sources", [])[:6]:
        text = source.get("text") or source.get("content") or ""
        source_name = source.get("document_name") or "retrieved context"
        add_segments(text, source_name)

    return segments


def score_attachment_segment(
    segment: dict[str, Any],
    query_terms: set[str],
    position: int,
) -> float:
    text = segment.get("text", "")
    lowered = text.lower()
    score = max(0.0, 8.0 - position * 0.35)
    score += sum(1.0 for term in query_terms if term in lowered)

    if ":" in text and len(text.split(":", 1)[0]) < 60:
        score += 0.4
    if any(char.isdigit() for char in text):
        score += 0.25
    if text[0].isupper():
        score += 0.15
    if len(text) > 240:
        score -= 0.2

    return score


def build_attachment_fallback_summary(
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
) -> str:
    """Extractive summary from uploaded files + RAG retrieved chunks.

    Respects "N lines only", "N points", "N sentences" requests by
    capping the number of bullet points to the requested count.
    Also uses RAG chunks directly when parsed_attachments is empty but
    the retrieval pipeline fetched relevant content.
    """
    segments = collect_attachment_segments(attachments, rag_result)

    # If no segments from attachments, fall back to RAG retrieval chunks
    if not segments:
        rag_chunks = rag_result.get("chunks") or rag_result.get("results") or []
        for chunk in rag_chunks:
            text = chunk.get("text") or chunk.get("content") or ""
            if text.strip():
                segments.append({"text": text.strip(), "source": chunk.get("source", "Retrieved context")})

    if not segments:
        return ""

    # Detect "N lines only" / "N points" / "N sentences" constraint.
    # IMPORTANT: search only the CURRENT user request, not the full prompt
    # which may include previous turns that requested e.g. "4 lines" or
    # "summarise in 5 points" " those would contaminate the current request.
    _USER_REQUEST_MARKER = "Current user request:"
    _ASSISTANT_MARKER    = "\nAssistant:"
    _current_request = prompt or ""
    if _USER_REQUEST_MARKER in _current_request:
        _after = _current_request.split(_USER_REQUEST_MARKER, 1)[-1]
        if _ASSISTANT_MARKER in _after:
            _after = _after.split(_ASSISTANT_MARKER, 1)[0]
        _current_request = _after.strip()

    prompt_lower_cur = _current_request.lower()
    _LINE_COUNT_PATTERN = re.compile(
        r"(?:in|only|just)?\s*(\d+)\s*(?:lines?|points?|bullets?|sentences?|items?)"
    )
    max_bullets = 6   # default
    _match = _LINE_COUNT_PATTERN.search(prompt_lower_cur)
    if _match:
        requested = int(_match.group(1))
        max_bullets = max(1, min(requested, 12))   # clamp 1-12

    query_terms = extract_query_terms(prompt)
    ranked_segments = sorted(
        enumerate(segments),
        key=lambda item: (
            -score_attachment_segment(item[1], query_terms, item[0]),
            item[0],
        ),
    )

    chosen_segments: list[dict[str, Any]] = []
    for original_index, segment in ranked_segments:
        if len(chosen_segments) >= max_bullets:
            break
        if chosen_segments and segment["text"] == chosen_segments[-1]["text"]:
            continue
        chosen_segments.append(segment)

    if not chosen_segments:
        return ""

    wants_steps = any(keyword in prompt_lower_cur for keyword in STEP_KEYWORDS)
    source_names = ", ".join(
        dict.fromkeys(
            attachment.get("name", "upload") for attachment in attachments if attachment.get("name")
        )
    )
    if not source_names:
        source_names = "the uploaded document"

    intro = (
        f"Practical steps from {source_names}:"
        if wants_steps and source_names != "the uploaded document"
        else "Practical steps from the uploaded document:"
        if wants_steps
        else f"Summary from {source_names} ({len(chosen_segments)} {'line' if len(chosen_segments)==1 else 'lines'}):"
        if source_names != "the uploaded document"
        else f"Summary from the uploaded document ({len(chosen_segments)} {'line' if len(chosen_segments)==1 else 'lines'}):"
    )
    bullet_prefixes = [f"{index}. " if wants_steps else "- " for index in range(1, len(chosen_segments) + 1)]
    lines = [intro]

    for prefix, segment in zip(bullet_prefixes, chosen_segments, strict=False):
        line = normalize_response_text(segment["text"])
        lines.append(f"{prefix}{line}")

    return "\n".join(lines)


def assess_evidence_sufficiency(
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
    semantic_profile: dict[str, Any],
) -> dict[str, Any]:
    query_terms = extract_query_terms(prompt)
    sources = rag_result.get("sources", [])[:6]
    attachment_text = "\n".join(
        (attachment.get("context_text") or attachment.get("excerpt") or "").strip()
        for attachment in attachments
        if (attachment.get("context_text") or attachment.get("excerpt"))
    )
    source_text = "\n".join(
        (source.get("text") or source.get("excerpt") or "").strip()
        for source in sources
        if (source.get("text") or source.get("excerpt"))
    )
    combined_evidence = "\n".join(part for part in (attachment_text, source_text) if part).strip()
    evidence_terms = extract_query_terms(combined_evidence)
    supported_terms = sorted(query_terms & evidence_terms)
    coverage_ratio = len(supported_terms) / max(len(query_terms), 1)
    top_source_score = max((safe_float(source.get("score"), 0.0) for source in sources), default=0.0)
    avg_source_score = (
        sum(safe_float(source.get("score"), 0.0) for source in sources) / max(len(sources), 1)
        if sources
        else 0.0
    )
    grounded_request = bool(attachments) or (
        rag_result.get("retrieved_count", 0) > 0
        and semantic_profile.get("intent") in {"summarization", "analysis", "implementation", "troubleshooting"}
    )

    if not combined_evidence:
        strength = "none"
    elif coverage_ratio >= 0.55 or top_source_score >= 0.85:
        strength = "strong"
    elif coverage_ratio >= 0.3 or top_source_score >= 0.4 or avg_source_score >= 0.28:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "grounded_request": grounded_request,
        "query_terms": sorted(query_terms),
        "supported_terms": supported_terms,
        "coverage_ratio": round(coverage_ratio, 4),
        "top_source_score": round(top_source_score, 4),
        "average_source_score": round(avg_source_score, 4),
        "retrieved_count": rag_result.get("retrieved_count", 0),
        "evidence_strength": strength,
        "combined_evidence_characters": len(combined_evidence),
        "allow_model_answer": (not grounded_request) or strength in {"moderate", "strong"},
    }


def response_supported_by_evidence(
    prompt: str,
    response: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    evidence_segments = collect_attachment_segments(attachments, rag_result)
    if not evidence_segments:
        return False, {
            "supported": False,
            "reason": "no_evidence_segments",
            "response_support_ratio": 0.0,
            "query_support_ratio": 0.0,
        }

    evidence_text = "\n".join(segment["text"] for segment in evidence_segments)
    evidence_terms = extract_query_terms(evidence_text)
    response_terms = extract_query_terms(response)
    query_terms = extract_query_terms(prompt)

    response_support = len(response_terms & evidence_terms) / max(len(response_terms), 1)
    # When the user prompt has ≤2 specific terms (e.g. "summarize this document"
    # → all stopwords), the query-support gate becomes meaningless: there are
    # no terms to match. In that case only the response-support gate governs,
    # since a model summary that overlaps with evidence is by definition grounded.
    if len(query_terms) <= 2:
        query_support = 1.0
        supported = response_support >= 0.14
        reason_unsupported = "low_response_overlap"
    else:
        query_support = len(query_terms & evidence_terms) / max(len(query_terms), 1)
        supported = response_support >= 0.14 and query_support >= 0.2
        reason_unsupported = "low_evidence_overlap"

    return supported, {
        "supported": supported,
        "reason": "supported" if supported else reason_unsupported,
        "response_support_ratio": round(response_support, 4),
        "query_support_ratio": round(query_support, 4),
        "evidence_segment_count": len(evidence_segments),
    }


def build_insufficient_evidence_response(
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
    evidence_assessment: dict[str, Any],
) -> str:
    source_names = list(
        dict.fromkeys(
            [
                attachment.get("name", "upload")
                for attachment in attachments
                if attachment.get("name")
            ]
            + [
                source.get("document_name", "retrieved context")
                for source in rag_result.get("sources", [])[:3]
                if source.get("document_name")
            ]
        )
    )
    source_label = ", ".join(source_names[:3]) if source_names else "the current evidence"
    request_hint = trim_text(prompt, 140)
    return (
        f'I do not have enough grounded evidence to answer "{request_hint}" reliably from {source_label}.\n\n'
        f"- Retrieved evidence strength: {evidence_assessment.get('evidence_strength', 'unknown')}\n"
        f"- Query-term coverage in evidence: {evidence_assessment.get('coverage_ratio', 0.0):.2f}\n"
        "- Try asking for a summary of the uploaded file, a quote from a specific section, or upload a more relevant document."
    )


def build_safe_grounded_fallback(
    prompt: str,
    attachments: list[dict[str, Any]],
    rag_result: dict[str, Any],
    evidence_assessment: dict[str, Any],
) -> tuple[str, str]:
    fallback_summary = build_attachment_fallback_summary(prompt, attachments, rag_result)
    if fallback_summary:
        return fallback_summary, "extractive-evidence"
    return (
        build_insufficient_evidence_response(
            prompt,
            attachments,
            rag_result,
            evidence_assessment,
        ),
        "insufficient-evidence",
    )


def apply_quality_guardrails(
    request_context: dict[str, Any],
    prompt: str,
    attachments: list[dict[str, Any]],
    persist_attachments: bool,
) -> list[str]:
    normalized_prompt = (prompt or "").strip().lower()
    semantic_profile = request_context.get("semantic_profile") or {}
    intent = str(semantic_profile.get("intent", "")).lower()
    recommended_model_variant = str(
        semantic_profile.get("recommended_model_variant", "")
    ).lower()
    attachment_chars = int(safe_float(semantic_profile.get("attachment_characters"), 0))
    document_heavy = bool(attachments) and (
        persist_attachments
        or attachment_chars > 900
        or any(keyword in normalized_prompt for keyword in DOCUMENT_ANALYSIS_KEYWORDS)
    )

    reasons: list[str] = []
    requires_high_quality, high_quality_reasons = prompt_requests_high_quality_response(
        prompt,
        attachments,
    )
    if document_heavy:
        reasons.extend(high_quality_reasons)

    if document_heavy:
        reasons.append("document-grounded request")
    if persist_attachments and attachments:
        reasons.append("persistent_indexing_enabled")
    if recommended_model_variant == "full" and document_heavy:
        reasons.append("semantic-full-route")
    elif intent in {"analysis", "implementation", "troubleshooting"}:
        reasons.append("reasoning-oriented request")

    high_risk_grounded = document_heavy and (
        requires_high_quality
        or persist_attachments
        or attachment_chars > 4_000
        or intent in {"analysis", "implementation", "troubleshooting"}
    )

    if document_heavy:
        if not request_context.get("model_preference") and high_risk_grounded:
            request_context["model_preference"] = "full"
        request_context["accuracy_floor"] = max(
            safe_float(request_context.get("accuracy_floor"), 0.0),
            0.9 if high_risk_grounded else 0.82,
        )
    elif intent in {"analysis", "implementation", "troubleshooting"}:
        request_context["accuracy_floor"] = max(
            safe_float(request_context.get("accuracy_floor"), 0.0),
            0.78,
        )
    elif recommended_model_variant == "medium":
        request_context["accuracy_floor"] = max(
            safe_float(request_context.get("accuracy_floor"), 0.0),
            0.72,
        )

    request_context["quality_guardrail_reasons"] = reasons
    return reasons


# ---------------------------------------------------------------------------
# Follow-up classifier helpers
# ---------------------------------------------------------------------------

_FOLLOWUP_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "that", "this", "these",
    "those", "then", "than", "with", "from", "have", "will", "your", "they", "them",
    "were", "been", "into", "only", "just", "does", "about", "more", "also", "some",
    "okay", "sure", "next", "please", "give", "tell", "show", "help", "want",
}
_CONTINUATION_WORDS = {
    "ok", "okay", "go", "and", "yes", "sure", "proceed", "continue", "more",
    "next", "then", "again", "retry", "try", "now", "well", "mmk", "hmm",
}


def _is_true_followup(user_question: str) -> bool:
    """Return True only when the question has no meaningful new topic content.

    Rules:
    * Pure punctuation / single-char ('?', '!') -> True
    * All words are continuation words or very short (<=3 chars) -> True
    * Question has a content word (4+ alpha chars, not a stopword) -> False
    """
    words = user_question.lower().split()
    if not words or (len(words) == 1 and not words[0].isalpha()):
        return True
    content_words = [
        w for w in words
        if len(w) >= 4 and w.isalpha() and w not in _FOLLOWUP_STOPWORDS
    ]
    if content_words:
        return False   # has a real topic -- treat as new question
    return len(words) <= 5



def _get_clean_vllm_answer(user_question: str) -> str | None:
    """Call vLLM with a minimal prompt to get a reliable answer from model knowledge."""
    if not user_question or len(user_question.strip()) < 3:
        return None
    for variant in ("medium", "full"):
        if not _is_vllm_live(variant):
            continue
        url_base, model_name = resolve_vllm_endpoint(target_id=None, model_variant=variant)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
                {"role": "user", "content": user_question},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        try:
            r = _health_session.post(
                f"{url_base}/chat/completions",
                json=payload,
                timeout=(3, 30),
            )
            if r.status_code == 200:
                choices = (r.json().get("choices") or [])
                if choices:
                    text = (choices[0].get("message") or {}).get("content", "").strip()
                    if text and len(text) >= 10:
                        return text
        except Exception:
            continue
    return None


def _rule_based_response(prompt: str, history: list | None = None) -> str:
    """Produce a useful reply when vLLM inference is unavailable or low-quality.

    Tries vLLM with a clean minimal prompt first; falls back to a generic
    message if vLLM is also unavailable.
    """
    # Extract the bare user question from the full prompt payload
    user_question = ""
    prompt_text = prompt or ""
    _USER_REQUEST_MARKER = "Current user request:"
    _ASSISTANT_MARKER = "\nAssistant:"
    if _USER_REQUEST_MARKER in prompt_text:
        after = prompt_text.split(_USER_REQUEST_MARKER, 1)[-1]
        if _ASSISTANT_MARKER in after:
            after = after.split(_ASSISTANT_MARKER, 1)[0]
        user_question = after.strip()
    if not user_question:
        for line in reversed(prompt_text.splitlines()):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith(
                ("System:", "Assistant:", "[[", "<|", "###", "User:", "Context:", "Retrieved")
            ):
                user_question = stripped_line
                break

    # Try vLLM with a clean minimal prompt -- no system context that causes hallucinations
    clean_answer = _get_clean_vllm_answer(user_question)
    if clean_answer:
        return clean_answer

    # vLLM unavailable -- return a helpful generic message
    topic = (user_question or prompt_text)[:120].strip()
    short_topic = topic.split()[0] if topic.split() else "that"
    return (
        f'I received your question about "{topic}".\n\n'
        f"The AI inference backend is currently starting up or under high load. "
        f"Please try again in a moment -- routing, RAG retrieval, and monitoring are fully operational."
    )


_health_session = requests.Session()  # no retries -- health probes must fail fast


def resolve_vllm_endpoint(
    target_id: str | None,
    model_variant: str | None,
) -> tuple[str, str]:
    """
    Resolve (vLLM URL, HF model name) for a routing target.

    Priority:
      1. Zoo entry's ``vllm_endpoint_env`` + ``vllm_model_id`` fields when the
         target_id matches a registered model. This lets the zoo own per-target
         endpoint mapping (e.g. the MoE container vs the dedicated Llama2-7B
         CPU fallback container) without code changes.
      2. The variant-keyed ``VLLM_URL_MAP`` / ``VLLM_MODEL_MAP``.
      3. ``VLLM_MEDIUM_URL`` + TinyLlama, the historical defaults.

    Used by ``run_vllm_inference`` and ``_is_vllm_live`` so dispatch and the
    liveness probe always agree.

    Honest-fallback rule (paper §6.3): if a dedicated endpoint env (MoE / STEM /
    fallback) resolves to the *same* URL as VLLM_FULL_URL — meaning the
    operator did NOT bring up a dedicated container — the requested HF model
    name is also rebound to whatever vllm-full actually serves. Sending the
    dedicated model name to a container that never loaded it would cause vLLM
    to return 400 and force the variant-escalation path, which silently
    corrupts the audit trail (the chosen variant looks like e.g. "stem-math"
    but the real serve is dense full). Rebinding here keeps the metadata
    truthful: the candidate's model_variant remains as picked, but the
    audit/UI sees the actual served model id.
    """
    url: str | None = None
    model_name: str | None = None
    used_full_fallback = False

    if target_id:
        try:
            zoo_entry = get_model_zoo().get_model(target_id)
        except Exception:
            zoo_entry = None
        if zoo_entry:
            env_name = zoo_entry.get("vllm_endpoint_env")
            if env_name:
                # Re-read at lookup time so a deployment can rotate the URL
                # without restarting the API; falls back to the cached value.
                url = os.getenv(env_name) or _VLLM_ENDPOINTS_BY_ENV.get(env_name)
                # Honest-fallback: dedicated endpoint env collapsed to the full
                # container — flag it so we substitute the served model name.
                if (
                    env_name not in {"VLLM_MEDIUM_URL", "VLLM_FULL_URL"}
                    and url
                    and url == VLLM_FULL_URL
                ):
                    used_full_fallback = True
            zoo_model_id = zoo_entry.get("vllm_model_id") or zoo_entry.get("model_id")
            if zoo_model_id and "/" in zoo_model_id:
                model_name = zoo_model_id

    if url is None:
        url = VLLM_URL_MAP.get((model_variant or "").lower(), VLLM_MEDIUM_URL)
    if model_name is None:
        model_name = VLLM_MODEL_MAP.get(
            (model_variant or "").lower(), "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

    if used_full_fallback:
        full_served = VLLM_MODEL_MAP.get("full", "Qwen/Qwen2.5-1.5B-Instruct")
        if model_name != full_served:
            logger.info(
                "Endpoint fallback: target=%s wanted %s but dedicated container "
                "is not deployed (env collapsed to VLLM_FULL_URL); serving with "
                "%s instead.",
                target_id, model_name, full_served,
            )
            model_name = full_served
    return url, model_name


def _is_vllm_live(model_variant: str, target_id: str | None = None) -> bool:
    """Check if the designated vLLM server is responsive by querying its /health endpoint."""
    url_base, _ = resolve_vllm_endpoint(target_id, model_variant)
    health_url = url_base.rsplit('/v1', 1)[0] + "/health"
    try:
        r = _health_session.get(health_url, timeout=2)
        return r.ok
    except Exception:
        return False
# Variant escalation chain when the chosen endpoint is offline. STEM/MoE/
# fallback all step *down* into "full" first since that's the always-on
# container in the default 2-container stack.
_VARIANT_UPGRADE_MAP: dict[str, str | None] = {
    "ultra-light":  "medium",
    "medium":       "full",
    "full":         None,
    "moe":          "full",
    "stem-math":    "full",
    "stem-science": "full",
    "stem-coding":  "full",
}

# Per-variant context window. Mirrors --max-model-len in
# docker-compose.ubuntu-vgpu.yml. Used to clamp max_tokens so the
# request fits within input + output ≤ ctx_limit; otherwise vLLM
# returns 400 and the caller escalates to a different variant.
_VARIANT_MAX_MODEL_LEN: dict[str, int] = {
    "ultra-light":  2048,
    "medium":       2048,
    "full":         4096,
    "moe":          8192,
    "stem-math":    2048,
    "stem-science": 2048,
    # 4096, not 2048: a coding prompt carries the task plus surrounding context,
    # which overflows 2048 and makes vLLM 400. 8192 is not worth its KV cost on
    # this slice.
    "stem-coding":  4096,
}


@lru_cache(maxsize=32)
def _variant_output_cap(model_variant: str) -> int:
    """Per-variant `max_output_tokens` from the model zoo, 0 when uncapped.

    Read from the same registry CSS prices candidates against, so the router's
    expectation and the dispatcher's ceiling cannot disagree. Cached because this
    sits in the hot dispatch path and the zoo is reloaded explicitly, not per
    request.
    """
    try:
        for target in load_routing_targets(ROUTING_TARGETS_PATH):
            if str(target.get("model_variant", "")).lower() == (model_variant or "").lower():
                cap = int(safe_float(target.get("max_output_tokens"), 0.0))
                if cap > 0:
                    return cap
    except Exception:  # noqa: BLE001 - never let a config read break dispatch
        logger.debug("output cap lookup failed for %s", model_variant, exc_info=True)
    return 0


def run_vllm_inference(
    model_variant: str,
    prompt: str,
    allow_rule_based_fallback: bool = True,
    _attempted: set | None = None,
    target_id: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str, str]:
    """
    Call vLLM OpenAI API inference and degrade safely when the backend is unavailable.

    ``target_id`` (optional) is the model_zoo entry id of the selected
    candidate. When supplied it drives endpoint + HF model name resolution
    via :py:func:`resolve_vllm_endpoint`, allowing dedicated vLLM containers
    for MoE / STEM / Llama2-7B-fallback. When absent, falls back to the
    legacy variant-keyed defaults.

    ``timeout_s`` (optional) overrides the read timeout for callers that are not
    latency-bound. VLLM_TIMEOUT_SECONDS is tuned for chat, where a user is waiting;
    the coding agent is not that caller. Measured: the 7B coder needs well over 45 s
    to emit a whole file, so it hit the chat timeout, this function returned "", and
    the agent billed ~16 gCO2 for an empty response and escalated on it. The energy
    was spent either way — the only thing the short timeout bought was throwing away
    the answer we had already paid for.
    """
    url_base, model_name = resolve_vllm_endpoint(target_id, model_variant)
    infer_url = f"{url_base}/chat/completions"

    if _attempted is None:
        _attempted = set()
    _attempted.add(model_variant)

    # STEM variants get more headroom + lower temperature: engineering math
    # derivations are step-by-step and chain-of-thought-heavy, and 1024 tokens
    # truncates partway through complex problems.
    if model_variant == "stem-math":
        _max_tokens, _temperature = 2048, 0.1
    elif model_variant in ("stem-science", "stem-coding"):
        _max_tokens, _temperature = 2048, 0.15
    else:
        _max_tokens, _temperature = 1024, 0.2

    # Clamp max_tokens to fit inside the container's max_model_len. vLLM
    # rejects (400) when input + max_tokens > ctx, which would otherwise
    # cascade through _escalate() to a different variant — defeating
    # STEM/MoE routing whenever the requested output budget equals ctx, and
    # silently drifting accumulating conversations onto the largest model.
    # 256-token reserve absorbs tokenizer drift between our 3.2 chars/token
    # estimate and the model's actual count (markdown / code / unicode pushes
    # the real ratio lower than the estimate suggests).
    ctx_limit = _VARIANT_MAX_MODEL_LEN.get(model_variant, 4096)
    input_tokens = estimate_token_count(prompt, model_variant)
    safe_output = max(128, ctx_limit - input_tokens - 256)
    _max_tokens = min(_max_tokens, safe_output)

    # Per-variant verbosity cap (model_zoo `max_output_tokens`). Carbon here is
    # power x duration and duration scales with emitted tokens, so an unbounded
    # output budget on a weak, rambling model is a direct carbon cost: the
    # a three-arm routing comparison measured TinyLlama emitting 1.6x the tokens of
    # Qwen2.5-1.5B on identical prompts. The caps sit well above observed mean
    # lengths, so they act as a rail against runaway generation rather than
    # routine truncation, and CSS prices each candidate against the same ceiling.
    _variant_cap = _variant_output_cap(model_variant)
    if _variant_cap > 0:
        _max_tokens = min(_max_tokens, _variant_cap)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _max_tokens,
        "temperature": _temperature,
    }

    def _escalate(skip_same_backend: bool = False) -> tuple[str, str] | None:
        """Try the next variant up the chain, dropping target_id (escalations
        intentionally route to whatever generic container backs that variant).

        When ``skip_same_backend`` is True, skip a hop whose dispatch URL
        matches the current one. ultra-light and medium both back to
        VLLM_MEDIUM_URL (TinyLlama, 2048 ctx) so context-overflow failures on
        ultra-light have no chance of succeeding on medium — jumping straight
        to full saves a wasted call and prevents a second 400 in the log.
        """
        next_variant = _VARIANT_UPGRADE_MAP.get(model_variant)
        # Skip identical-backend hops (ultra-light → medium when both serve TinyLlama)
        while skip_same_backend and next_variant and next_variant not in _attempted:
            try:
                next_url, _ = resolve_vllm_endpoint(None, next_variant)
                cur_url, _ = resolve_vllm_endpoint(target_id, model_variant)
            except Exception:
                break
            if next_url != cur_url:
                break
            _attempted.add(next_variant)
            next_variant = _VARIANT_UPGRADE_MAP.get(next_variant)
        if next_variant and next_variant not in _attempted:
            logger.info("Auto-escalating inference from %s to %s", model_variant, next_variant)
            return run_vllm_inference(
                next_variant, prompt, allow_rule_based_fallback, _attempted,
                timeout_s=timeout_s,
            )
        return None

    if not _is_vllm_live(model_variant, target_id=target_id):
        logger.warning(
            "vLLM endpoint for variant=%s target=%s not live; fallback allowed=%s",
            model_variant, target_id, allow_rule_based_fallback,
        )
        if (escalated := _escalate()) is not None:
            return escalated
        return (_rule_based_response(prompt), model_variant) if allow_rule_based_fallback else ("", model_variant)

    try:
        response = http_session.post(
            infer_url, json=payload, timeout=(5, timeout_s or VLLM_TIMEOUT_SECONDS)
        )

        if response.status_code != 200:
            logger.warning(
                "vLLM model '%s' unavailable (%d): %s | requested max_tokens=%d, ctx=%d, est_input=%d",
                model_name, response.status_code, response.text[:300],
                _max_tokens, ctx_limit, input_tokens,
            )
            # Context-overflow 400 means the prompt won't fit in the current
            # variant's ctx; if the next variant shares the same backend (and
            # therefore the same ctx) escalating to it is guaranteed to fail
            # for the same reason. Skip past same-backend hops to reach a real
            # bigger-ctx model.
            _is_ctx_overflow = (
                response.status_code == 400
                and ("maximum context length" in response.text.lower()
                     or "context_length_exceeded" in response.text.lower())
            )
            if (escalated := _escalate(skip_same_backend=_is_ctx_overflow)) is not None:
                return escalated
            return (_rule_based_response(prompt), model_variant) if allow_rule_based_fallback else ("", model_variant)

        result = response.json()
        choices = result.get("choices") or []
        if not choices:
            return (_rule_based_response(prompt), model_variant) if allow_rule_based_fallback else ("", model_variant)

        text = choices[0].get("message", {}).get("content", "")
        if text:
            return (text, model_variant)
        return (_rule_based_response(prompt), model_variant) if allow_rule_based_fallback else ("", model_variant)

    except Exception as exc:
        logger.error("vLLM inference failed for '%s': %s", model_name, exc)
        if (escalated := _escalate()) is not None:
            return escalated
        return (_rule_based_response(prompt), model_variant) if allow_rule_based_fallback else ("", model_variant)

def _sign_entry(entry: dict[str, Any]) -> str:
    """HMAC-SHA256 signature for audit log integrity (Section 3.6.1)."""
    payload_bytes = json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hmac.new(AUDIT_HMAC_KEY, payload_bytes, hashlib.sha256).hexdigest()


def log_decision(entry: dict[str, Any]) -> None:
    # Add cryptographic signature before writing (Section 3.6.1)
    signed_entry = dict(entry)
    signed_entry.setdefault("request_id", str(uuid4()))
    signed_entry["_hmac"] = _sign_entry({k: v for k, v in signed_entry.items() if k != "_hmac"})
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(signed_entry, ensure_ascii=False) + "\n")
    logger.info("Logged decision for conversation %s (signed)", entry.get("conversation_id"))


_AUDIT_READ_BLOCK_BYTES = 64 * 1024


def _iter_log_lines_reverse(path: Path, block_size: int = _AUDIT_READ_BLOCK_BYTES):
    """Yield non-empty lines of a text file from last to first.

    Seeks backwards in blocks rather than reading the whole file, so the cost of
    answering a query is proportional to how far back the caller has to look —
    not to the total size of the log, which only ever grows.
    """
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        carry = b""
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size) + carry
            parts = chunk.split(b"\n")
            # The first element may continue into the block before this one, so
            # hold it back and prepend it on the next iteration.
            carry = parts.pop(0)
            for part in reversed(parts):
                if part.strip():
                    yield part.decode("utf-8", errors="replace")
        if carry.strip():
            yield carry.decode("utf-8", errors="replace")


def _audit_entry_matches(
    entry: dict[str, Any],
    from_ts: float | None,
    to_ts: float | None,
    model_filter: str | None,
    tier_filter: str | None,
    tenant_filter: str | None,
    min_carbon_g: float | None,
) -> bool:
    """Whether one audit entry satisfies every supplied filter."""
    if from_ts is not None or to_ts is not None:
        # An unparseable timestamp is not grounds for dropping the entry: it is
        # still a signed record of a decision. Keep it and let the other filters
        # decide, matching the previous behaviour.
        try:
            ts = datetime.fromisoformat(
                str(entry.get("timestamp", "")).replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            ts = None
        if ts is not None:
            if from_ts is not None and ts < from_ts:
                return False
            if to_ts is not None and ts > to_ts:
                return False

    if model_filter and entry.get("selected_model") != model_filter:
        return False

    # user_tier — standard|premium|esg|batch
    if tier_filter and entry.get("user_tier") != tier_filter:
        return False

    # tenant_id — the multi-tenant isolation key
    if tenant_filter:
        entry_tenant = entry.get("tenant_id") or DEFAULT_TENANT_ID
        if entry_tenant != tenant_filter:
            return False

    if min_carbon_g is not None:
        entry_carbon = safe_float(
            entry.get("estimated_carbon_g")
            or (entry.get("selected_candidate") or {}).get("estimated_carbon_g"),
            0.0,
        )
        if entry_carbon < min_carbon_g:
            return False

    return True


def read_audit_log(
    from_ts: float | None = None,
    to_ts: float | None = None,
    model_filter: str | None = None,
    tier_filter: str | None = None,
    tenant_filter: str | None = None,
    min_carbon_g: float | None = None,
    max_entries: int = 200,
) -> list[dict[str, Any]]:
    """Query the audit log with filters (Section 3.6.1 compliance query API).

    Returns the **most recent** matching entries, newest first, capped at
    ``max_entries``.

    The log is append-only, so this reads it backwards. The previous
    implementation scanned forward from the start of the file and stopped at the
    first ``max_entries`` matches, which meant that once a tenant had more than
    that many entries the endpoint returned the oldest records in the log and
    never surfaced recent activity — the wrong end of the file for a compliance
    query, and a scan whose cost grew with total history.

    `tier_filter` filters on `user_tier` (standard|premium|esg|batch).
    `tenant_filter` filters on `tenant_id` — the multi-tenant isolation key.
    """
    if not LOG_FILE.exists() or max_entries <= 0:
        return []
    results: list[dict[str, Any]] = []
    try:
        for line in _iter_log_lines_reverse(LOG_FILE):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn or hand-edited line must not be fatal
            if not isinstance(entry, dict):
                continue
            if not _audit_entry_matches(
                entry, from_ts, to_ts, model_filter,
                tier_filter, tenant_filter, min_carbon_g,
            ):
                continue
            results.append(entry)
            if len(results) >= max_entries:
                break
    except OSError as exc:
        logger.error("Audit log read failed: %s", exc)
    return results


def build_routing_metadata(
    request_context: dict[str, Any],
    policy_config: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    eco_actions: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = ranked_candidates[0] if ranked_candidates else {}
    return {
        "policy_version": request_context["policy_coefficients"].get("version")
        or policy_config.get("version"),
        "user_tier": request_context["user_tier"],
        "request_context": {
            "sla_ms": request_context["sla_ms"],
            "accuracy_floor": request_context["accuracy_floor"],
            "deferral_tolerance_ms": request_context["deferral_tolerance_ms"],
            "region_preference": request_context["region_preference"],
            "model_preference": request_context["model_preference"],
            "recommended_model_variant": request_context.get("recommended_model_variant"),
            "priority": request_context.get("priority"),
            "mode": request_context.get("mode"),
        },
        "input_understanding": request_context.get("semantic_profile") or {},
        "task_profile": task_profile or {},
        "selected_candidate": selected,
        "candidate_rankings": ranked_candidates[:5],
        "eco_actions": eco_actions,
        "quality_guardrail_reasons": request_context.get("quality_guardrail_reasons", []),
    }


async def process_chat_request(
    prompt: str,
    priority: str = "",
    mode: str = "",
    conversation_id: str | None = None,
    task_profile_override: dict[str, Any] | None = None,
    attachments: list[UploadFile] | None = None,
    user_tier: str = "standard",
    accuracy_floor: float | None = None,
    sla_ms: int | None = None,
    deferral_tolerance_ms: int | None = None,
    region_preference: str | None = None,
    model_preference: str | None = None,
    persist_attachments: bool = False,
    top_k: int = DEFAULT_RAG_TOP_K,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> dict[str, Any]:
    cleaned_prompt = prompt.strip()
    normalized_priority = priority.strip().lower()
    normalized_mode = mode.strip().lower()
    normalized_user_tier = user_tier.strip().lower() or "standard"
    parsed_attachments = await read_attachments(attachments)

    if not cleaned_prompt and not parsed_attachments:
        raise HTTPException(status_code=400, detail="Provide a prompt or attach at least one file.")

    effective_prompt = cleaned_prompt or "Analyze the uploaded files and answer using the indexed evidence."
    conversation_title = build_conversation_title(cleaned_prompt or "Indexed document analysis")

    try:
        conversation = store.ensure_conversation(conversation_id, conversation_title, tenant_id=tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc

    history = store.list_messages(conversation["id"], include_internal=True, tenant_id=tenant_id)

    # Conversation-scoping inputs for the semantic cache. A context-dependent
    # followup ("why?", "explain that", "the second one") must only reuse a
    # cached answer from the *same* thread and the *same* recent context —
    # otherwise the bare-prompt embedding match returns text generated for a
    # different conversation. Compute both once here (against the pre-turn
    # history the answer is conditioned on) and reuse for lookup + store.
    _conv_id = conversation["id"]
    _is_followup = semantic_cache.is_context_dependent_followup(effective_prompt)
    _recent_history_fp = semantic_cache.recent_history_fingerprint(
        history, keep_recent=HISTORY_SUMMARY_KEEP_RECENT,
    )

    #  NemoGuardrails: input rail check — MUST run before the semantic-cache
    #  lookup and the arithmetic short-circuit below. Input safety is a property
    #  of the user's prompt, not of how we answer it: if the cache (or a
    #  deterministic short-circuit) returns early, the rails would never run and
    #  a harmful prompt that matches a cached entry would be served un-checked.
    #  Running here guarantees I.3/I.9 gate every request path.
    _guardrail_trace: dict[str, Any] = {}
    if GUARDRAILS_ENABLED:
        _gr_result = apply_guardrails(effective_prompt, phase="input")
        _guardrail_trace["input"] = _gr_result
        if _gr_result.get("blocked"):
            refusal_text = _gr_result.get("safe_replacement") or (
                "I can't help with that request."
            )
            blocked_user_msg = store.save_message(
                conversation["id"],
                "user",
                effective_prompt,
                metadata={
                    "priority": normalized_priority,
                    "mode": normalized_mode,
                    "user_tier": normalized_user_tier,
                    "tenant_id": tenant_id,
                    "blocked_by_guardrails": True,
                    "guardrail_phase": "input",
                },
                tenant_id=tenant_id,
            )
            blocked_assistant_msg = store.save_message(
                conversation["id"],
                "assistant",
                refusal_text,
                metadata={
                    "blocked_by_guardrails": True,
                    "guardrail_phase": "input",
                    "guardrail_reason": _gr_result.get("reason", ""),
                    "tenant_id": tenant_id,
                    "guardrails": _guardrail_trace,
                },
                tenant_id=tenant_id,
            )
            conv_after = store.get_conversation(conversation["id"], tenant_id=tenant_id) or conversation
            return {
                "status": "blocked",
                "reason": _gr_result.get("reason", "guardrail-input-blocked"),
                "guardrail_trace": _guardrail_trace,
                "conversation": conv_after,
                "messages": store.list_messages(conversation["id"], tenant_id=tenant_id),
                "user_message": blocked_user_msg,
                "assistant_message": blocked_assistant_msg,
                "tenant_id": tenant_id,
            }

    #  Budget pre-check (observability only — never blocks)
    # Tenant budgets used to hard-block at the token cap (HTTP 429), but the
    # caps are tuned for production-scale traffic and trip almost immediately
    # during development. Keep the evaluation so the UI sidebar can still show
    # current usage / cost, but never raise.
    _budget_eval: dict[str, Any] | None = None
    try:
        _est_input_tokens = max(len(effective_prompt) // 4, 1)  # rough char/4 heuristic
        _budget_eval = budgets_mod.evaluate_budget(
            tenant_id, _est_input_tokens, audit_reader=read_audit_log,
        )
    except Exception as _be_exc:
        logger.warning("Budget evaluation failed (non-blocking): %s", _be_exc)
        _budget_eval = None

    # Deterministic arithmetic short-circuit (runs before cache lookup).
    # Pure arithmetic is microseconds to evaluate locally and the answer is
    # always correct; routing it through the semantic cache risks returning
    # an LLM's wrong answer that was cached before the natural-language math
    # normaliser landed (e.g. "square root of 12794587" → bogus "1" persisted
    # at similarity=1.0). Detecting and skipping the cache here also means
    # the deterministic answer doesn't get cached itself further down — see
    # the store call near the end of process_chat_request.
    _arith_prelookup = None
    if not parsed_attachments and effective_prompt:
        try:
            _arith_strip_prelookup = re.sub(
                r"^(?:what\s+is|calculate|compute|evaluate|solve|find|simplify|what'?s)\s+",
                "",
                effective_prompt,
                flags=re.IGNORECASE,
            ).strip(" ?=!,.")
            _arith_prelookup = _try_arithmetic(_arith_strip_prelookup)
        except Exception:  # noqa: BLE001
            _arith_prelookup = None

    #  Semantic cache lookup (skip vLLM on near-duplicate prompt)
    _cache_hit = None
    if _arith_prelookup is None:
        try:
            _cache_hit = semantic_cache.lookup(
                tenant_id, effective_prompt, parsed_attachments,
                conversation_id=_conv_id,
                recent_history_fp=_recent_history_fp,
                is_followup=_is_followup,
            )
        except Exception as _ce_exc:
            logger.warning("Semantic cache lookup failed (non-blocking): %s", _ce_exc)
    if _cache_hit:
        # Re-derive the routing recommendation for *this* prompt and re-tag the
        # cache hit's model_variant accordingly. The stored model_variant
        # records whatever model originally generated the cached text, which
        # may have been an over-escalation (auto-escalate-to-full from a prior
        # bug, manual model_preference, etc.). Surfacing that stale label
        # makes every cache hit look like the request went to the heavy model
        # — the "all B2B prompts going to Qwen2.5-1.5B" symptom. No inference
        # is run here, so the displayed variant should reflect today's green
        # routing decision, not yesterday's escalation.
        try:
            _cache_profile = infer_prompt_profile(
                effective_prompt,
                attachments=parsed_attachments,
                persist_attachments=persist_attachments,
                conversation_message_count=len(history),
                moe_available=False,
            )
            _retag_variant = _cache_profile.get("recommended_model_variant") or _cache_hit.get("model_variant")
        except Exception:  # noqa: BLE001 — best-effort retag
            _retag_variant = _cache_hit.get("model_variant")

        # Guard against serving a stale answer generated by a *weaker* model than
        # this prompt now warrants. The re-tag above only relabels; the response
        # text is still whatever the original (possibly under-powered) model
        # produced. E.g. a coding prompt answered by TinyLlama and cached before
        # the router learned to send code to stem-coding: relabeling it
        # "stem-coding" would hand back low-quality code under a misleading tag.
        # When the current route is strictly more capable than the cached
        # origin, treat the hit as a miss and regenerate. (The reverse — a
        # cached heavy-model answer re-tagged greener — is still served: reusing
        # a *better* answer is always safe.) Once regenerated and re-stored at
        # the stronger variant, subsequent lookups match in rank and serve.
        _cached_origin_variant = _cache_hit.get("model_variant")
        if variant_capability_tier(_retag_variant) > variant_capability_tier(_cached_origin_variant):
            logger.info(
                "Semantic cache hit ignored: cached answer from '%s' is weaker "
                "than the '%s' this prompt now routes to; regenerating.",
                _cached_origin_variant, _retag_variant,
            )
            # Drop the stale weak entry so the regenerated stronger answer
            # replaces it; otherwise the weak entry keeps winning the similarity
            # tie and every hit regenerates (cache never converges).
            try:
                semantic_cache.invalidate(tenant_id, effective_prompt, parsed_attachments)
            except Exception as _inv_exc:  # noqa: BLE001 — best-effort purge
                logger.warning("Semantic cache invalidate failed (non-blocking): %s", _inv_exc)
            _cache_hit = None

    if _cache_hit:
        _cache_hit["original_model_variant"] = _cached_origin_variant
        _cache_hit["model_variant"] = _retag_variant

        cached_user_msg = store.save_message(
            conversation["id"], "user", effective_prompt,
            metadata={
                "priority": normalized_priority,
                "mode": normalized_mode,
                "attachments": parsed_attachments,
                "persist_attachments": persist_attachments,
                "user_tier": normalized_user_tier,
                "tenant_id": tenant_id,
                "from_semantic_cache": False,
            },
            tenant_id=tenant_id,
        )
        cached_assistant_msg = store.save_message(
            conversation["id"], "assistant", _cache_hit["response"],
            metadata={
                "model_variant": _retag_variant,
                "from_semantic_cache": True,
                "semantic_cache": {
                    "similarity":              _cache_hit.get("similarity"),
                    "cached_prompt":           _cache_hit.get("cached_prompt"),
                    "cached_at_iso":           _cache_hit.get("cached_at_iso"),
                    "hit_count":               _cache_hit.get("hit_count"),
                    "original_model_variant":  _cache_hit.get("original_model_variant"),
                },
                "tokens": {
                    "input":  _cache_hit.get("input_tokens", 0),
                    "output": _cache_hit.get("output_tokens", 0),
                    "total":  _cache_hit.get("input_tokens", 0) + _cache_hit.get("output_tokens", 0),
                },
                "tenant_id": tenant_id,
            },
            tenant_id=tenant_id,
        )
        conversation_after = store.get_conversation(conversation["id"], tenant_id=tenant_id) or conversation
        return {
            "status": "success",
            "conversation": conversation_after,
            "messages": store.list_messages(conversation["id"], tenant_id=tenant_id),
            "user_message": cached_user_msg,
            "assistant_message": cached_assistant_msg,
            "semantic_cache": _cache_hit,
            "tenant_id": tenant_id,
            "budget": _budget_eval,
            "model_variant": _retag_variant,
            "resolved_model_name": MODEL_NAME_MAP.get(
                _retag_variant or "medium", _retag_variant or "medium"
            ),
            "sustainability_score": 1.0,    # cache hit avoids inference entirely
            "system_co2_g": 0.0,
            "system_power_w": 0.0,
            "grid_carbon": None,
            "grid_signal": {},
            "grid_power": None,
            "context_messages_used": 0,
            "context_attachments_used": 0,
            "retrieved_chunks_used": 0,
            "task_profile": {},
            "input_understanding": {},
            "stem_domain": None,
            "general_domain": None,
            "system_metrics": {},
            "retrieval": {},
            "routing": {},
            "rag_status": rag_service.status(),
            "guardrail_trace": {},
        }

    # Model Zoo: check MoE availability for profiler
    zoo = get_model_zoo()
    zoo_targets = zoo.available_targets()
    moe_available = any(t.get("moe") and t.get("available") for t in zoo_targets)

    semantic_profile = infer_prompt_profile(
        effective_prompt,
        attachments=parsed_attachments,
        persist_attachments=persist_attachments,
        conversation_message_count=len(history),
        moe_available=moe_available,
    )

    policy_config = load_policy_config(POLICY_CONFIG_PATH)
    routing_targets = load_routing_targets(ROUTING_TARGETS_PATH)

    # Fetch grid signal up-front so the RL policy lookup can specialise per zone.
    # The Electricity Maps client caches aggressively, so this is cheap.
    grid_signal = fetch_grid_signal()
    grid_carbon = safe_float(grid_signal.get("carbon_intensity"), 475.0)
    grid_zone = (grid_signal.get("zone") or "primary") if isinstance(grid_signal, dict) else "primary"

    #  RL Controller: get current learned policy for this tier
    rl = get_rl_controller()
    rl_context = {
        "grid_carbon": grid_carbon,
        "grid_zone": grid_zone,
        "complexity": safe_float(semantic_profile.get("complexity_score"), 0.5),
        "priority": semantic_profile.get("priority", normalized_priority or "medium"),
        "user_tier": normalized_user_tier,
    }
    rl_policy = rl.get_policy(normalized_user_tier, context=rl_context, zone=grid_zone)

    request_context = build_request_context(
        {
            "priority": normalized_priority,
            "mode": normalized_mode,
            "user_tier": normalized_user_tier,
            "accuracy_floor": accuracy_floor,
            "sla_ms": sla_ms,
            "deferral_tolerance_ms": deferral_tolerance_ms,
            "region_preference": region_preference,
            "model_preference": model_preference,
            "semantic_profile": semantic_profile,
            # Override the static policy coefficients with RL-learned ones
            "_rl_policy_override": rl_policy,
        },
        policy_config,
    )
    # Inject RL weights into coefficients (replacing static config weights)
    request_context["policy_coefficients"].update({
        k: rl_policy[k] for k in ("carbon", "latency", "accuracy", "cost")
    })
    request_context["policy_coefficients"]["rl_controlled"] = True
    request_context["policy_coefficients"]["rl_version"] = rl_policy.get("version", 1)
    request_context["policy_coefficients"]["rl_exploration"] = rl_policy.get("exploration_applied", False)
    request_context["policy_coefficients"]["rl_source"] = rl_policy.get("source", "tier")
    request_context["policy_coefficients"]["rl_zone"] = rl_policy.get("zone")
    quality_guardrail_reasons = apply_quality_guardrails(
        request_context,
        effective_prompt,
        parsed_attachments,
        persist_attachments,
    )

    # grid_signal already fetched up-front for the RL zone lookup; reuse it.
    system_metrics = fetch_system_metrics()
    raw_grid_power = grid_signal.get("power_total_mw")
    system_power = safe_float(system_metrics.get("system_total_power"), 0.0)

    # Multi-region zone signals (Section 3.5.3)
    zone_signals: dict[str, Any] = {}
    zone_carbon_map: dict[str, float] = {}
    if MULTI_REGION_ENABLED:
        try:
            zone_signals = fetch_all_zone_signals()
            zone_carbon_map = get_zone_carbon_map(zone_signals)
        except Exception as exc:
            logger.warning("Multi-zone carbon fetch failed: %s", exc)

    # 48-hour forecast for deferred scheduling (Section 3.5.2)
    forecast = get_zone_forecast()

    ranked_candidates = rank_routing_candidates(
        request_context, routing_targets, grid_carbon,
        zone_carbon_map=zone_carbon_map or None,
    )

    #  GPU-aware routing adjustment (live GPU utilisation from sidecar)
    gpu_metrics = extract_gpu_metrics(system_metrics)
    ranked_candidates = apply_gpu_routing_adjustment(ranked_candidates, gpu_metrics)

    # STEM domain override: steer to a STEM-specialised model only when the
    # prompt is genuinely complex. For simple STEM queries (e.g. "what is 2+2",
    # "times table of 7"), the CSS-ranked greenest candidate must win — forcing
    # a 225 W Qwen2.5-Math model for trivial arithmetic contradicts the
    # platform's sustainability promise. Threshold = 0.5 on complexity_score.
    _stem_domain = semantic_profile.get("stem_domain")
    _stem_source = semantic_profile.get("stem_source")
    _stem_complexity = safe_float(semantic_profile.get("complexity_score"), 0.0)
    _STEM_COMPLEXITY_GATE = 0.5
    # Keyword-confirmed STEM (regex hit on a domain vocabulary term like
    # "table of", "equation", "derivative") is high-confidence: route to the
    # domain model regardless of complexity. A short prompt like "table of 17"
    # must still land on stem-math, even though complexity ≈ 0.17. The gate
    # only applies when stem_domain came from the noisy semantic classifier.
    _stem_hoist = bool(_stem_domain) and not model_preference and (
        _stem_source == "keyword" or _stem_complexity > _STEM_COMPLEXITY_GATE
    )
    if _stem_hoist:
        _stem_variant = STEM_VARIANT_FOR_DOMAIN.get(_stem_domain, "full")
        # Prefer the exact STEM zoo entry if available, else steer the top candidate
        _stem_candidate = next(
            (c for c in ranked_candidates if c.get("model_variant") == _stem_variant),
            None,
        )
        if _stem_candidate:
            ranked_candidates = [_stem_candidate] + [
                c for c in ranked_candidates if c is not _stem_candidate
            ]
            logger.info(
                "STEM domain '%s' (source=%s, complexity=%.2f) → routing to variant '%s'",
                _stem_domain, _stem_source or "unknown", _stem_complexity, _stem_variant,
            )
        else:
            # No dedicated STEM entry; escalate to full for better reasoning
            _full_candidate = next(
                (c for c in ranked_candidates if c.get("model_variant") == "full"), None
            )
            if _full_candidate:
                ranked_candidates = [_full_candidate] + [
                    c for c in ranked_candidates if c is not _full_candidate
                ]
    elif _stem_domain and _stem_complexity <= _STEM_COMPLEXITY_GATE:
        logger.info(
            "STEM domain '%s' (source=%s) detected but complexity=%.2f ≤ %.2f — deferring to CSS ranking for green routing",
            _stem_domain, _stem_source or "unknown", _stem_complexity, _STEM_COMPLEXITY_GATE,
        )

    selected_candidate = ranked_candidates[0] if ranked_candidates else {
        "model_variant": "medium",
        "hardware": "vgpu",
        "region": "local",
        "css_score": 0.5,
        "estimated_latency_ms": request_context["sla_ms"],
        "estimated_accuracy": request_context["accuracy_floor"],
        "estimated_cost_units": 0.3,
        "estimated_carbon_g": 0.0,
        "op_carbon_g": 0.0,
        "emb_carbon_g": 0.0,
    }

    # EcoServe actions with forecast + zone signals (Section 3.5)
    eco_actions = evaluate_ecoserve_actions(
        request_context, grid_carbon, ranked_candidates,
        forecast=forecast, zone_signals=zone_signals or None,
    )

    estimated_duration_seconds = max(
        safe_float(selected_candidate.get("estimated_latency_ms"), 0.0) / 1000.0,
        0.05,
    )
    system_co2 = calculate_system_co2(system_power, grid_carbon, duration_seconds=estimated_duration_seconds)

    # LLMCarbon total carbon from candidate (preferred) or system estimate
    candidate_carbon = safe_float(selected_candidate.get("estimated_carbon_g"), 0.0)
    if candidate_carbon <= 0.0:
        selected_candidate["estimated_carbon_g"] = round(system_co2, 6)

    # MoE latency guard + expert placement (Section 5.1, Algorithm 3)
    if selected_candidate.get("is_moe"):
        moe_total_ms = safe_float(selected_candidate.get("estimated_latency_ms"), 0)
        zoo = get_model_zoo()
        moe_model_id = selected_candidate.get("target_id")
        placement_plan = zoo.plan_expert_placement(moe_model_id) if moe_model_id else {"fallback": True}
        if placement_plan.get("fallback"):
            logger.warning(
                "MoE expert placement failed for %s (%s); falling back to dense",
                moe_model_id, placement_plan.get("reason"),
            )
        # Effective latency = base latency + placement-aware comm overhead
        if not placement_plan.get("fallback") and "estimated_comm_ms" in placement_plan:
            placement_comm_ms = float(placement_plan["estimated_comm_ms"])
            base_dispatch_ms = moe_total_ms - safe_float(selected_candidate.get("moe_comm_latency_ms"), 0.0)
            moe_total_ms = base_dispatch_ms + placement_comm_ms
            selected_candidate = {
                **selected_candidate,
                "moe_comm_latency_ms":   round(placement_comm_ms, 2),
                "estimated_latency_ms":  round(moe_total_ms, 2),
                "expert_placement":      placement_plan,
            }
        # Guard: SLA blow-out OR placement failed → dense fallback
        if placement_plan.get("fallback") or moe_total_ms > request_context["sla_ms"] * 1.5:
            dense_fallback = next(
                (c for c in ranked_candidates if not c.get("is_moe")), None
            )
            if dense_fallback:
                selected_candidate = {
                    **dense_fallback,
                    "moe_latency_fallback":  True,
                    "moe_fallback_reason":   placement_plan.get("reason") or "sla_blowout",
                    "moe_attempted_target":  moe_model_id,
                }
                logger.info(
                    "MoE %s replaced by dense fallback (%s); reason=%s, moe_total_ms=%.0f",
                    moe_model_id, dense_fallback.get("target_id"),
                    selected_candidate["moe_fallback_reason"], moe_total_ms,
                )

    # Deferred queue: enqueue if conditions met (Section 3.5.2)
    deferred = False
    if eco_actions.get("deferral_recommended"):
        deferred_queue = get_deferred_queue(forecast_provider=fetch_grid_signal)
        # For deferred requests: we return immediately with a 202 Accepted signal in eco_actions
        deferred = True  # caller checks eco_actions["deferral_recommended"]

    task_profile_payload: dict[str, Any] = {
        "query": effective_prompt,
        "priority": request_context["priority"],
        "mode": request_context["mode"],
        "latency_sla": request_context["sla_ms"],
        "accuracy_req": request_context["accuracy_floor"],
        "task_profile": task_profile_override
        or {
            "accuracy_req": request_context["accuracy_floor"],
            "latency_sla": request_context["sla_ms"],
        },
    }
    task_profile = infer_task_profile(task_profile_payload)

    rag_documents = [
        document
        for document in (
            attachment_to_rag_document(attachment, conversation["id"])
            for attachment in parsed_attachments
        )
        if document
    ]

    indexing_summary = None
    if persist_attachments and rag_documents:
        indexing_summary = rag_service.index_documents(
            rag_documents,
            source="chat-upload",
            persist=True,
            tenant_id=tenant_id,
        )
        ephemeral_documents: list[dict[str, Any]] = []
    else:
        ephemeral_documents = rag_documents

    rag_result = rag_service.retrieve(
        effective_prompt,
        top_k=max(1, top_k),
        ephemeral_documents=ephemeral_documents,
        tenant_id=tenant_id,
    )
    evidence_assessment = assess_evidence_sufficiency(
        effective_prompt,
        parsed_attachments,
        rag_result,
        semantic_profile,
    )

    history_summary, memory_metadata = summarize_history_for_context(
        conversation["id"],
        history,
    )
    prompt_payload, prompt_cache_metadata = get_or_build_prompt_payload(
        history,
        effective_prompt,
        parsed_attachments,
        rag_result,
        history_summary,
        stem_domain=_stem_domain,
    )

    #  Token counting (pre-inference)
    selected_variant = selected_candidate["model_variant"]
    # Use the *uncapped* count for the overflow check below — clamping it at
    # the variant's ctx cap makes a 3000-token prompt look like exactly 2048
    # to a 2048-ctx variant, defeating the gate. Re-clamp after for metrics.
    _raw_input_tokens = estimate_token_count(prompt_payload["prompt"], selected_variant, clamp_to_cap=False)
    input_tokens = min(_raw_input_tokens, _MODEL_MAX_TOKENS.get(selected_variant, 4096))

    #  Token overflow escalation
    # If input_tokens exceeds the selected variant's *usable* window (ctx -
    # output reserve), escalate to the next larger variant so the model
    # receives the full context. ultra-light and medium share the TinyLlama
    # 2048 ctx; ultra-light has no smaller cap than medium so an "ultra-light
    # overflow" implicitly means "this prompt also overflows medium" — jump
    # past medium to full when that happens.
    _VARIANT_ORDER = ["ultra-light", "medium", "full", "moe"]
    _VARIANT_CAPS  = {"ultra-light": 2048, "medium": 2048, "full": 4096, "moe": 8192}
    _OUTPUT_RESERVE = 384   # leave room for response + tokenizer drift
    _overflow_escalated = False
    _original_variant   = selected_variant

    _ctx_cap = _VARIANT_CAPS.get(selected_variant, 4096)
    if _raw_input_tokens > (_ctx_cap - _OUTPUT_RESERVE):
        _cur_idx = _VARIANT_ORDER.index(selected_variant) if selected_variant in _VARIANT_ORDER else 0
        for _esc_variant in _VARIANT_ORDER[_cur_idx + 1:]:
            _esc_cap = _VARIANT_CAPS[_esc_variant]
            if _esc_cap == _ctx_cap:
                # Same-ctx hop (e.g. ultra-light → medium both share 2048) —
                # the prompt won't fit in the next either; skip past it.
                continue
            if _raw_input_tokens <= (_esc_cap - _OUTPUT_RESERVE):
                logger.info(
                    "Token overflow: %d tokens > %d usable cap for '%s'  escalating to '%s'",
                    _raw_input_tokens, _ctx_cap - _OUTPUT_RESERVE, selected_variant, _esc_variant,
                )
                selected_variant = _esc_variant
                input_tokens = min(_raw_input_tokens, _esc_cap)
                # Update selected_candidate with escalated variant so metadata is correct
                selected_candidate = {
                    **selected_candidate,
                    "model_variant":        _esc_variant,
                    "model_name":           MODEL_NAME_MAP.get(_esc_variant, _esc_variant),
                    "overflow_escalated":   True,
                    "original_variant":     _original_variant,
                }
                _overflow_escalated = True
                break

    # -- Pre-model: deterministic arithmetic/table intercept ----------------
    # Answer directly for deterministic computations so vLLM is never called
    # and the final quality-safety-net cannot replace a correct answer with the
    # "backend starting up" fallback message.
    _direct_response: str | None = None
    _used_direct_response = False
    # Strip only punctuation/whitespace — NOT letters. The original "?! `t" accidentally
    # stripped the letter 't', breaking "table of N" queries.
    _norm_q = re.sub(r"^[\s?!`]+|[\s?!`]+$", "", effective_prompt.lower())

    # 1) Multiplication / math tables
    # Use (?![\d.]) instead of (?!\.) so the digit group can't backtrack
    # across a decimal point: "math table of 123.693" must NOT match as "12".
    # A trailing decimal means the user asked about a non-integer — defer to
    # the math model rather than producing a wrong integer table.
    _TABLE_PRE = [
        re.compile(r"(?:table\s+of\s+|times\s+table\s+of\s+|multiplication\s+table\s+(?:of\s+)?|maths?\s+table\s+(?:of\s+)?|table\s+)(\d{1,3})(?![\d.])"),
        re.compile(r"(\d{1,3})(?![\d.])\s*(?:times|x|ka|ki)?\s*table"),
        re.compile(r"arithm\w*\s+table\s+(?:of\s+)?(\d{1,3})(?![\d.])"),
        re.compile(r"(\d{1,3})(?![\d.])\s*(?:ka|ki)\s+(?:pahada|table)"),
    ]
    _pre_table_num: int | None = None
    for _pt in _TABLE_PRE:
        _pm = _pt.search(_norm_q)
        if _pm:
            try:
                _pre_table_num = int(_pm.group(1))
                break
            except (ValueError, IndexError):
                pass
    if _pre_table_num is not None and 1 <= _pre_table_num <= 999:
        _rows = "\n".join(
            f"{_pre_table_num} x {i:2d} = {_pre_table_num * i}" for i in range(1, 13)
        )
        _direct_response = f"Multiplication table of {_pre_table_num}:\n\n{_rows}"

    # 2) Simple arithmetic expressions (2+5=?, sqrt(16), 3^4, sin(0), …)
    if _direct_response is None:
        # Strip surrounding words that are just framing ("what is", "calculate", etc.)
        _arith_strip = re.sub(
            r"^(?:what\s+is|calculate|compute|evaluate|solve|find|simplify|what'?s)\s+",
            "",
            _norm_q,
            flags=re.IGNORECASE,
        ).strip(" ?=!,.")
        _arith_result = _try_arithmetic(_arith_strip)
        if _arith_result:
            _direct_response = _arith_result

    #  (Input guardrail rail — phase="input" — already ran before the semantic
    #  cache lookup above, so it gates cache hits and short-circuits too.)

    #  Capture wall-clock start for actual latency measurement
    _inference_start = time.monotonic()

    # Every inference leg that actually burned GPU time, for ex-post carbon.
    # A quality retry runs the model twice; billing only the survivor undercounts
    # the request, so each leg is timed and billed separately.
    _inference_legs: list[dict[str, Any]] = []

    # Multimodal outputs (populated by the vision / image-gen branches below).
    _generated_images: list[str] = []
    _multimodal_meta: dict[str, Any] | None = None
    _request_modality = str(semantic_profile.get("modality") or "text").lower()

    # Multimodal requests are not *text*-grounded: the evidence is the image
    # itself (handled by the VLM), and generation has no retrieval at all. Clear
    # the text-RAG grounding flags so the post-dispatch grounding-verification /
    # insufficient-evidence machinery does not overwrite the multimodal answer.
    if _request_modality in {"vision", "image-gen"}:
        evidence_assessment = {
            **evidence_assessment, "grounded_request": False, "allow_model_answer": True,
        }

    allow_rule_based_fallback = not evidence_assessment.get("grounded_request", False)
    if _request_modality == "image-gen" and selected_candidate.get("is_diffusion"):
        # ── Image generation (diffusion) ──
        # The generic input rail already ran on the prompt above. Text-to-image
        # prompts state harmful content declaratively (a description of the
        # pixels), so run the image-generation content rail before dispatching to
        # the diffusion backend. Steps were carbon-capped by CSS.
        _gen_prompt = cleaned_prompt or effective_prompt
        _imggen_gr = (
            apply_guardrails(_gen_prompt, phase="image-gen")
            if GUARDRAILS_ENABLED else {"blocked": False}
        )
        if GUARDRAILS_ENABLED:
            _guardrail_trace["image_gen"] = _imggen_gr
        if _imggen_gr.get("blocked"):
            # Refuse before any pixels are synthesized — no image, no carbon spend.
            model_response = _imggen_gr.get(
                "safe_replacement",
                "I can't generate that image. This request violates the image "
                "content policy.",
            )
            _generated_images = []
            _multimodal_meta = {
                "kind": "image-generation",
                "blocked": True,
                "guardrail_reason": _imggen_gr.get("reason", ""),
            }
            logger.warning(
                "Image generation blocked by guardrails: %s",
                _imggen_gr.get("reason", ""),
            )
        else:
            _steps = int(selected_candidate.get("diffusion_steps") or 30)
            _gen = run_image_generation(selected_candidate, _gen_prompt, _steps)
            _generated_images = [_gen["image_data_uri"]]
            _multimodal_meta = {
                "kind": "image-generation",
                "backend": _gen["backend"],
                "model": _gen["model"],
                "steps": _gen["steps"],
                "width": _gen["width"],
                "height": _gen["height"],
                "note": _gen["note"],
            }
            _gen_real = _gen["backend"] in {"nim", "huggingface"}
            model_response = (
                f"Generated an image for: “{_gen_prompt.strip()[:200]}”"
                + (f"\n\n_(via {_gen['note']})_" if _gen_real
                   else f"\n\n_({_gen['note']} — set NIM_SDXL_URL/NIM_FLUX_URL or HF_TOKEN for real generation.)_")
            )
        _used_direct_response = True
        quality_guardrail_reasons = []
        _accuracy_outcome = "clean"
    elif _request_modality == "vision":
        # ── Image analysis (VLM) ──
        _image_uris = [
            a["image_data_uri"] for a in parsed_attachments
            if a.get("is_image") and a.get("image_data_uri")
        ]
        _vlm = run_vlm_inference(selected_candidate, cleaned_prompt or effective_prompt, _image_uris)
        model_response = _vlm["text"]
        _multimodal_meta = {
            "kind": "image-analysis",
            "backend": _vlm["backend"],
            "model": _vlm["model"],
            "image_count": len(_image_uris),
        }
        _used_direct_response = True
        quality_guardrail_reasons = []
        _accuracy_outcome = "clean"
    elif evidence_assessment.get("grounded_request") and not evidence_assessment.get("allow_model_answer", True):
        model_response = ""
        _accuracy_outcome = "insufficient-evidence-precheck"
    elif _direct_response is not None:
        model_response = _direct_response
        _used_direct_response = True
        quality_guardrail_reasons = []  # Do not escalate a deliberate fast-path answer
        _accuracy_outcome = "clean"
    else:
        _requested_variant = selected_candidate["model_variant"]
        _leg_start = time.monotonic()
        model_response, _final_var = run_vllm_inference(
            _requested_variant,
            prompt_payload["prompt"],
            allow_rule_based_fallback=allow_rule_based_fallback,
            target_id=selected_candidate.get("target_id"),
        )
        _leg_duration_s = time.monotonic() - _leg_start
        if _final_var and _final_var != _requested_variant:
            selected_candidate["model_variant"] = _final_var
            selected_candidate["auto_escalated"] = True
            # run_vllm_inference escalates internally and reports only the variant;
            # without this the target_id — and so the carbon bill — stays pinned to
            # the model that was ranked but never ran.
            selected_candidate["target_id"] = resolve_zoo_target(
                _final_var,
                selected_candidate.get("target_id"),
                selected_candidate.get("hardware"),
            ) or selected_candidate.get("target_id")
            selected_variant = _final_var
        _inference_legs.append({
            "leg": "primary",
            "requested_variant": _requested_variant,
            "served_variant": selected_candidate["model_variant"],
            "target_id": selected_candidate.get("target_id"),
            "duration_s": _leg_duration_s,
        })
        _accuracy_outcome = "clean"

    _actual_latency_ms = (time.monotonic() - _inference_start) * 1000.0
    _actual_latency_s  = _actual_latency_ms / 1000.0

    #  Token counting (post-inference) 
    output_tokens = estimate_token_count(model_response, selected_variant)
    total_tokens  = input_tokens + output_tokens

    #  GPU utilisation fallback estimate 
    # When hardware counters return 0 (no GPU / GPU metrics disabled), compute
    # an estimate from inference timing, queue depth, and model VRAM footprint.
    if gpu_metrics["utilization_pct"] == 0.0:
        _queue_depth = 0
        try:
            _dq = get_deferred_queue(forecast_provider=fetch_grid_signal)
            _queue_depth = int(_dq.status().get("queue_size", 0))
        except Exception:
            pass

        _est_util = estimate_gpu_utilization(
            actual_latency_ms   = _actual_latency_ms,
            estimated_latency_ms= safe_float(
                selected_candidate.get("estimated_latency_ms"), 150.0
            ),
            model_variant       = selected_variant,
            queue_size          = _queue_depth,
            used_memory_mb      = gpu_metrics["used_memory_mb"],
            total_memory_mb     = gpu_metrics["total_memory_mb"],
        )
        gpu_metrics = {
            **gpu_metrics,
            "utilization_pct":    _est_util,
            "utilization_source": "estimated",     # vs "hardware"
            "gpu_available":      True,
            "constrained":        _est_util > 80.0,
        }
        _cache_gpu_util_estimate(_est_util)   # expose to sidebar polling
        logger.info(
            "GPU util hardware=0  estimated %.1f%% "
            "(wall=%.0fms est=%.0fms queue=%d)",
            _est_util, _actual_latency_ms,
            safe_float(selected_candidate.get("estimated_latency_ms"), 150.0),
            _queue_depth,
        )

    #  GPU CO2 attributed to this request (measured draw; 0.0 on a vGPU slice,
    #  which reports power.draw = [N/A]).  Request carbon proper is computed
    #  ex-post further down, once the served model and response have settled.
    gpu_co2_g = compute_gpu_co2(
        gpu_metrics["power_w"],
        grid_carbon,
        _actual_latency_s,
    )

    model_response = normalize_response_text(model_response)

    # NemoGuardrails: output rail check (skip for deterministic direct responses)
    if GUARDRAILS_ENABLED and not _used_direct_response and model_response.strip():
        _gr_out = apply_guardrails(model_response, phase="output", context=effective_prompt)
        _guardrail_trace["output"] = _gr_out
        if _gr_out.get("blocked"):
            model_response = _gr_out.get(
                "safe_replacement",
                "I'm sorry, I can't provide that response.",
            )
        elif _gr_out.get("redactions") and _gr_out.get("redacted_text"):
            model_response = _gr_out["redacted_text"]

    grounded_request = evidence_assessment.get("grounded_request", False)
    grounding_verification: dict[str, Any] = {
        "supported": not grounded_request,
        "reason": "not-grounded-request" if not grounded_request else "pending",
        "response_support_ratio": 1.0 if not grounded_request else 0.0,
        "query_support_ratio": evidence_assessment.get("coverage_ratio", 0.0),
        "evidence_segment_count": 0,
    }

    if grounded_request and not model_response.strip():
        fallback_response, fallback_type = build_safe_grounded_fallback(
            effective_prompt,
            parsed_attachments,
            rag_result,
            evidence_assessment,
        )
        model_response = normalize_response_text(fallback_response)
        selected_candidate = {
            **selected_candidate,
            "quality_fallback_used": True,
            "quality_fallback_type": fallback_type,
            "grounded_answer_required": True,
        }
        grounding_verification = {
            "supported": fallback_type == "extractive-evidence",
            "reason": fallback_type,
            "response_support_ratio": 1.0 if fallback_type == "extractive-evidence" else 0.0,
            "query_support_ratio": evidence_assessment.get("coverage_ratio", 0.0),
            "evidence_segment_count": len(collect_attachment_segments(parsed_attachments, rag_result)),
        }
        _accuracy_outcome = (
            "grounded-fallback"
            if fallback_type == "extractive-evidence"
            else "insufficient-evidence-fallback"
        )

    # Auto-escalate-to-full gating: only retry when the response shows
    # *positive* signs of being broken (hallucination, gibberish, empty/
    # fragment). Brevity alone is not a signal — a short, correct answer
    # ("Paris.") should not force a re-run on the largest available model,
    # which is anti-green and was the dominant reason every prompt was
    # landing on Qwen2.5-1.5B in practice. See response_has_quality_red_flag.
    if (
        quality_guardrail_reasons
        and not selected_candidate.get("quality_fallback_used")
        and selected_candidate.get("model_variant") != "full"
        and response_has_quality_red_flag(model_response)
    ):
        full_candidate = next(
            (candidate for candidate in ranked_candidates if candidate.get("model_variant") == "full"),
            None,
        )
        if full_candidate:
            try:
                # Capture the origin variant BEFORE the mutations below overwrite it,
                # so quality_retry_from records where we escalated *from*.
                _retry_from = selected_candidate.get("model_variant")
                _retry_start = time.monotonic()
                model_response, _final_var = run_vllm_inference(
                    "full",
                    prompt_payload["prompt"],
                    allow_rule_based_fallback=not grounded_request,
                    target_id=full_candidate.get("target_id"),
                )
                _retry_duration_s = time.monotonic() - _retry_start
                if _final_var and _final_var != selected_candidate.get("model_variant"):
                    selected_candidate["model_variant"] = _final_var
                    selected_candidate["auto_escalated"] = True
                model_response = normalize_response_text(model_response)
                _retry_target_id = resolve_zoo_target(
                    _final_var or "full",
                    full_candidate.get("target_id"),
                    full_candidate.get("hardware"),
                ) or full_candidate.get("target_id")
                selected_candidate = {
                    **full_candidate,
                    "target_id": _retry_target_id,
                    "model_variant": _final_var or "full",
                    "quality_retry_triggered": True,
                    "quality_retry_from": _retry_from,
                }
                # The first leg already burned GPU time. Bill both.
                _inference_legs.append({
                    "leg": "quality_retry",
                    "requested_variant": "full",
                    "served_variant": _final_var or "full",
                    "target_id": _retry_target_id,
                    "duration_s": _retry_duration_s,
                })
                _accuracy_outcome = "quality_retry"
            except RuntimeError as exc:
                logger.warning("Quality retry on full model failed: %s", exc)
            except Exception as exc:
                logger.warning("Quality retry on full model raised: %r", exc)

    if (
        quality_guardrail_reasons
        and not selected_candidate.get("quality_fallback_used")
        and looks_low_quality_response(model_response)
    ):
        fallback_summary = build_attachment_fallback_summary(
            effective_prompt,
            parsed_attachments,
            rag_result,
        )
        if fallback_summary:
            model_response = fallback_summary
            selected_candidate = {
                **selected_candidate,
                "quality_fallback_used": True,
                "quality_fallback_type": "extractive-summary",
            }
            _accuracy_outcome = "fallback"

    #  Off-topic detection for document-grounded requests 
    # When a file is attached, check that the response actually addresses
    # the current request. Small models frequently answer the PREVIOUS question
    # from conversation history instead of the uploaded document.
    if parsed_attachments and quality_guardrail_reasons:
        _user_prompt_lower = cleaned_prompt.lower()
        _resp_lower = model_response.lower()
        # Collect keywords from the user prompt (4+ char words)
        _prompt_kws = set(re.findall(r"\b[a-z]{4,}\b", _user_prompt_lower)) - {
            "this", "that", "then", "when", "with", "from", "have", "will",
            "what", "your", "they", "them", "were", "been", "into", "only",
        }
        # Also add keywords from attachment filenames
        for _att in parsed_attachments:
            _fname = (_att.get("name") or "").lower()
            _prompt_kws.update(re.findall(r"[a-z]{4,}", _fname))
        # If fewer than 15% of prompt keywords appear in response, it's off-topic
        if _prompt_kws:
            _hits = sum(1 for kw in _prompt_kws if kw in _resp_lower)
            _ratio = _hits / len(_prompt_kws)
            if _ratio < 0.15:
                logger.info(
                    "Off-topic response detected (%.0f%% keyword match) for doc request  "
                    "using extractive fallback",
                    _ratio * 100,
                )
                _fallback = build_attachment_fallback_summary(
                    effective_prompt, parsed_attachments, rag_result
                )
                if _fallback:
                    model_response = _fallback
                    selected_candidate = {
                        **selected_candidate,
                        "quality_fallback_used": True,
                        "quality_fallback_type": "off-topic-extractive",
                    }
                    _accuracy_outcome = "off-topic-fallback"

    if grounded_request and model_response.strip():
        grounded_supported, grounding_verification = response_supported_by_evidence(
            effective_prompt,
            model_response,
            parsed_attachments,
            rag_result,
        )
        if not grounded_supported:
            fallback_response, fallback_type = build_safe_grounded_fallback(
                effective_prompt,
                parsed_attachments,
                rag_result,
                evidence_assessment,
            )
            model_response = normalize_response_text(fallback_response)
            selected_candidate = {
                **selected_candidate,
                "quality_fallback_used": True,
                "quality_fallback_type": fallback_type,
                "grounding_failed": True,
            }
            grounding_verification = {
                **grounding_verification,
                "supported": fallback_type == "extractive-evidence",
                "reason": fallback_type,
            }
            _accuracy_outcome = (
                "grounded-fallback"
                if fallback_type == "extractive-evidence"
                else "insufficient-evidence-fallback"
            )

    #  Final safety net
    # Skip for direct (deterministic) responses — math tables and arithmetic answers
    # are intentionally short/numeric and would be falsely flagged as low-quality.
    if not _used_direct_response and (looks_low_quality_response(model_response) or not model_response.strip()):
        if grounded_request:
            logger.warning(
                "All grounded quality paths exhausted; using evidence-safe fallback "
                "(response was %d chars)",
                len(model_response),
            )
            fallback_response, fallback_type = build_safe_grounded_fallback(
                effective_prompt,
                parsed_attachments,
                rag_result,
                evidence_assessment,
            )
            model_response = normalize_response_text(fallback_response)
            selected_candidate = {
                **selected_candidate,
                "final_fallback": True,
                "final_fallback_type": fallback_type,
            }
            grounding_verification = {
                **grounding_verification,
                "supported": fallback_type == "extractive-evidence",
                "reason": fallback_type,
            }
            _accuracy_outcome = (
                "grounded-fallback"
                if fallback_type == "extractive-evidence"
                else "insufficient-evidence-fallback"
            )
        else:
            logger.warning(
                "All quality paths exhausted  using context-aware rule-based fallback "
                "(response was %d chars)", len(model_response)
            )
            model_response = _rule_based_response(
                prompt=effective_prompt,
                history=history,   # inject conversation history for short follow-up detection
            )
            selected_candidate = {
                **selected_candidate,
                "final_fallback": True,
                "final_fallback_type": "rule-based-context-aware",
            }
            _accuracy_outcome = "rule-based-fallback"

    #  EX-POST carbon accounting
    # Everything above can still change the served model (overflow escalation,
    # run_vllm_inference's internal escalation, the quality retry) or replace the
    # response text (extractive / off-topic / grounding / rule-based fallbacks).
    # This is the first point where both have settled, so carbon is computed here
    # and nowhere earlier.
    #
    # Distinct from selected_candidate["estimated_carbon_g"], which is the EX-ANTE
    # spec-derived forecast CSS used to *pick* a candidate before running it. That
    # stays as-is; this is what the request actually cost.
    output_tokens = estimate_token_count(
        model_response,
        selected_candidate.get("model_variant") or selected_variant,
    )
    total_tokens = input_tokens + output_tokens

    _zoo = get_model_zoo()
    _carbon_legs: list[dict[str, Any]] = []
    for _idx, _leg in enumerate(_inference_legs):
        if not _leg.get("target_id"):
            continue
        _is_last = _idx == len(_inference_legs) - 1
        _leg_carbon = _zoo.compute_request_carbon(
            model_id=_leg["target_id"],
            grid_carbon_g_per_kwh=grid_carbon,
            measured_duration_s=_leg["duration_s"],
            # Only the surviving leg produced the tokens the user actually got.
            output_tokens=output_tokens if _is_last else 0,
            input_tokens=input_tokens,
        )
        _leg_carbon.update({
            "leg": _leg["leg"],
            "requested_variant": _leg["requested_variant"],
            "served_variant": _leg["served_variant"],
        })
        _carbon_legs.append(_leg_carbon)

    measured_compute_s = round(sum(l["duration_s"] for l in _inference_legs), 4)
    request_carbon_g = round(
        sum(safe_float(c.get("total_carbon_g"), 0.0) for c in _carbon_legs), 8
    )
    request_op_carbon_g = round(
        sum(safe_float(c.get("op_carbon_g"), 0.0) for c in _carbon_legs), 8
    )
    request_emb_carbon_g = round(
        sum(safe_float(c.get("emb_carbon_g"), 0.0) for c in _carbon_legs), 8
    )
    request_energy_j = round(
        sum(safe_float(c.get("energy_j"), 0.0) for c in _carbon_legs), 4
    )
    request_energy_wh = round(request_energy_j / 3600.0, 8)
    _served_leg = _carbon_legs[-1] if _carbon_legs else {}
    served_target_id = _served_leg.get("model_id", selected_candidate.get("target_id"))

    # Recomputed from the same leg list, so the ledger cannot disagree with itself
    # about which model ran (the pre-fix ordering read estimated_carbon_g before the
    # quality retry swapped selected_candidate out from under it).
    co2_per_token_ug = round((request_carbon_g / max(total_tokens, 1)) * 1e6, 2)
    co2_per_output_token_ug = round((request_carbon_g / max(output_tokens, 1)) * 1e6, 3)
    # Measured, not floored by the spec estimate as the pre-fix value was.
    infer_duration_s = measured_compute_s

    carbon_accounting = {
        "basis": "modeled-tdp-x-measured-duration",
        "carbon_schema": 2,
        "measured_compute_s": measured_compute_s,
        "op_carbon_g": request_op_carbon_g,
        "emb_carbon_g": request_emb_carbon_g,
        "total_carbon_g": request_carbon_g,
        "energy_j": request_energy_j,
        "energy_wh": request_energy_wh,
        "emb_rate_g_per_device_s": _served_leg.get("emb_rate_g_per_device_s", 0.0),
        "device_utilization": _served_leg.get("device_utilization", 0.0),
        "device_share": _served_leg.get("device_share", 0.0),
        "served_model_variant": selected_candidate.get("model_variant"),
        "served_target_id": served_target_id,
        "grid_carbon_g_per_kwh": grid_carbon,
        "billed_legs": _carbon_legs,
        "power_basis": "spec_tdp_upper_bound",
        "limitation": (
            "TDP is a spec constant, not a reading. GPU power telemetry is unavailable "
            "on this vGPU slice (nvidia-smi power.draw returns [N/A]), so operational "
            "carbon is an upper bound on true draw. Duration and token counts are measured."
        ),
    }

    routing_metadata = build_routing_metadata(
        request_context,
        policy_config,
        ranked_candidates,
        eco_actions,
        task_profile=task_profile,
    )
    retrieval_metadata = {
        "search_mode": rag_result.get("search_mode"),
        "retrieved_count": rag_result.get("retrieved_count"),
        "candidate_count": rag_result.get("candidate_count"),
        "context_characters": rag_result.get("context_characters"),
        "sources": rag_result.get("sources", [])[:5],
        "evidence_assessment": evidence_assessment,
        "grounding_verification": grounding_verification,
        "knowledge_base_status": rag_service.status(),
        "indexing_summary": indexing_summary,
        "memory_optimization": {
            **memory_metadata,
            **prompt_cache_metadata,
            "history_summary": history_summary,
        },
    }

    user_message = store.save_message(
        conversation["id"],
        "user",
        effective_prompt,
        metadata={
            "priority": request_context["priority"],
            "mode": request_context["mode"],
            "attachments": parsed_attachments,
            "persist_attachments": persist_attachments,
            "user_tier": normalized_user_tier,
            "tenant_id": tenant_id,
            "input_understanding": semantic_profile,
        },
        tenant_id=tenant_id,
    )

    assistant_message = store.save_message(
        conversation["id"],
        "assistant",
        model_response,
        metadata={
            "model_variant": selected_candidate["model_variant"],
            "resolved_model_name": MODEL_NAME_MAP.get(
                selected_candidate["model_variant"],
                selected_candidate["model_variant"],
            ),
            "tokens": {
                "input":              input_tokens,
                "output":             output_tokens,
                "total":              total_tokens,
                "co2_per_token_ug":   co2_per_token_ug,
                "model_context_cap":  _MODEL_MAX_TOKENS.get(selected_variant, 4096),
            },
            "gpu": {
                **gpu_metrics,
                "co2_g":              round(gpu_co2_g, 6),
                "inference_duration_s": round(infer_duration_s, 3),
                "routing_adjusted":   gpu_metrics.get("constrained", False),
            },
            "sustainability": {
                "score": round(selected_candidate.get("css_score", 0.0), 3),
                "grid_carbon": grid_carbon,
                "grid_power": raw_grid_power,
                "system_power_w": system_power,
                "system_co2_g": round(system_co2, 6),
                # Ex-post: billed against measured duration and the model that actually
                # served. Keeps its name so existing consumers keep working.
                "estimated_request_co2_g": round(request_carbon_g, 6),
                "carbon_accounting": carbon_accounting,
                "gpu_co2_g":          round(gpu_co2_g, 6),
                "actual_latency_ms":  round(_actual_latency_ms, 1),
                "task_profile": task_profile,
                "context_messages_used": prompt_payload["context_messages_used"],
                "context_attachments_used": prompt_payload["context_attachments_used"],
                "retrieved_chunks_used": prompt_payload["retrieved_chunks_used"],
                "grid": grid_signal,
                "system_metrics": system_metrics,
            },
            "routing": routing_metadata,
            "retrieval": retrieval_metadata,
            "grounding": grounding_verification,
            "input_understanding": semantic_profile,
            "memory": retrieval_metadata["memory_optimization"],
            "tenant_id": tenant_id,
            "guardrails": _guardrail_trace,
            # Multimodal: generated/analysed images + dispatch provenance.
            "modality": _request_modality,
            "images": _generated_images,
            "multimodal": _multimodal_meta,
        },
        tenant_id=tenant_id,
    )

    conversation = store.get_conversation(conversation["id"], tenant_id=tenant_id) or conversation
    messages = store.list_messages(conversation["id"], tenant_id=tenant_id)

    #  Store in semantic cache for future near-duplicate hits
    # Skip caching deterministic direct responses (arithmetic, multiplication
    # tables, etc.): the shortcut is essentially free to re-compute, and
    # caching it doesn't add value — it just risks pinning a value with the
    # currently-resolved model_variant metadata for a result that never
    # actually involved that model.
    if not _used_direct_response:
        try:
            semantic_cache.store(
                tenant_id, effective_prompt, model_response,
                model_variant=selected_candidate.get("model_variant", ""),
                input_tokens=input_tokens, output_tokens=output_tokens,
                attachments=parsed_attachments,
                conversation_id=_conv_id,
                recent_history_fp=_recent_history_fp,
            )
        except Exception as _se_exc:
            logger.warning("Semantic cache store failed (non-blocking): %s", _se_exc)

    log_decision(
        {
            "timestamp": utc_now_iso(),
            "tenant_id": tenant_id,
            "conversation_id": conversation["id"],
            "user_message_id": user_message.get("id"),
            "assistant_message_id": assistant_message.get("id"),
            "query_preview": trim_text(effective_prompt, 260),
            "priority": request_context["priority"],
            "mode": request_context["mode"],
            "user_tier": normalized_user_tier,
            "task_profile": task_profile,
            "input_understanding": semantic_profile,
            "policy_coefficients": request_context["policy_coefficients"],
            "request_context": request_context,
            "candidate_rankings": ranked_candidates[:5],
            "selected_candidate": selected_candidate,
            "eco_actions": eco_actions,
            "attachment_names": [attachment.get("name") for attachment in parsed_attachments],
            "persist_attachments": persist_attachments,
            "indexing_summary": indexing_summary,
            "rag_retrieved_count": rag_result.get("retrieved_count"),
            "rag_sources": rag_result.get("sources", [])[:5],
            "evidence_assessment": evidence_assessment,
            "grounding_verification": grounding_verification,
            "memory_optimization": retrieval_metadata["memory_optimization"],
            "system_metrics": system_metrics,
            "system_power_w": system_power,
            "grid_signal": grid_signal,
            "grid_carbon": grid_carbon,
            "grid_power": raw_grid_power,
            "system_co2_g": round(system_co2, 6),
            "sustainability_score": round(selected_candidate.get("css_score", 0.0), 3),
            "selected_model": selected_candidate.get("model_variant"),
            "resolved_model_name": MODEL_NAME_MAP.get(
                selected_candidate.get("model_variant", "medium"),
                selected_candidate.get("model_variant", "medium"),
            ),
            "response_preview": trim_text(model_response, 260),
            "context_messages_used": prompt_payload["context_messages_used"],
            "context_attachments_used": prompt_payload["context_attachments_used"],
            "retrieved_chunks_used": prompt_payload["retrieved_chunks_used"],
            # RL fields
            "actual_latency_ms": round(_actual_latency_ms, 2),
            "accuracy_outcome": _accuracy_outcome,
            "rl_policy_version": rl_policy.get("version", 1),
            "rl_episode": rl_policy.get("episode_count", 0),
            # Observability fields
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total_tokens,
                "co2_per_token_ug": co2_per_token_ug,
            },
            "gpu_utilization_pct": safe_float(gpu_metrics.get("utilization_pct"), 0.0),
            "gpu_power_w": safe_float(gpu_metrics.get("power_w"), 0.0),
            "gpu_co2_g": round(gpu_co2_g, 6),
            "infer_duration_s": round(infer_duration_s, 3),
            # Ex-post carbon (schema 2). Additive: every key above keeps its name and
            # its readers. Rows written before this change lack these fields and carry
            # a constant emb_carbon_g; gate any aggregation on carbon_schema == 2 and
            # never average schema-1 and schema-2 rows into one series.
            "carbon_schema": 2,
            "carbon_basis": "modeled-tdp-x-measured-duration",
            "request_carbon_g": request_carbon_g,
            "op_carbon_g": request_op_carbon_g,
            "emb_carbon_g": request_emb_carbon_g,
            "energy_j": request_energy_j,
            "energy_wh": request_energy_wh,
            "measured_compute_s": measured_compute_s,
            "served_target_id": served_target_id,
            "carbon_legs": [
                {
                    "leg": _c.get("leg"),
                    "model_id": _c.get("model_id"),
                    "requested_variant": _c.get("requested_variant"),
                    "served_variant": _c.get("served_variant"),
                    "duration_s": _c.get("duration_s"),
                    "op_carbon_g": _c.get("op_carbon_g"),
                    "emb_carbon_g": _c.get("emb_carbon_g"),
                    "total_carbon_g": _c.get("total_carbon_g"),
                }
                for _c in _carbon_legs
            ],
        }
    )

    #  RL: fire-and-forget outcome recording (non-blocking) 
    _rl_request_id = conversation["id"] + ":" + assistant_message.get("id", "")
    _rl_selected_scores = {
        k: safe_float(selected_candidate.get(k), 0.0)
        for k in ("carbon_score", "latency_score", "accuracy_score", "cost_score")
    }
    _rl_all_scores = [
        {k: safe_float(c.get(k), 0.0) for k in ("carbon_score", "latency_score", "accuracy_score", "cost_score", "css_score")}
        for c in ranked_candidates
    ]

    def _rl_update():
        try:
            rl.record_outcome(
                user_tier=normalized_user_tier,
                actual_latency_ms=_actual_latency_ms,
                sla_ms=float(request_context["sla_ms"]),
                # Ex-post: what the request actually cost, billed to the model that
                # actually served. The ex-ante estimated_carbon_g this used to read
                # is a spec constant, so the carbon reward term was learning nothing.
                actual_carbon_g=request_carbon_g,
                actual_cost_units=safe_float(selected_candidate.get("estimated_cost_units"), 0.3),
                accuracy_outcome=_accuracy_outcome,
                selected_scores=_rl_selected_scores,
                all_candidate_scores=_rl_all_scores,
                request_id=_rl_request_id,
                zone=grid_zone,
                grid_carbon_g_per_kwh=grid_carbon,
            )
        except Exception as _exc:
            logger.warning("RL record_outcome failed (non-critical): %s", _exc)

        # Learned quality/latency estimator update (feeds CSS accuracy/latency,
        # M5-adjacent). Only when the selected candidate carries the static
        # baselines it was scored against — i.e. it came from CSS ranking, not a
        # rule-based/direct-response fallback (which has no comparable dispatch
        # latency). Keyed on the dispatched variant.
        try:
            _ql_baseline_latency = safe_float(selected_candidate.get("baseline_latency_ms"), 0.0)
            if _ql_baseline_latency > 0.0:
                get_quality_latency_estimator().observe(
                    semantic_profile=semantic_profile,
                    variant=str(selected_candidate.get("model_variant") or selected_variant),
                    baseline_accuracy=safe_float(selected_candidate.get("baseline_accuracy"), 0.5),
                    baseline_latency_ms=_ql_baseline_latency,
                    actual_latency_ms=_actual_latency_ms,
                    accuracy_outcome=_accuracy_outcome,
                    # Teaches the length head how verbose this variant actually is
                    # on prompts like this one. The baseline is the same reference
                    # CSS priced the candidate against, so the learned scale is a
                    # ratio against a known constant rather than an absolute.
                    baseline_output_tokens=float(CSS_REFERENCE_OUTPUT_TOKENS),
                    actual_output_tokens=float(output_tokens or 0.0),
                )
        except Exception as _exc:
            logger.warning("ql_estimator observe failed (non-critical): %s", _exc)

    threading.Thread(target=_rl_update, daemon=True, name="rl-update").start()

    return {
        "status": "success",
        "tenant_id": tenant_id,
        "budget": _budget_eval,
        "semantic_cache": None,
        "conversation": conversation,
        "messages": messages,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "sustainability_score": round(selected_candidate.get("css_score", 0.0), 3),
        "model_variant": selected_candidate.get("model_variant"),
        "resolved_model_name": MODEL_NAME_MAP.get(
            selected_candidate.get("model_variant", "medium"),
            selected_candidate.get("model_variant", "medium"),
        ),
        "task_profile": task_profile,
        "input_understanding": semantic_profile,
        "stem_domain": _stem_domain,
        "general_domain": semantic_profile.get("general_domain"),
        "modality": _request_modality,
        "images": _generated_images,
        "multimodal": _multimodal_meta,
        "system_metrics": system_metrics,
        "system_power_w": system_power,
        "grid_carbon": grid_carbon,
        "grid_power": raw_grid_power,
        "grid_signal": grid_signal,
        "system_co2_g": round(system_co2, 6),
        "context_messages_used": prompt_payload["context_messages_used"],
        "context_attachments_used": prompt_payload["context_attachments_used"],
        "retrieved_chunks_used": prompt_payload["retrieved_chunks_used"],
        "retrieval": retrieval_metadata,
        "routing": routing_metadata,
        "rag_status": rag_service.status(),
        "guardrail_trace": _guardrail_trace,
    }


@app.post("/api/chat")
async def chat(
    prompt: str = Form(""),
    priority: str = Form(""),
    mode: str = Form(""),
    conversation_id: str | None = Form(None),
    task_profile: str | None = Form(None),
    attachments: list[UploadFile] | None = File(None),
    user_tier: str = Form("standard"),
    accuracy_floor: float | None = Form(None),
    sla_ms: int | None = Form(None),
    deferral_tolerance_ms: int | None = Form(None),
    region_preference: str | None = Form(None),
    model_preference: str | None = Form(None),
    persist_attachments: bool = Form(False),
    top_k: int = Form(DEFAULT_RAG_TOP_K),
    tenant_id: str = Depends(resolve_tenant),
):
    task_profile_override: dict[str, Any] | None = None
    if task_profile:
        try:
            parsed_profile = json.loads(task_profile)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="task_profile must be valid JSON.") from exc
        if not isinstance(parsed_profile, dict):
            raise HTTPException(status_code=400, detail="task_profile must be a JSON object.")
        task_profile_override = parsed_profile

    return await process_chat_request(
        prompt=prompt,
        priority=priority,
        mode=mode,
        conversation_id=conversation_id,
        task_profile_override=task_profile_override,
        attachments=attachments,
        user_tier=user_tier,
        accuracy_floor=accuracy_floor,
        sla_ms=sla_ms,
        deferral_tolerance_ms=deferral_tolerance_ms,
        region_preference=region_preference,
        model_preference=model_preference,
        persist_attachments=persist_attachments,
        top_k=top_k,
        tenant_id=tenant_id,
    )


@app.post("/decision")
async def legacy_decision(
    request: InferenceRequest,
    tenant_id: str = Depends(resolve_tenant),
):
    payload = await process_chat_request(
        prompt=request.query,
        priority=request.priority,
        mode=request.mode,
        conversation_id=request.conversation_id,
        task_profile_override=request.task_profile,
        attachments=None,
        user_tier=request.user_tier,
        accuracy_floor=request.accuracy_floor,
        sla_ms=request.sla_ms,
        deferral_tolerance_ms=request.deferral_tolerance_ms,
        region_preference=request.region_preference,
        model_preference=request.model_preference,
        persist_attachments=request.persist_attachments,
        top_k=request.top_k,
        tenant_id=tenant_id,
    )

    return {
        "status": payload["status"],
        "conversation_id": payload["conversation"]["id"],
        "sustainability_score": payload["sustainability_score"],
        "model_variant": payload["model_variant"],
        "resolved_model_name": payload["resolved_model_name"],
        "task_profile": payload["task_profile"],
        "system_metrics": payload["system_metrics"],
        "system_power_w": payload["system_power_w"],
        "grid_carbon": payload["grid_carbon"],
        "grid_power": payload["grid_power"],
        "system_co2_g": payload["system_co2_g"],
        "context_messages_used": payload["context_messages_used"],
        "context_attachments_used": payload["context_attachments_used"],
        "retrieved_chunks_used": payload["retrieved_chunks_used"],
        "retrieval": payload["retrieval"],
        "routing": payload["routing"],
        "model_response": payload["assistant_message"]["content"],
    }


@app.get("/api/conversations")
async def list_conversations(tenant_id: str = Depends(resolve_tenant)):
    return {
        "tenant_id": tenant_id,
        "conversations": store.list_conversations(tenant_id=tenant_id),
    }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    tenant_id: str = Depends(resolve_tenant),
):
    conversation = store.get_conversation(conversation_id, tenant_id=tenant_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "tenant_id": tenant_id,
        "conversation": conversation,
        "messages": store.list_messages(conversation_id, tenant_id=tenant_id),
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    tenant_id: str = Depends(resolve_tenant),
):
    deleted = store.delete_conversation(conversation_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "conversation_id": conversation_id, "tenant_id": tenant_id}


@app.get("/api/rag/status")
async def rag_status():
    return {"status": "ok", "rag": rag_service.status()}


@app.get("/api/rag/documents")
async def rag_documents(tenant_id: str = Depends(resolve_tenant)):
    return {
        "tenant_id": tenant_id,
        "documents": rag_service.list_documents(tenant_id=tenant_id),
    }


@app.post("/api/rag/index")
async def rag_index(
    files: list[UploadFile] | None = File(None),
    source: str = Form("manual-upload"),
    tenant_id: str = Depends(resolve_tenant),
):
    parsed_attachments = await read_attachments(files)
    documents = [
        document
        for document in (
            attachment_to_rag_document(attachment, None)
            for attachment in parsed_attachments
        )
        if document
    ]
    if not documents:
        raise HTTPException(status_code=400, detail="Upload at least one text-extractable file.")

    indexing_summary = rag_service.index_documents(
        documents, source=source, persist=True, tenant_id=tenant_id,
    )
    return {
        "status": "success",
        "tenant_id": tenant_id,
        "indexing": indexing_summary,
        "rag": rag_service.status(),
        "documents": rag_service.list_documents(tenant_id=tenant_id),
    }


@app.delete("/api/rag/documents/{document_id}")
async def delete_rag_document(
    document_id: str,
    tenant_id: str = Depends(resolve_tenant),
):
    deleted = rag_service.delete_document(document_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Indexed document not found.")
    return {
        "status": "deleted",
        "document_id": document_id,
        "tenant_id": tenant_id,
        "rag": rag_service.status(),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "conversation_store_ready": store.healthcheck(),
        "rag_ready": True,
        "timestamp": utc_now_iso(),
    }


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe. Returns 'ok' when store + RAG are up.
    vLLM status is reported separately and does NOT block readiness — the API
    runs in graceful-degradation mode when no vLLM backend is serving.
    """
    vllm_ready = _is_vllm_live("medium") or _is_vllm_live("full")

    store_ready = store.healthcheck()
    rag_ready = True
    # API is ready as long as the conversation store and RAG are operational.
    # vLLM unavailability triggers graceful degradation, not a hard failure.
    overall_status = "ok" if store_ready and rag_ready else "degraded"

    return {
        "status": overall_status,
        "conversation_store_ready": store_ready,
        "rag_ready": rag_ready,
        "vllm_ready": vllm_ready,
        "inference_mode": "inference" if vllm_ready else "graceful-degradation",
        "timestamp": utc_now_iso(),
    }

#  New endpoints (P5, P6, P7, P2 paper sections) 

@app.get("/api/audit")
async def query_audit_log(
    from_iso: str | None = Query(None, description="ISO-8601 start time"),
    to_iso: str | None = Query(None, description="ISO-8601 end time"),
    model: str | None = Query(None, description="Filter by model variant (ultra-light|medium|full|moe)"),
    tier: str | None = Query(None, description="Filter by user_tier (standard|premium|esg|batch)"),
    tenant: str | None = Query(None, alias="tenant", description="(legacy alias for tier)"),
    min_carbon_g: float | None = Query(None, description="Minimum carbon threshold (gCO2)"),
    limit: int = Query(100, ge=1, le=500),
    tenant_id: str = Depends(resolve_tenant),
):
    """Compliance audit trail query (Section 3.6.1).

    Always scoped to the resolved tenant_id from `X-Tenant-Id`. The legacy
    `tenant=` query string still works as a tier filter for backwards-compat.
    """
    from_ts: float | None = None
    to_ts: float | None = None
    try:
        if from_iso:
            from datetime import datetime, timezone
            from_ts = datetime.fromisoformat(from_iso.replace("Z", "+00:00")).timestamp()
        if to_iso:
            from datetime import datetime, timezone
            to_ts = datetime.fromisoformat(to_iso.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {exc}") from exc

    effective_tier = tier or tenant
    entries = read_audit_log(
        from_ts=from_ts,
        to_ts=to_ts,
        model_filter=model,
        tier_filter=effective_tier,
        tenant_filter=tenant_id,
        min_carbon_g=min_carbon_g,
        max_entries=limit,
    )
    return {
        "count": len(entries),
        "filters": {
            "from": from_iso,
            "to": to_iso,
            "model": model,
            "tier": effective_tier,
            "tenant_id": tenant_id,
            "min_carbon_g": min_carbon_g,
        },
        "entries": entries,
    }


@app.get("/api/queue/status")
async def queue_status():
    """Deferred request queue status (Section 3.5.2)."""
    queue = get_deferred_queue(forecast_provider=fetch_grid_signal)
    return {"status": "ok", "queue": queue.status()}


@app.post("/api/queue/dispatch-now")
async def queue_dispatch_now():
    """Force-dispatch all deferred items (e.g., when low-carbon window arrives early)."""
    queue = get_deferred_queue(forecast_provider=fetch_grid_signal)
    grid_signal = fetch_grid_signal()
    ci = safe_float(grid_signal.get("carbon_intensity"), 475.0)
    queue.update_carbon(ci)
    return {"status": "dispatched", "current_carbon_g_per_kwh": ci, "queue": queue.status()}


@app.get("/api/model-zoo")
async def model_zoo_status():
    """Model Zoo registry status and available targets (Section 3.3)."""
    zoo = get_model_zoo()
    return {
        "status": "ok",
        "zoo": zoo.status(),
        "models": zoo.list_models(),
    }


@app.get("/api/model-zoo/{model_id}/carbon")
async def model_carbon_estimate(
    model_id: str,
    duration_ms: float = Query(200.0, description="Inference duration in ms"),
    token_count: int = Query(256, description="Estimated token count"),
):
    """On-demand LLMCarbon carbon estimate for a model (Section 3.4.2, 4.1)."""
    zoo = get_model_zoo()
    grid_carbon = safe_float(fetch_grid_signal().get("carbon_intensity"), 475.0)
    carbon = zoo.compute_total_carbon(
        model_id,
        grid_carbon_g_per_kwh=grid_carbon,
        inference_duration_s=duration_ms / 1000.0,
        token_count=token_count,
    )
    return {
        "model_id": model_id,
        "grid_carbon_g_per_kwh": grid_carbon,
        "duration_ms": duration_ms,
        "token_count": token_count,
        "carbon": carbon,
    }


@app.get("/api/model-zoo/{model_id}/expert-health")
async def expert_health(model_id: str):
    """MoE expert health status (Section 5.3)."""
    zoo = get_model_zoo()
    health = zoo.get_expert_health(model_id)
    model = zoo.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in zoo")
    if not model.get("moe"):
        return {"model_id": model_id, "moe": False, "message": "Not an MoE model"}
    return {
        "model_id": model_id,
        "moe": True,
        "healthy": zoo.is_moe_healthy(model_id),
        "health": health,
    }


@app.post("/api/model-zoo/{model_id}/expert-health")
async def update_expert_health_endpoint(
    model_id: str,
    healthy_experts: int = Form(...),
    total_experts: int = Form(...),
):
    """Update MoE expert health from an external health checker (Section 5.3)."""
    zoo = get_model_zoo()
    zoo.update_expert_health(model_id, healthy_experts, total_experts)
    return {"status": "updated", "model_id": model_id, "health": zoo.get_expert_health(model_id)}


@app.post("/api/model-zoo/reconcile")
async def reconcile_moe_now():
    """Force one FT-MoE reconciler tick on demand (paper §5.3)."""
    zoo = get_model_zoo()
    actions = zoo.reconcile_moe_health(
        health_probe=_moe_health_probe,
        rebalance_threshold=MOE_RECONCILER_REBALANCE_RATIO,
        disable_threshold=MOE_RECONCILER_DISABLE_RATIO,
    )
    return {
        "status":   "ok",
        "interval_s": MOE_RECONCILER_INTERVAL_S,
        "thresholds": {
            "rebalance": MOE_RECONCILER_REBALANCE_RATIO,
            "disable":   MOE_RECONCILER_DISABLE_RATIO,
        },
        "actions":  actions,
    }


@app.get("/api/model-zoo/{model_id}/expert-placement")
async def expert_placement(model_id: str):
    """Inspect the current MoE expert placement plan (paper §5.1, Algorithm 3)."""
    zoo = get_model_zoo()
    model = zoo.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    if not model.get("moe"):
        return {"model_id": model_id, "moe": False, "message": "Not an MoE model"}
    return zoo.plan_expert_placement(model_id)


# ── Model Zoo auto-updater (gated by human approval) ─────────────────────────

@app.get("/api/model-zoo/updates")
async def model_zoo_updates(_admin: bool = Depends(require_admin)):
    """List pending zoo entries fetched from the trusted source.

    Pending entries never affect routing until an admin explicitly approves them.
    """
    updater = get_zoo_updater()
    return {
        "status":  "ok",
        "updater": updater.status(),
        "pending": updater.list_pending(),
    }


@app.post("/api/model-zoo/updates/check-now")
async def model_zoo_updates_check(_admin: bool = Depends(require_admin)):
    """Trigger one fetch+validate cycle against the trusted source synchronously."""
    summary = get_zoo_updater().check_now()
    return {"status": "ok", "summary": summary}


@app.post("/api/model-zoo/updates/{update_id}/approve")
async def model_zoo_update_approve(
    update_id: str,
    _admin: bool = Depends(require_admin),
):
    """Apply a pending update to the live model zoo. Admin-only."""
    result = get_zoo_updater().approve(update_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Pending update '{update_id}' not found")
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "approve failed"))
    return result


@app.post("/api/model-zoo/updates/{update_id}/reject")
async def model_zoo_update_reject(
    update_id: str,
    reason: str = Form(""),
    _admin: bool = Depends(require_admin),
):
    """Drop a pending update without applying it. Admin-only."""
    result = get_zoo_updater().reject(update_id, reason=reason)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Pending update '{update_id}' not found")
    return result


@app.get("/api/grid/zones")
async def grid_zones():
    """Multi-zone grid carbon signals (Section 3.5.3)."""
    if MULTI_REGION_ENABLED:
        signals = fetch_all_zone_signals()
    else:
        primary = fetch_grid_signal()
        signals = {"local": primary}
    carbon_map = {zone: safe_float(s.get("carbon_intensity"), 475.0) for zone, s in signals.items()}
    best_zone = min(carbon_map, key=carbon_map.get) if carbon_map else None
    return {
        "multi_region_enabled": MULTI_REGION_ENABLED,
        "zones": signals,
        "carbon_map": carbon_map,
        "primary_zone": next(iter(signals), None),
        "best_zone": best_zone,
        "best_signal": signals.get(best_zone) if best_zone else None,
    }


@app.get("/api/grid/forecast")
async def grid_forecast(zone: str | None = Query(None)):
    """48-hour grid carbon forecast for scheduling (Section 3.2.2)."""
    forecast = get_zone_forecast(zone)
    return {
        "zone": zone or "primary",
        "point_count": len(forecast),
        "forecast": forecast,
    }


@app.get("/api/policy/suggest")
async def suggest_policy(
    tier: str | None = Query(None, description="User tier to analyze (standard|premium|esg|batch)"),
    tenant: str | None = Query(None, description="(legacy alias for tier)"),
    lookback_entries: int = Query(200, ge=10, le=1000),
    tenant_id: str = Depends(resolve_tenant),
):
    """
    RL policy foundation (Section 3.6.2):
    Analyze recent decision traces and suggest coefficient adjustments.
    Returns observed averages and gap vs SLAs for operator review.
    """
    entries = read_audit_log(
        tier_filter=(tier or tenant),
        tenant_filter=tenant_id,
        max_entries=lookback_entries,
    )
    if not entries:
        return {"status": "insufficient_data", "entries_analyzed": 0}

    # Compute observed statistics
    css_scores = [safe_float(e.get("sustainability_score"), 0) for e in entries]
    carbon_values = [
        safe_float((e.get("selected_candidate") or {}).get("estimated_carbon_g"), 0)
        for e in entries
    ]
    latency_violations = sum(
        1 for e in entries
        if safe_float((e.get("selected_candidate") or {}).get("estimated_latency_ms"), 0)
        > safe_float((e.get("request_context") or {}).get("sla_ms"), 9999)
    )
    model_distribution: dict[str, int] = {}
    for e in entries:
        m = e.get("selected_model", "unknown")
        model_distribution[m] = model_distribution.get(m, 0) + 1

    n = len(entries)
    avg_css = sum(css_scores) / n if n else 0
    avg_carbon = sum(carbon_values) / n if n else 0
    sla_violation_rate = latency_violations / n if n else 0

    # Heuristic coefficient suggestions (simplified RL reward signal, Section 3.6.2)
    suggestions: list[str] = []
    current_policy = entries[-1].get("policy_coefficients", {}) if entries else {}
    w_carbon = safe_float(current_policy.get("carbon"), 0.32)
    w_latency = safe_float(current_policy.get("latency"), 0.26)

    if sla_violation_rate > 0.10:
        suggestions.append(
            f"SLA violation rate {sla_violation_rate:.1%} > 10%; consider increasing latency weight "
            f"(currently {w_latency:.2f}) by 0.05 and reducing carbon weight (currently {w_carbon:.2f})."
        )
    if avg_carbon > 0.01:
        suggestions.append(
            f"Average carbon {avg_carbon*1000:.3f} mgCO2/request is elevated; "
            "enable deferral or activate multi-region routing."
        )
    if avg_css < 0.5:
        suggestions.append(
            "Average CSS score is below 0.5; review accuracy_floor constraints "
            "which may be eliminating better candidates."
        )
    if not suggestions:
        suggestions.append(
            f"Policy appears well-tuned (CSS={avg_css:.3f}, SLA violations={sla_violation_rate:.1%}, "
            f"avg_carbon={avg_carbon*1000:.3f}mgCO2)."
        )

    return {
        "status": "ok",
        "tier_filter": (tier or tenant),
        "tenant_id": tenant_id,
        "entries_analyzed": n,
        "observed": {
            "avg_css_score": round(avg_css, 4),
            "avg_carbon_g": round(avg_carbon, 8),
            "sla_violation_rate": round(sla_violation_rate, 4),
            "model_distribution": model_distribution,
        },
        "current_policy": current_policy,
        "suggestions": suggestions,
    }


#  RL Controller endpoints 

@app.get("/api/rl/status")
async def rl_status():
    """
    Current RL policy state per tenant tier.
    Weights are NOT user-configurable  determined solely by online REINFORCE learning.
    """
    rl = get_rl_controller()
    return {"status": "ok", "rl": rl.status()}


@app.get("/api/routing/quality-latency-estimator")
async def quality_latency_estimator_status():
    """
    Learned per-prompt quality/latency estimator state (read-only).
    Refines the static per-model accuracy/latency baselines that feed CSS with a
    correction learned online from observed outcomes. Carbon is not adjusted.
    Per-variant `n_obs`/`trusted` show which variants have enough evidence to be
    applied (below the warm-up threshold the estimator returns baselines).
    """
    return {"status": "ok", "estimator": get_quality_latency_estimator().snapshot()}


@app.get("/api/multimodal/status")
async def multimodal_status():
    """
    Multimodal capability status (read-only): which NVIDIA NIM endpoints are
    configured for image analysis (VLM) and image generation (diffusion). When
    an endpoint is unset the router still accepts the modality and returns a
    labelled placeholder (graceful fallback), so capabilities are always
    'available' — 'live' distinguishes a real NIM backend from the fallback.
    """
    import multimodal as _mm
    hf_gen = bool(_mm.HF_IMAGE_FALLBACK_ENABLED and _mm.HF_TOKEN)
    caps = [
        {"modality": "vision", "kind": "image-analysis",
         "env": "NIM_VLM_URL", "model": "nvidia/nemotron-nano-12b-v2-vl",
         "live": bool(NIM_VLM_URL), "fallback": "metadata"},
        {"modality": "image-gen", "kind": "image-generation",
         "env": "NIM_SDXL_URL", "model": "stabilityai/stable-diffusion-xl",
         "live": bool(NIM_SDXL_URL),
         "fallback": "huggingface" if hf_gen else "placeholder"},
        {"modality": "image-gen", "kind": "image-generation",
         "env": "NIM_FLUX_URL", "model": "black-forest-labs/flux.1-dev",
         "live": bool(NIM_FLUX_URL),
         "fallback": "huggingface" if hf_gen else "placeholder"},
    ]
    return {
        "status": "ok",
        "any_live": any(c["live"] for c in caps),
        # Generation produces REAL images whenever NIM is live OR the HF fallback
        # is enabled (HF_TOKEN present) — placeholders only when both are absent.
        "image_generation_real": any(bool(NIM_SDXL_URL) or bool(NIM_FLUX_URL) for _ in [0]) or hf_gen,
        "hf_generation_fallback": hf_gen,
        "hf_model": _mm._HF_DEFAULT_IMAGE_MODEL if hf_gen else None,
        "capabilities": caps,
    }


@app.get("/api/rl/history")
async def rl_history(
    tier: str | None = Query(None, description="Tenant tier (standard|premium|esg|batch)"),
    last_n: int = Query(100, ge=1, le=200),
):
    """Recent per-request reward observations driving policy updates."""
    rl = get_rl_controller()
    return {
        "status": "ok",
        "tier_filter": tier,
        "history": rl.reward_history(tier=tier, last_n=last_n),
    }


@app.post("/api/rl/reset/{tier}")
async def rl_reset_tier(tier: str):
    """
    Operator safety valve: reset a tier to its initial policy weights.
    NOT a UI control  for operator use only when a tier has diverged.
    """
    rl = get_rl_controller()
    valid_tiers = {"standard", "premium", "esg", "batch"}
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier '{tier}'. Valid: {sorted(valid_tiers)}",
        )
    result = rl.reset_tier(tier)
    return {"status": "reset", "tier": tier, "rl": result}


@app.get("/api/system/metrics")
async def system_metrics_api():
    """Live GPU + CPU + power metrics from the sidecar service.

    When hardware GPU counters return 0 (GPU metrics disabled or no GPU present),
    injects the latest inference-derived estimate so the sidebar always shows
    a meaningful GPU utilisation value.
    """
    raw = fetch_system_metrics()
    gpu = extract_gpu_metrics(raw)

    # Inject inference-based estimate when hardware provides nothing
    if gpu["utilization_pct"] == 0.0:
        cached = _get_cached_gpu_util_estimate()
        if cached is not None and cached > 0.0:
            gpu = {
                **gpu,
                "utilization_pct":    cached,
                "utilization_source": "estimated",
                "gpu_available":      True,
                "constrained":        cached > 80.0,
            }

    return {
        "status": "ok",
        "gpu": gpu,
        "cpu": {
            "utilization_pct": round(safe_float(raw.get("system_cpu_utilization"), 0.0), 1),
            "power_w":         round(safe_float(raw.get("system_cpu_power"), 0.0), 2),
        },
        "system": {
            "total_power_w":   round(safe_float(raw.get("system_total_power"), 0.0), 2),
            "energy_j":        round(safe_float(raw.get("system_energy"), 0.0), 4),
            "co2_emission_g":  round(safe_float(raw.get("system_co2_emission"), 0.0), 6),
        },
        "raw": raw,
    }

def _safe_int(v, default: int = 0) -> int:
    try:
        return int(safe_float(v, default))
    except Exception:
        return default


def _obs_percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _obs_entry_ts(entry):
    from datetime import datetime
    try:
        return datetime.fromisoformat(entry.get("timestamp", "").replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _obs_aggregate_kpis(entries: list, window_minutes: int) -> dict:
    """Return the KPI block for the given audit entries (used for current + prior window)."""
    total = len(entries)
    latencies = [safe_float(e.get("actual_latency_ms"), 0.0) for e in entries if e.get("actual_latency_ms")]
    co2_values = [safe_float(e.get("system_co2_g"), 0.0) for e in entries]
    css_scores = [safe_float(e.get("sustainability_score"), 0.0) for e in entries]
    grid_ci_values = [safe_float(e.get("grid_carbon"), 0.0) for e in entries if e.get("grid_carbon")]
    tokens_in = sum(_safe_int(((e.get("tokens") or {}).get("input")), 0) for e in entries)
    tokens_out = sum(_safe_int(((e.get("tokens") or {}).get("output")), 0) for e in entries)
    tokens_total = sum(_safe_int(((e.get("tokens") or {}).get("total")), 0) for e in entries)
    grounding_failures = sum(
        1 for e in entries
        if (e.get("grounding_verification") or {}).get("supported") is False
        and (e.get("grounding_verification") or {}).get("reason") not in (None, "not-grounded-request")
    )
    deferred_count = sum(
        1 for e in entries if (e.get("eco_actions") or {}).get("deferral_recommended")
    )
    rag_used_count = sum(1 for e in entries if _safe_int(e.get("rag_retrieved_count"), 0) > 0)
    return {
        "total_requests": total,
        "requests_per_min": round(total / max(window_minutes, 1), 2),
        "tokens_total": tokens_total,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "co2_total_g": round(sum(co2_values), 6),
        "co2_avg_g": round(sum(co2_values) / total, 6) if total else 0.0,
        "css_avg": round(sum(css_scores) / total, 3) if total else 0.0,
        "grid_ci_avg": round(sum(grid_ci_values) / len(grid_ci_values), 1) if grid_ci_values else 0.0,
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "latency_p50_ms": round(_obs_percentile(latencies, 50), 2),
        "latency_p95_ms": round(_obs_percentile(latencies, 95), 2),
        "latency_p99_ms": round(_obs_percentile(latencies, 99), 2),
        "grounding_failure_rate": round(grounding_failures / total, 4) if total else 0.0,
        "grounding_failures": grounding_failures,
        "deferred_count": deferred_count,
        "deferred_rate": round(deferred_count / total, 4) if total else 0.0,
        "rag_used_count": rag_used_count,
        "rag_use_rate": round(rag_used_count / total, 4) if total else 0.0,
    }


def _obs_delta(curr: float, prior: float) -> dict:
    """Return absolute and pct change for KPI period-over-period."""
    if prior is None:
        return {"abs": 0.0, "pct": None}
    abs_d = (curr or 0) - (prior or 0)
    if not prior:
        return {"abs": round(abs_d, 6), "pct": None}
    return {"abs": round(abs_d, 6), "pct": round(abs_d / prior * 100.0, 2)}


@app.get("/api/observability/summary")
async def observability_summary(
    window_minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    bucket_seconds: int = Query(60, ge=10, le=3600, description="Time-series bucket size"),
    slo_p95_ms: float = Query(3000.0, ge=50.0, le=60000.0, description="P95 latency SLO target (ms)"),
    slo_error_rate: float = Query(0.01, ge=0.0, le=1.0, description="Acceptable error rate (0..1)"),
    energy_price_usd_kwh: float = Query(0.12, ge=0.0, le=10.0, description="Electricity price USD/kWh"),
    cloud_input_usd_per_1k: float = Query(0.0015, ge=0.0, description="Cloud-equivalent $/1k input tokens"),
    cloud_output_usd_per_1k: float = Query(0.0020, ge=0.0, description="Cloud-equivalent $/1k output tokens"),
    tenant_id: str = Depends(resolve_tenant),
):
    """LLM observability aggregate metrics.

    Datadog/Elastic-style rollup of the audit log: KPIs (with prior-period
    deltas), latency percentiles + 2D heatmap, model usage distribution,
    SLO + error-budget burn rate, cost & efficiency analytics, time series
    for requests/latency/carbon/tokens, anomalies, top conversations,
    and recent traces. Always scoped to the resolved tenant_id.
    """
    from datetime import datetime, timezone

    now_ts = time.time()
    from_ts = now_ts - (window_minutes * 60.0)
    prior_from_ts = from_ts - (window_minutes * 60.0)
    # Pull both windows in one fetch (current + prior of equal length)
    all_entries = read_audit_log(
        from_ts=prior_from_ts, max_entries=4000, tenant_filter=tenant_id,
    )
    entries = [e for e in all_entries if _obs_entry_ts(e) >= from_ts]
    prior_entries = [e for e in all_entries if prior_from_ts <= _obs_entry_ts(e) < from_ts]

    _percentile = _obs_percentile
    _ts = _obs_entry_ts

    total = len(entries)
    latencies = [safe_float(e.get("actual_latency_ms"), 0.0) for e in entries if e.get("actual_latency_ms")]
    co2_values = [safe_float(e.get("system_co2_g"), 0.0) for e in entries]
    css_scores = [safe_float(e.get("sustainability_score"), 0.0) for e in entries]
    grid_ci_values = [safe_float(e.get("grid_carbon"), 0.0) for e in entries if e.get("grid_carbon")]
    tokens_in = [_safe_int(((e.get("tokens") or {}).get("input")), 0) for e in entries]
    tokens_out = [_safe_int(((e.get("tokens") or {}).get("output")), 0) for e in entries]
    tokens_total = [_safe_int(((e.get("tokens") or {}).get("total")), 0) for e in entries]

    grounding_failures = sum(
        1 for e in entries
        if (e.get("grounding_verification") or {}).get("supported") is False
        and (e.get("grounding_verification") or {}).get("reason") not in (None, "not-grounded-request")
    )
    deferred_count = sum(
        1 for e in entries if (e.get("eco_actions") or {}).get("deferral_recommended")
    )
    rag_used_count = sum(1 for e in entries if _safe_int(e.get("rag_retrieved_count"), 0) > 0)

    # Distributions
    by_model: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    by_intent: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    by_region: dict[str, int] = {}
    co2_by_model: dict[str, float] = {}
    latency_by_model: dict[str, list[float]] = {}
    tokens_by_model: dict[str, int] = {}

    for e in entries:
        m = e.get("selected_model") or "unknown"
        by_model[m] = by_model.get(m, 0) + 1
        co2_by_model[m] = co2_by_model.get(m, 0.0) + safe_float(e.get("system_co2_g"), 0.0)
        latency_by_model.setdefault(m, []).append(safe_float(e.get("actual_latency_ms"), 0.0))
        tokens_by_model[m] = tokens_by_model.get(m, 0) + _safe_int(((e.get("tokens") or {}).get("total")), 0)

        tier = e.get("user_tier") or "standard"
        by_tier[tier] = by_tier.get(tier, 0) + 1

        intent = ((e.get("input_understanding") or {}).get("intent")) or "unspecified"
        by_intent[intent] = by_intent.get(intent, 0) + 1

        prio = e.get("priority") or "normal"
        by_priority[prio] = by_priority.get(prio, 0) + 1

        region = ((e.get("selected_candidate") or {}).get("selected_region")) or "default"
        by_region[region] = by_region.get(region, 0) + 1

    # Time series (oldest → newest, fixed bucket count)
    bucket_count = max(1, (window_minutes * 60) // bucket_seconds)
    series_start = now_ts - (bucket_count * bucket_seconds)
    series = [
        {
            "bucket_start": series_start + i * bucket_seconds,
            "requests": 0,
            "latency_ms_avg": 0.0,
            "latency_ms_p95": 0.0,
            "co2_g": 0.0,
            "tokens": 0,
            "grid_ci": 0.0,
            "_lat": [],
            "_grid": [],
        }
        for i in range(bucket_count)
    ]
    for e in entries:
        t = _ts(e)
        idx = int((t - series_start) // bucket_seconds)
        if 0 <= idx < bucket_count:
            b = series[idx]
            b["requests"] += 1
            b["co2_g"] += safe_float(e.get("system_co2_g"), 0.0)
            b["tokens"] += _safe_int(((e.get("tokens") or {}).get("total")), 0)
            b["_lat"].append(safe_float(e.get("actual_latency_ms"), 0.0))
            gc = safe_float(e.get("grid_carbon"), 0.0)
            if gc > 0:
                b["_grid"].append(gc)

    for b in series:
        if b["_lat"]:
            b["latency_ms_avg"] = round(sum(b["_lat"]) / len(b["_lat"]), 2)
            b["latency_ms_p95"] = round(_percentile(b["_lat"], 95), 2)
        if b["_grid"]:
            b["grid_ci"] = round(sum(b["_grid"]) / len(b["_grid"]), 1)
        b["co2_g"] = round(b["co2_g"], 6)
        b.pop("_lat", None)
        b.pop("_grid", None)

    # Latency histogram (10 bins, log-ish for LLM range)
    bins = [50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 30000]
    histogram = [{"le_ms": b, "count": 0} for b in bins]
    histogram.append({"le_ms": None, "count": 0})  # +Inf
    for v in latencies:
        placed = False
        for h in histogram[:-1]:
            if v <= h["le_ms"]:
                h["count"] += 1
                placed = True
                break
        if not placed:
            histogram[-1]["count"] += 1

    # Recent traces (newest first, capped at 50)
    traces = []
    for e in sorted(entries, key=_ts, reverse=True)[:50]:
        sel = e.get("selected_candidate") or {}
        ground = e.get("grounding_verification") or {}
        toks = e.get("tokens") or {}
        eco = e.get("eco_actions") or {}
        traces.append({
            "timestamp": e.get("timestamp"),
            "conversation_id": e.get("conversation_id"),
            "request_id": e.get("assistant_message_id") or e.get("user_message_id"),
            "query_preview": e.get("query_preview"),
            "model": e.get("selected_model"),
            "model_name": e.get("resolved_model_name"),
            "tier": e.get("user_tier"),
            "priority": e.get("priority"),
            "intent": ((e.get("input_understanding") or {}).get("intent")),
            "complexity": ((e.get("input_understanding") or {}).get("complexity_label")),
            "latency_ms": safe_float(e.get("actual_latency_ms"), 0.0),
            "tokens_in": _safe_int(toks.get("input"), 0),
            "tokens_out": _safe_int(toks.get("output"), 0),
            "tokens_total": _safe_int(toks.get("total"), 0),
            "co2_g": safe_float(e.get("system_co2_g"), 0.0),
            "grid_carbon": safe_float(e.get("grid_carbon"), 0.0),
            "css_score": safe_float(e.get("sustainability_score"), 0.0),
            "gpu_utilization_pct": safe_float(e.get("gpu_utilization_pct"), 0.0),
            "rag_retrieved": _safe_int(e.get("rag_retrieved_count"), 0),
            "deferred": bool(eco.get("deferral_recommended")),
            "grounding_supported": ground.get("supported"),
            "grounding_reason": ground.get("reason"),
            "accuracy_outcome": safe_float(e.get("accuracy_outcome"), 0.0),
            "rl_policy_version": e.get("rl_policy_version"),
        })

    # Per-model latency rollup
    model_rollup = []
    for m, count in sorted(by_model.items(), key=lambda x: -x[1]):
        lats = latency_by_model.get(m, [])
        model_rollup.append({
            "model": m,
            "requests": count,
            "share_pct": round(count / total * 100, 1) if total else 0.0,
            "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0.0,
            "p95_latency_ms": round(_percentile(lats, 95), 2),
            "total_co2_g": round(co2_by_model.get(m, 0.0), 6),
            "total_tokens": tokens_by_model.get(m, 0),
        })

    # Top conversations (chattiest)
    by_conv: dict[str, dict[str, Any]] = {}
    for e in entries:
        cid = e.get("conversation_id") or "unknown"
        b = by_conv.setdefault(cid, {"conversation_id": cid, "requests": 0, "tokens": 0, "co2_g": 0.0, "last_ts": ""})
        b["requests"] += 1
        b["tokens"] += _safe_int(((e.get("tokens") or {}).get("total")), 0)
        b["co2_g"] += safe_float(e.get("system_co2_g"), 0.0)
        ts = e.get("timestamp", "")
        if ts > b["last_ts"]:
            b["last_ts"] = ts
    top_conversations = sorted(by_conv.values(), key=lambda x: -x["requests"])[:10]
    for c in top_conversations:
        c["co2_g"] = round(c["co2_g"], 6)

    # Anomaly: simple z-score on latencies
    anomalies = []
    if latencies and len(latencies) >= 10:
        mean = sum(latencies) / len(latencies)
        variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
        std = variance ** 0.5 if variance > 0 else 1.0
        for e in entries:
            v = safe_float(e.get("actual_latency_ms"), 0.0)
            if std > 0 and (v - mean) / std > 2.5:
                anomalies.append({
                    "timestamp": e.get("timestamp"),
                    "request_id": e.get("assistant_message_id"),
                    "latency_ms": round(v, 2),
                    "z_score": round((v - mean) / std, 2),
                    "model": e.get("selected_model"),
                })
        anomalies = sorted(anomalies, key=lambda x: -x["z_score"])[:10]

    # ── KPIs (current + prior + deltas) ─────────────────────────────────────
    kpis = _obs_aggregate_kpis(entries, window_minutes)
    prior_kpis = _obs_aggregate_kpis(prior_entries, window_minutes)
    delta_keys = (
        "total_requests", "requests_per_min", "tokens_total",
        "co2_total_g", "co2_avg_g", "css_avg", "grid_ci_avg",
        "latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "grounding_failure_rate", "deferred_rate", "rag_use_rate",
    )
    deltas = {k: _obs_delta(kpis.get(k, 0.0), prior_kpis.get(k, 0.0)) for k in delta_keys}

    # ── SLO + error budget ──────────────────────────────────────────────────
    slo_p95_compliance_pct = (
        round(sum(1 for v in latencies if v <= slo_p95_ms) / len(latencies) * 100, 2)
        if latencies else 100.0
    )
    error_budget_pct = (
        round(max(0.0, slo_error_rate - kpis["grounding_failure_rate"]) / max(slo_error_rate, 1e-9) * 100, 2)
        if slo_error_rate > 0 else 100.0
    )
    error_budget_burned_pct = round(max(0.0, 100.0 - error_budget_pct), 2)
    p95_breach_count = sum(1 for v in latencies if v > slo_p95_ms)
    slo_status = "healthy"
    if kpis["latency_p95_ms"] > slo_p95_ms or kpis["grounding_failure_rate"] > slo_error_rate:
        slo_status = "breach"
    elif kpis["latency_p95_ms"] > slo_p95_ms * 0.9 or error_budget_burned_pct > 75:
        slo_status = "warn"
    slo = {
        "p95_target_ms": slo_p95_ms,
        "p95_actual_ms": kpis["latency_p95_ms"],
        "p95_compliance_pct": slo_p95_compliance_pct,
        "p95_breach_count": p95_breach_count,
        "error_target_rate": slo_error_rate,
        "error_actual_rate": kpis["grounding_failure_rate"],
        "error_budget_remaining_pct": error_budget_pct,
        "error_budget_burned_pct": error_budget_burned_pct,
        "status": slo_status,
    }

    # ── Latency heatmap (time × latency-bin → density) ──────────────────────
    heat_lat_bins = [50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 30000, None]
    # Cap total cells for transmission: max 60 time columns
    heat_col_count = min(60, bucket_count)
    heat_col_seconds = max(bucket_seconds, (window_minutes * 60) // heat_col_count)
    heat_start = now_ts - (heat_col_count * heat_col_seconds)
    heatmap_cells = [[0] * heat_col_count for _ in range(len(heat_lat_bins))]
    heatmap_col_starts = [heat_start + i * heat_col_seconds for i in range(heat_col_count)]
    for e in entries:
        v = safe_float(e.get("actual_latency_ms"), 0.0)
        col = int((_ts(e) - heat_start) // heat_col_seconds)
        if not (0 <= col < heat_col_count):
            continue
        row = len(heat_lat_bins) - 1
        for i, edge in enumerate(heat_lat_bins[:-1]):
            if v <= edge:
                row = i
                break
        heatmap_cells[row][col] += 1
    heatmap_max = max((c for row in heatmap_cells for c in row), default=0)

    # ── Cost & efficiency analytics ─────────────────────────────────────────
    energy_kwh_total = 0.0
    cloud_input_usd = 0.0
    cloud_output_usd = 0.0
    cost_by_model: dict[str, dict[str, float]] = {}
    for e in entries:
        gpu_w = safe_float(e.get("gpu_power_w"), 0.0)
        dur_s = safe_float(e.get("infer_duration_s"), 0.0)
        if dur_s == 0.0:
            dur_s = safe_float(e.get("actual_latency_ms"), 0.0) / 1000.0
        kwh = (gpu_w * dur_s) / 3_600_000.0  # Wh→kWh
        energy_kwh_total += kwh
        toks = e.get("tokens") or {}
        ti = _safe_int(toks.get("input"), 0)
        to = _safe_int(toks.get("output"), 0)
        cloud_input_usd += (ti / 1000.0) * cloud_input_usd_per_1k
        cloud_output_usd += (to / 1000.0) * cloud_output_usd_per_1k
        m = e.get("selected_model") or "unknown"
        b = cost_by_model.setdefault(m, {"energy_kwh": 0.0, "cloud_usd": 0.0, "tokens": 0})
        b["energy_kwh"] += kwh
        b["cloud_usd"] += (ti / 1000.0) * cloud_input_usd_per_1k + (to / 1000.0) * cloud_output_usd_per_1k
        b["tokens"] += _safe_int(toks.get("total"), 0)

    energy_usd = energy_kwh_total * energy_price_usd_kwh
    cloud_usd = cloud_input_usd + cloud_output_usd
    cost = {
        "energy_kwh": round(energy_kwh_total, 6),
        "energy_usd": round(energy_usd, 6),
        "energy_price_usd_kwh": energy_price_usd_kwh,
        "cloud_equivalent_usd": round(cloud_usd, 6),
        "cloud_input_usd": round(cloud_input_usd, 6),
        "cloud_output_usd": round(cloud_output_usd, 6),
        "savings_usd": round(cloud_usd - energy_usd, 6),
        "savings_pct": (round((cloud_usd - energy_usd) / cloud_usd * 100, 2)
                        if cloud_usd > 1e-9 else None),
        "tokens_per_request": round(kpis["tokens_total"] / total, 2) if total else 0.0,
        "energy_kwh_per_1k_tokens": (round(energy_kwh_total / (kpis["tokens_total"] / 1000.0), 6)
                                     if kpis["tokens_total"] else 0.0),
        "co2_per_1k_tokens_g": (round(kpis["co2_total_g"] / (kpis["tokens_total"] / 1000.0), 6)
                                 if kpis["tokens_total"] else 0.0),
        "by_model": [
            {
                "model": m,
                "energy_kwh": round(v["energy_kwh"], 6),
                "energy_usd": round(v["energy_kwh"] * energy_price_usd_kwh, 6),
                "cloud_usd": round(v["cloud_usd"], 6),
                "tokens": v["tokens"],
            }
            for m, v in sorted(cost_by_model.items(), key=lambda kv: -kv[1]["cloud_usd"])
        ],
    }

    return {
        "status": "ok",
        "tenant_id": tenant_id,
        "window": {
            "minutes": window_minutes,
            "from_iso": datetime.fromtimestamp(from_ts, tz=timezone.utc).isoformat(),
            "to_iso": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "prior_from_iso": datetime.fromtimestamp(prior_from_ts, tz=timezone.utc).isoformat(),
            "bucket_seconds": bucket_seconds,
        },
        "kpis": kpis,
        "prior_kpis": prior_kpis,
        "deltas": deltas,
        "slo": slo,
        "cost": cost,
        "distributions": {
            "by_model": by_model,
            "by_tier": by_tier,
            "by_intent": by_intent,
            "by_priority": by_priority,
            "by_region": by_region,
        },
        "model_rollup": model_rollup,
        "time_series": series,
        "latency_histogram": histogram,
        "heatmap": {
            "lat_bin_edges_ms": heat_lat_bins,
            "col_starts": heatmap_col_starts,
            "col_seconds": heat_col_seconds,
            "cells": heatmap_cells,
            "max_count": heatmap_max,
        },
        "top_conversations": top_conversations,
        "anomalies": anomalies,
        "traces": traces,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-tenant control plane: budgets, semantic cache, CSRD/GHG reporting
# ─────────────────────────────────────────────────────────────────────────────

class _BudgetUpdateModel(BaseModel):
    monthly_token_limit: float | None = None
    daily_token_limit: float | None = None
    monthly_cost_usd_limit: float | None = None
    soft_warn_pct: float | None = None
    hard_block: bool | None = None


@app.get("/api/tenancy/whoami")
async def tenancy_whoami(tenant_id: str = Depends(resolve_tenant)):
    """Returns the resolved tenant_id from the X-Tenant-Id header."""
    return tenant_metadata(tenant_id)


@app.get("/api/budgets")
async def get_all_budgets(_admin: bool = Depends(require_admin)):
    """List all configured tenant budgets (admin)."""
    return {"status": "ok", "budgets": budgets_mod.list_budgets()}


@app.get("/api/budgets/me")
async def get_my_budget(tenant_id: str = Depends(resolve_tenant)):
    """Return the calling tenant's current budget + live usage."""
    budget = budgets_mod.get_budget(tenant_id)
    usage = budgets_mod.usage_for_tenant(tenant_id, audit_reader=read_audit_log)
    return {"status": "ok", "tenant_id": tenant_id, "budget": budget, "usage": usage}


@app.get("/api/budgets/{target_tenant}")
async def get_budget_for_tenant(
    target_tenant: str,
    _admin: bool = Depends(require_admin),
):
    """Inspect a specific tenant's budget + usage (admin)."""
    target = normalise_tenant_id(target_tenant)
    budget = budgets_mod.get_budget(target)
    usage = budgets_mod.usage_for_tenant(target, audit_reader=read_audit_log)
    return {"status": "ok", "tenant_id": target, "budget": budget, "usage": usage}


@app.post("/api/budgets/{target_tenant}")
async def set_budget_for_tenant(
    target_tenant: str,
    payload: _BudgetUpdateModel,
    _admin: bool = Depends(require_admin),
):
    """Create or update a per-tenant budget override (admin)."""
    target = normalise_tenant_id(target_tenant)
    fields = {k: v for k, v in payload.dict().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No budget fields supplied.")
    updated = budgets_mod.set_budget(target, fields)
    return {"status": "ok", "budget": updated}


@app.delete("/api/budgets/{target_tenant}")
async def delete_budget_for_tenant(
    target_tenant: str,
    _admin: bool = Depends(require_admin),
):
    """Remove a per-tenant override; tenant reverts to default budget (admin)."""
    target = normalise_tenant_id(target_tenant)
    removed = budgets_mod.delete_budget(target)
    if not removed:
        raise HTTPException(status_code=404, detail="No override exists for that tenant.")
    return {"status": "deleted", "tenant_id": target}


@app.get("/api/cache/status")
async def cache_status(tenant_id: str = Depends(resolve_tenant)):
    """Semantic cache stats for the calling tenant."""
    return {"status": "ok", "cache": semantic_cache.status(tenant_id)}


@app.post("/api/cache/clear")
async def cache_clear(tenant_id: str = Depends(resolve_tenant)):
    """Clear the calling tenant's semantic cache."""
    removed = semantic_cache.clear(tenant_id)
    return {"status": "cleared", "tenant_id": tenant_id, "removed": removed}


@app.post("/api/cache/clear-all")
async def cache_clear_all(_admin: bool = Depends(require_admin)):
    """Admin: clear semantic cache for all tenants."""
    removed = semantic_cache.clear(None)
    return {"status": "cleared", "scope": "all", "removed": removed}


# ── Interaction feedback (thumbs up/down) ────────────────────────────────────
# Captures a per-message quality signal from real usage. Stored in SQLite
# (upsert per message so a user can change their vote) and exportable as a
# JSONL fine-tuning / preference dataset via /api/feedback/export. This is the
# honest, offline path to "models that improve from usage": collect labelled
# (prompt, response, vote) pairs now, curate + fine-tune LoRA adapters later,
# then swap them into the vLLM containers. No model is trained inline here.

class FeedbackModel(BaseModel):
    message_id: str
    rating: str | int    # "up"/"down" or +1/-1 (normalised server-side)
    reason: str = ""
    conversation_id: str | None = None


def _normalize_rating(value: Any) -> int | None:
    """Map assorted rating spellings to +1 / -1, or None if unrecognised."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return 1 if v > 0 else (-1 if v < 0 else None)
    s = str(value or "").strip().lower()
    if s in {"up", "1", "+1", "positive", "good", "thumbs_up", "thumbsup", "like"}:
        return 1
    if s in {"down", "-1", "negative", "bad", "thumbs_down", "thumbsdown", "dislike"}:
        return -1
    return None


@app.post("/api/feedback")
async def submit_feedback(
    payload: FeedbackModel,
    tenant_id: str = Depends(resolve_tenant),
):
    """Record a thumbs up/down on an assistant message (upsert)."""
    if not (payload.message_id or "").strip():
        raise HTTPException(status_code=400, detail="message_id is required.")
    rating = _normalize_rating(payload.rating)
    if rating is None:
        raise HTTPException(
            status_code=400,
            detail="rating must be one of: up, down, +1, -1.",
        )
    saved = store.save_feedback(
        payload.message_id.strip(),
        rating,
        (payload.reason or "").strip()[:2000],
        tenant_id=tenant_id,
    )
    if saved is None:
        raise HTTPException(
            status_code=404,
            detail="No assistant message with that id for this tenant.",
        )
    logger.info(
        "Feedback recorded: message=%s rating=%+d tenant=%s",
        payload.message_id, rating, tenant_id,
    )
    return {"status": "ok", "feedback": saved, "stats": store.feedback_stats(tenant_id)}


@app.get("/api/feedback/stats")
async def feedback_stats(tenant_id: str = Depends(resolve_tenant)):
    """Up/down counts for the calling tenant."""
    return {"status": "ok", "tenant_id": tenant_id, "stats": store.feedback_stats(tenant_id)}


@app.get("/api/feedback/export")
async def feedback_export(
    scope: str = Query("tenant", pattern="^(tenant|all)$",
        description="'tenant' = calling tenant only; 'all' = every tenant"),
    only_positive: bool = Query(False,
        description="Export only up-voted pairs (supervised fine-tuning set)"),
    fmt: str = Query("jsonl", pattern="^(jsonl|json)$"),
    tenant_id: str = Depends(resolve_tenant),
    _admin: bool = Depends(require_admin),
):
    """Admin: export collected feedback as a fine-tuning / preference dataset.

    Each record is a (prompt, response, rating, model_variant) tuple. `fmt=jsonl`
    downloads a newline-delimited file ready for a training pipeline.
    """
    export_tenant = None if scope == "all" else tenant_id
    records = store.export_feedback_dataset(
        tenant_id=export_tenant, only_positive=only_positive,
    )
    if fmt == "jsonl":
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        label = "positive" if only_positive else "all"
        filename = f"feedback_dataset_{scope}_{label}.jsonl"
        return Response(
            content=body,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return {"status": "ok", "count": len(records), "records": records}


@app.get("/api/sustainability/csrd-report")
async def sustainability_csrd_report(
    period_from_iso: str | None = Query(None, description="ISO-8601 start (defaults to last 30d)"),
    period_to_iso: str | None = Query(None, description="ISO-8601 end (defaults to now)"),
    energy_price_usd_kwh: float = Query(0.12, ge=0.0, le=10.0),
    market_based_renewable_pct: float = Query(0.0, ge=0.0, le=1.0,
        description="Fraction of consumption matched by REC/PPA (0..1)"),
    fmt: str = Query("json", pattern="^(json|csv)$", description="Response format"),
    tenant_id: str = Depends(resolve_tenant),
):
    """CSRD ESRS-E1 / GHG Protocol Scope 2 report for the calling tenant.

    Aggregates the HMAC-signed audit log into a CSRD-aligned report. Pass
    `fmt=csv` to download a Watershed/Persefoni/Sweep importable CSV.
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    if not period_to_iso:
        period_to_iso = now.isoformat()
    if not period_from_iso:
        period_from_iso = (now - timedelta(days=30)).isoformat()

    try:
        from_ts = datetime.fromisoformat(period_from_iso.replace("Z", "+00:00")).timestamp()
        to_ts = datetime.fromisoformat(period_to_iso.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid period: {exc}") from exc

    entries = read_audit_log(
        from_ts=from_ts, to_ts=to_ts,
        tenant_filter=tenant_id, max_entries=100_000,
    )
    report = csrd_reporting.build_report(
        entries,
        tenant_id=tenant_id,
        period_from_iso=period_from_iso,
        period_to_iso=period_to_iso,
        energy_price_usd_kwh=energy_price_usd_kwh,
        market_based_renewable_pct=market_based_renewable_pct,
    )
    if fmt == "csv":
        body = csrd_reporting.report_to_csv(report)
        filename = f"csrd_report_{tenant_id}_{period_from_iso[:10]}_{period_to_iso[:10]}.csv"
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Model onboarding (/api/models)
#
# Browse Hugging Face, size a quantization plan against real headroom, download,
# serve, and register. The pipeline exists because measurement showed CSS
# losing to always-full on both axes — the ladder has no genuinely cheaper rung,
# and a router can only be as good as the candidates it ranks.
#
# Route order matters: the static single-segment paths are declared before
# `/{model_id}/...` so a literal "jobs" or "registry" is never parsed as a
# model id.
# ─────────────────────────────────────────────────────────────────────────────

_ONBOARDING_SERVICE: ModelOnboardingService | None = None


def _current_grid_ci() -> float:
    """Live grid carbon intensity, for metering one-off jobs (quantization)."""
    return safe_float(fetch_grid_signal().get("carbon_intensity"), 400.0)


def _resolve_hf_token() -> str:
    """Prefer the encrypted token over the plaintext env var.

    HF_TOKEN in .env is readable by anything that can read the file. HF_TOKEN_ENC
    holds a secret-box token decryptable only with SECRET_KEY, so the token is
    not sitting in plaintext for anything that can read the file.
    """
    enc = os.getenv("HF_TOKEN_ENC", "").strip()
    if enc:
        try:
            return secret_decrypt(enc, SECRET_KEY)
        except ValueError as exc:
            logger.warning("HF_TOKEN_ENC failed to decrypt (%s); falling back to HF_TOKEN", exc)
    return os.getenv("HF_TOKEN", "").strip()


def _get_onboarding_service() -> ModelOnboardingService:
    global _ONBOARDING_SERVICE
    if _ONBOARDING_SERVICE is None:
        _ONBOARDING_SERVICE = ModelOnboardingService(
            get_model_zoo(),
            hf_token=_resolve_hf_token(),
            # The module never imports decision_engine (that would cycle), so
            # grid carbon arrives as an injected callable.
            grid_ci=_current_grid_ci,
            # The API container has no nvidia-smi, so the sidecar is the only
            # in-container source of GPU occupancy. When it reports zeros (its
            # default — DISABLE_GPU_METRICS=1), the planner refuses to size
            # rather than assuming the slice is empty.
            system_metrics=fetch_system_metrics,
            state_path=DATA_DIR / "model_onboarding.json",
        )
    return _ONBOARDING_SERVICE


def _audit_onboarding(action: str, detail: dict[str, Any]) -> None:
    """Onboarding changes what the router may select, so it joins the audit trail."""
    log_decision({"event": "model_onboarding", "action": action, **detail})


@app.get("/api/models/capability")
def api_models_capability() -> dict[str, Any]:
    """What onboarding can actually do on this host, and why not when it cannot."""
    try:
        return _get_onboarding_service().capability()
    except Exception as exc:  # noqa: BLE001 - capability must never 500
        return {"enabled": False, "reason": f"onboarding unavailable: {exc}"}


@app.get("/api/models/catalog/search")
def api_models_search(q: str = "", limit: int = 25, task: str | None = "text-generation") -> dict[str, Any]:
    svc = _get_onboarding_service()
    try:
        return {"results": svc.search(q, limit=limit, task=task or None)}
    except HFError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/models/catalog/preview")
def api_models_preview(
    repo_id: str,
    max_model_len: int = 2048,
    prefer: str | None = None,
    allow_local_quantize: bool = False,
) -> dict[str, Any]:
    """Plan without committing: what would be downloaded, at what size, and why."""
    svc = _get_onboarding_service()
    try:
        return svc.preview(
            repo_id,
            max_model_len=max_model_len,
            prefer=prefer,
            allow_local_quantize=allow_local_quantize,
        )
    except HFError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/models/registry")
def api_models_registry() -> dict[str, Any]:
    return {"models": _get_onboarding_service().registry()}


@app.get("/api/models/jobs")
def api_models_jobs(limit: int = 50) -> dict[str, Any]:
    return {"jobs": _get_onboarding_service().list_jobs(limit=limit)}


@app.get("/api/models/jobs/{job_id}")
def api_models_job(job_id: str) -> dict[str, Any]:
    job = _get_onboarding_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job.to_dict()


@app.post("/api/models/onboard")
async def api_models_onboard(request: Request) -> dict[str, Any]:
    body = await request.json()
    repo_id = str(body.get("repo_id") or "").strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    svc = _get_onboarding_service()
    try:
        job = svc.start(
            repo_id,
            model_id=body.get("model_id"),
            max_model_len=int(body.get("max_model_len") or 2048),
            prefer=body.get("prefer"),
            allow_local_quantize=bool(body.get("allow_local_quantize")),
            auto_serve=bool(body.get("auto_serve")),
            trust_remote_code=bool(body.get("trust_remote_code")),
            donor_id=body.get("donor_id"),
            max_output_tokens=int(body.get("max_output_tokens") or 512),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit_onboarding("start", {"repo_id": repo_id, "job_id": job.job_id, "model_id": job.model_id})
    return job.to_dict()


@app.post("/api/models/quantize")
async def api_models_quantize(request: Request) -> dict[str, Any]:
    """Quantize a model and stop there — no zoo entry, no routing.

    Same pipeline as ``/api/models/onboard`` with the registration step skipped.
    Quantizing a model and adding a rung to this deployment's ladder are separate
    decisions: the checkpoint is a deliverable an operator can download and run
    elsewhere, and forcing it into the registry would leave unavailable entries
    behind for models nobody here intends to serve.

    ``allow_local_quantize`` defaults to true here — unlike onboarding, where a
    pre-quantized sibling is the preferred outcome, a caller asking to *quantize*
    is asking for the calibration pass. It is still metered, and the plan still
    prefers an existing upstream checkpoint when one exists and ``prefer`` allows.
    """
    body = await request.json()
    repo_id = str(body.get("repo_id") or "").strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    svc = _get_onboarding_service()
    try:
        job = svc.start(
            repo_id,
            model_id=body.get("model_id"),
            max_model_len=int(body.get("max_model_len") or 2048),
            prefer=body.get("prefer") or "awq",
            allow_local_quantize=bool(body.get("allow_local_quantize", True)),
            trust_remote_code=bool(body.get("trust_remote_code")),
            register=False,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit_onboarding(
        "quantize_only", {"repo_id": repo_id, "job_id": job.job_id, "model_id": job.model_id}
    )
    return job.to_dict()


@app.get("/api/models/artifacts")
def api_models_artifacts() -> dict[str, Any]:
    """Quantized checkpoints on disk, whether or not they were registered."""
    return {"artifacts": _get_onboarding_service().list_artifacts()}


@app.get("/api/models/artifacts/{artifact_id}")
def api_models_artifact(artifact_id: str) -> dict[str, Any]:
    try:
        artifact = _get_onboarding_service().get_artifact(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"no quantized artifact named {artifact_id!r}")
    return artifact


@app.get("/api/models/artifacts/{artifact_id}/download")
def api_models_artifact_download(artifact_id: str) -> StreamingResponse:
    """Stream the quantized checkpoint as an uncompressed tar.

    Streamed rather than staged: these are multi-gigabyte directories and this
    host runs at 65% full, so building a copy to serve would need the space
    twice. Uncompressed because 4-bit safetensors do not compress — gzip would
    cost minutes of CPU per download for low single-digit percent.
    """
    svc = _get_onboarding_service()
    try:
        filename, chunks = svc.stream_artifact(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _audit_onboarding("artifact_download", {"artifact_id": artifact_id})
    return StreamingResponse(
        chunks,
        media_type="application/x-tar",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/models/artifacts/{artifact_id}")
def api_models_artifact_delete(artifact_id: str, _admin: bool = Depends(require_admin)) -> dict[str, Any]:
    """Delete a quantized checkpoint. Refused while the zoo still references it."""
    try:
        result = _get_onboarding_service().delete_artifact(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit_onboarding("artifact_delete", {"artifact_id": artifact_id, "freed_gb": result["freed_gb"]})
    return result


@app.post("/api/models/{model_id}/serve")
async def api_models_serve(model_id: str, request: Request) -> dict[str, Any]:
    body = await request.json() if await request.body() else {}
    svc = _get_onboarding_service()
    try:
        result = svc.serve(model_id, trust_remote_code=bool(body.get("trust_remote_code")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit_onboarding("serve", {"model_id": model_id, "ok": result["ok"], "detail": result["detail"]})
    return result


@app.post("/api/models/{model_id}/unserve")
def api_models_unserve(model_id: str) -> dict[str, Any]:
    result = _get_onboarding_service().unserve(model_id)
    _audit_onboarding("unserve", {"model_id": model_id, "ok": result["ok"]})
    return result


@app.post("/api/models/{model_id}/measure")
async def api_models_measure(model_id: str, request: Request) -> dict[str, Any]:
    """Promote an onboarded model to routable using measured figures.

    This is the only path that sets ``available: true``. The measurement comes
    from the caller's own evaluation, not from here — a system that can declare
    its own quality has not measured anything.
    """
    body = await request.json()
    measurement = body.get("measurement") or {}
    basis = str(body.get("basis") or "").strip()
    if not basis:
        raise HTTPException(
            status_code=400,
            detail="basis is required — it records where the figures came from (e.g. 'measured:onboard-bench-01')",
        )
    svc = _get_onboarding_service()
    try:
        entry = svc.apply_measurement(model_id, measurement, basis=basis)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _audit_onboarding(
        "measure",
        {
            "model_id": model_id,
            "basis": basis,
            "accuracy_baseline": entry.get("accuracy_baseline"),
            "latency_ms_p50": entry.get("latency_ms_p50"),
            "available": entry.get("available"),
        },
    )
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning (/api/finetune)
#
# Carbon-aware LoRA training: turn collected up-votes into an adapter that makes
# a *small* model good enough at this deployment's traffic that the router can
# stop escalating. The counterpart to model onboarding — onboarding imports a
# better rung, this one improves the one already there.
#
# Training is the largest single compute event the system performs, so it is
# metered and deferred to the cleanest window the forecast offers. Nobody is
# waiting on it, which is what makes that deferral free.
# ─────────────────────────────────────────────────────────────────────────────

_FINETUNE_SERVICE: FineTuningService | None = None


def _ft_feedback_records() -> list[dict[str, Any]]:
    """Up-voted (prompt, response) pairs across every tenant.

    Only positives: a down-vote says what *not* to say, which supervised
    fine-tuning cannot consume.
    """
    return store.export_feedback_dataset(tenant_id=None, only_positive=True)


def _ft_low_carbon_window(duration_hours: float) -> dict[str, Any] | None:
    """Adapt the module's ``duration_hours`` question to the forecast API.

    ``find_low_carbon_window`` takes a forecast list and a deferral budget in ms,
    so the budget is the job's own length plus the configured look-ahead — there
    is no point picking a window that ends before the job does.
    """
    try:
        forecast = get_zone_forecast()
    except Exception as exc:  # noqa: BLE001 - forecast is best-effort
        logger.warning("forecast unavailable for fine-tune scheduling: %s", exc)
        return None
    look_ahead_h = float(os.getenv("FINETUNE_LOOKAHEAD_H", "48"))
    budget_ms = int(max(duration_hours, 0.5) * 3600 * 1000 + look_ahead_h * 3600 * 1000)
    return find_low_carbon_window(forecast, budget_ms)


def _get_finetune_service() -> FineTuningService:
    global _FINETUNE_SERVICE
    if _FINETUNE_SERVICE is None:
        _FINETUNE_SERVICE = FineTuningService(
            get_model_zoo(),
            feedback_records=_ft_feedback_records,
            grid_ci=_current_grid_ci,
            find_low_carbon_window=_ft_low_carbon_window,
            system_metrics=fetch_system_metrics,
            state_path=DATA_DIR / "finetuning.json",
        )
    return _FINETUNE_SERVICE


@app.get("/api/finetune/capability")
def api_finetune_capability() -> dict[str, Any]:
    """What fine-tuning can do here, and what is blocking it when it cannot."""
    try:
        return _get_finetune_service().capability()
    except Exception as exc:  # noqa: BLE001 - capability must never 500
        return {"enabled": False, "reason": f"fine-tuning unavailable: {exc}"}


@app.get("/api/finetune/dataset")
def api_finetune_dataset(_admin: bool = Depends(require_admin)) -> dict[str, Any]:
    """Dataset the collected feedback would produce, with every rejection itemised."""
    return _get_finetune_service().dataset_preview()["stats"]


@app.get("/api/finetune/preview")
def api_finetune_preview(
    base_model_id: str,
    method: str = "qlora",
    lora_rank: int = 16,
    epochs: int = 3,
    _admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """Plan a run without committing: sizing, estimated carbon, and when to start."""
    try:
        return _get_finetune_service().preview(
            base_model_id, method=method, lora_rank=lora_rank, epochs=epochs
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/finetune/jobs")
def api_finetune_jobs(limit: int = 50, _admin: bool = Depends(require_admin)) -> dict[str, Any]:
    return {"jobs": _get_finetune_service().list_jobs(limit=limit)}


@app.get("/api/finetune/jobs/{job_id}")
def api_finetune_job(job_id: str, _admin: bool = Depends(require_admin)) -> dict[str, Any]:
    job = _get_finetune_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job.to_dict()


@app.post("/api/finetune/jobs")
async def api_finetune_submit(request: Request, _admin: bool = Depends(require_admin)) -> dict[str, Any]:
    body = await request.json()
    base_model_id = str(body.get("base_model_id") or "").strip()
    if not base_model_id:
        raise HTTPException(status_code=400, detail="base_model_id is required")
    svc = _get_finetune_service()
    try:
        job = svc.submit(
            base_model_id,
            method=str(body.get("method") or "qlora"),
            lora_rank=int(body.get("lora_rank") or 16),
            epochs=int(body.get("epochs") or 3),
            adapter_id=body.get("adapter_id"),
            force_now=bool(body.get("force_now")),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_decision({"event": "finetune", "action": "submit", "job_id": job.job_id,
                  "base_model_id": base_model_id, "adapter_id": job.adapter_id})
    return job.to_dict()


@app.post("/api/finetune/jobs/{job_id}/cancel")
def api_finetune_cancel(job_id: str, _admin: bool = Depends(require_admin)) -> dict[str, Any]:
    try:
        result = _get_finetune_service().cancel(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    log_decision({"event": "finetune", "action": "cancel", "job_id": job_id, "ok": result["ok"]})
    return result


@app.post("/api/finetune/adapters/{adapter_id}/serve")
def api_finetune_serve(adapter_id: str, _admin: bool = Depends(require_admin)) -> dict[str, Any]:
    """Serve base + adapter through one vLLM container (--enable-lora)."""
    try:
        result = _get_finetune_service().serve_adapter(adapter_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_decision({"event": "finetune", "action": "serve", "adapter_id": adapter_id, "ok": result["ok"]})
    return result


@app.post("/api/finetune/adapters/{adapter_id}/measure")
async def api_finetune_measure(
    adapter_id: str, request: Request, _admin: bool = Depends(require_admin)
) -> dict[str, Any]:
    """Promote a measured adapter to routable, and record its carbon payback.

    The only path that sets ``available: true`` for an adapter. Fine-tuning
    exists because quality is expected to change, so assuming the direction of
    that change is exactly the thing this refuses to do.
    """
    body = await request.json()
    basis = str(body.get("basis") or "").strip()
    if not basis:
        raise HTTPException(
            status_code=400,
            detail="basis is required — it records where the figures came from (e.g. 'measured:eval-01')",
        )
    try:
        entry = _get_finetune_service().apply_measurement(
            adapter_id, body.get("measurement") or {}, basis=basis
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_decision({
        "event": "finetune", "action": "measure", "adapter_id": adapter_id, "basis": basis,
        "accuracy_baseline": entry.get("accuracy_baseline"),
        "training_payback": entry.get("training_payback"),
    })
    return entry


if __name__ == "__main__":
    import uvicorn
    # Allows the script to be run directly for testing without Docker
    uvicorn.run("decision_engine:app", host="0.0.0.0", port=8100, reload=True)
