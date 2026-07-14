"""
Semantic response cache — Adaptive Green AI

Skips vLLM inference when a near-duplicate prompt was recently answered
for the same tenant. Uses the same SentenceTransformer that powers the
prompt profiler so we don't load a second embedding model.

Cache record
────────────
{
  "tenant_id": "...",
  "embedding": [...],          # normalized, dim=384 (MiniLM) or 256 (hash fallback)
  "prompt_preview": "...",
  "response": "...",
  "model_variant": "medium",
  "input_tokens": 42, "output_tokens": 218,
  "attachment_fingerprint": "sha256(name+size+sha256(bytes))",
  "created_at": 1731232323.4,
  "expires_at": 1731235923.4,
  "hit_count": 0
}

Stored in `data/semantic_cache.json`. Max entries (per tenant) keep the
file bounded; LRU eviction by created_at when full.
"""

from __future__ import annotations

import hashlib
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
DATA_DIR = BASE_DIR / "data"
FALLBACK_DATA_DIR = Path(os.getenv("FALLBACK_DATA_DIR", "/tmp/green-ai"))


def _resolve_writable(target: Path) -> Path:
    for candidate in (target, FALLBACK_DATA_DIR / target.name):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".write_probe_{os.getpid()}"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    return target


CACHE_PATH = _resolve_writable(
    Path(os.getenv("SEMANTIC_CACHE_PATH", DATA_DIR / "semantic_cache.json"))
)

# ── Tunables ─────────────────────────────────────────────────────────────────
ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_S", "3600"))
MAX_ENTRIES_PER_TENANT = int(os.getenv("SEMANTIC_CACHE_MAX_PER_TENANT", "200"))
MAX_PROMPT_PREVIEW = 400


_lock = threading.RLock()
_cache: dict[str, list[dict[str, Any]]] | None = None    # tenant_id -> entries
_stats: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0, "stores": 0}


# ── Embedding helpers (re-uses routing_policies infra) ───────────────────────

def _embed(text: str) -> list[float]:
    """Embed via SentenceTransformer if available, hashed fallback otherwise."""
    try:
        from routing_policies import _embed_texts
        return _embed_texts([text])[0]
    except Exception:
        return _hash_embed(text)


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        idx = sum(ord(c) for c in tok) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ── Persistence ──────────────────────────────────────────────────────────────

def _load() -> dict[str, list[dict[str, Any]]]:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        if CACHE_PATH.exists():
            try:
                _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                if not isinstance(_cache, dict):
                    _cache = {}
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Semantic cache file corrupt; starting fresh: %s", exc)
                _cache = {}
        else:
            _cache = {}
        return _cache


def _save_unlocked() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(_cache, ensure_ascii=False), encoding="utf-8"
    )


# ── Fingerprint ──────────────────────────────────────────────────────────────

def attachment_fingerprint(attachments: list[dict[str, Any]] | None) -> str:
    """Stable hash of the attachment set so cache hits require identical inputs."""
    if not attachments:
        return "none"
    h = hashlib.sha256()
    for a in sorted(attachments, key=lambda x: (x.get("name") or "")):
        name = (a.get("name") or "").encode()
        size = str(a.get("size_bytes") or 0).encode()
        ctype = (a.get("content_type") or "").encode()
        text = (a.get("context_text") or a.get("excerpt") or "").encode("utf-8", "ignore")
        h.update(name + b"|" + size + b"|" + ctype + b"|")
        h.update(hashlib.sha256(text).digest())
        h.update(b"||")
    return h.hexdigest()


# ── Conversation-context awareness ───────────────────────────────────────────
# A bare-prompt embedding match is unsafe for *followups* — a short or anaphoric
# turn ("why?", "explain that", "the second one") embeds close to the same
# generic followup in any other thread, so a cross-conversation hit returns text
# that answers a different question. For these turns we require the cached entry
# to come from the same conversation AND the same recent-history fingerprint, so
# a reused answer was conditioned on the same preceding context. Standalone
# queries ("capital of france") are unaffected and still hit cross-conversation.

_FOLLOWUP_STARTERS = (
    "why", "how about", "what about", "and ", "but ", "so ", "then ", "also ",
    "explain", "elaborate", "continue", "go on", "expand", "tell me more",
    "more on", "which one", "the first", "the second", "the third", "the last",
    "the other", "that one", "this one", "what else", "anything else",
    "give me an example", "for example", "same ",
)
_ANAPHORA = (
    " it ", " it.", " it?", " it,", " its ", " that ", " that.", " that?",
    " this ", " they ", " them ", " those ", " these ", " he ", " she ",
    " his ", " her ", " their ",
)


def is_context_dependent_followup(prompt: str) -> bool:
    """Heuristic: does this turn only make sense given the conversation so far?

    Biased toward True — a false positive merely scopes the lookup to the same
    thread (always safe); a false negative would serve a cross-context answer.
    """
    p = (prompt or "").strip().lower()
    if not p:
        return False
    if len(p.split()) <= 2:  # "why", "go on", "and?" — almost always a continuation
        return True
    if any(p.startswith(starter) for starter in _FOLLOWUP_STARTERS):
        return True
    padded = f" {p} "
    return any(token in padded for token in _ANAPHORA)


def recent_history_fingerprint(
    history: list[dict[str, Any]] | None,
    keep_recent: int = 4,
) -> str:
    """Stable hash of the last ``keep_recent`` turns — the context an answer was
    conditioned on. Compute once per request and reuse for both lookup + store."""
    if not history:
        return "empty"
    payload = [
        {"role": m.get("role"), "content": (m.get("content") or "").strip()[:300]}
        for m in history[-keep_recent:]
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ── Public API ───────────────────────────────────────────────────────────────

def lookup(
    tenant_id: str,
    prompt: str,
    attachments: list[dict[str, Any]] | None = None,
    *,
    conversation_id: str | None = None,
    recent_history_fp: str | None = None,
    is_followup: bool | None = None,
) -> dict[str, Any] | None:
    """Return cached entry if a close enough match exists, else None.

    For context-dependent followups the match is additionally scoped to the same
    ``conversation_id`` and ``recent_history_fp`` so a reused answer was generated
    against the same preceding context (see ``is_context_dependent_followup``).
    """
    if not ENABLED:
        return None
    if is_followup is None:
        is_followup = is_context_dependent_followup(prompt)
    cache = _load()
    fp = attachment_fingerprint(attachments)
    query_emb = _embed(prompt or "")
    now = time.time()
    best: dict[str, Any] | None = None
    best_sim = 0.0

    with _lock:
        entries = cache.get(tenant_id) or []
        # purge expired in-place
        live = [e for e in entries if float(e.get("expires_at", 0)) > now]
        if len(live) != len(entries):
            cache[tenant_id] = live
        for entry in live:
            if entry.get("attachment_fingerprint") != fp:
                continue
            if is_followup:
                # Only reuse an answer from the same thread + same recent context;
                # otherwise the cached text answers a different conversation.
                if entry.get("conversation_id") != conversation_id:
                    continue
                if entry.get("recent_history_fp") != recent_history_fp:
                    continue
            emb = entry.get("embedding") or []
            if len(emb) != len(query_emb):
                continue
            sim = _cosine(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best and best_sim >= SIMILARITY_THRESHOLD:
            best["hit_count"] = int(best.get("hit_count", 0)) + 1
            _stats["hits"] += 1
            _save_unlocked()
            return {
                "response":          best["response"],
                "model_variant":     best.get("model_variant"),
                "input_tokens":      int(best.get("input_tokens", 0)),
                "output_tokens":     int(best.get("output_tokens", 0)),
                "similarity":        round(best_sim, 4),
                "cached_prompt":     best.get("prompt_preview", ""),
                "cached_at_iso":     _iso(best.get("created_at", now)),
                "hit_count":         best["hit_count"],
            }
    _stats["misses"] += 1
    return None


def store(
    tenant_id: str,
    prompt: str,
    response: str,
    *,
    model_variant: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    attachments: list[dict[str, Any]] | None = None,
    ttl_seconds: int | None = None,
    conversation_id: str | None = None,
    recent_history_fp: str | None = None,
) -> None:
    if not ENABLED or not (response or "").strip():
        return
    fp = attachment_fingerprint(attachments)
    emb = _embed(prompt or "")
    now = time.time()
    ttl = int(ttl_seconds if ttl_seconds is not None else TTL_SECONDS)
    entry = {
        "embedding":              emb,
        "prompt_preview":         (prompt or "")[:MAX_PROMPT_PREVIEW],
        "response":               response,
        "model_variant":          model_variant,
        "input_tokens":           int(input_tokens),
        "output_tokens":          int(output_tokens),
        "attachment_fingerprint": fp,
        # Conversation scoping for followup hits (see lookup()). Stored on every
        # entry so a future context-dependent turn can require an exact match.
        "conversation_id":        conversation_id,
        "recent_history_fp":      recent_history_fp,
        "created_at":             now,
        "expires_at":             now + ttl,
        "hit_count":              0,
    }
    cache = _load()
    with _lock:
        bucket = cache.setdefault(tenant_id, [])
        bucket.append(entry)
        # LRU eviction by created_at
        if len(bucket) > MAX_ENTRIES_PER_TENANT:
            bucket.sort(key=lambda e: float(e.get("created_at", 0)), reverse=True)
            evicted = len(bucket) - MAX_ENTRIES_PER_TENANT
            del bucket[MAX_ENTRIES_PER_TENANT:]
            _stats["evictions"] += evicted
        _stats["stores"] += 1
        _save_unlocked()


def invalidate(
    tenant_id: str,
    prompt: str,
    attachments: list[dict[str, Any]] | None = None,
) -> int:
    """Drop cached entries that closely match *prompt* (same attachment set).

    Used when a hit is deemed unservable — e.g. the cached answer came from a
    weaker model than the prompt now routes to. Removing the stale entry lets
    the regenerated (stronger) answer take its place instead of the stale entry
    winning the similarity tie forever. Returns the number of entries removed.
    """
    if not ENABLED:
        return 0
    fp = attachment_fingerprint(attachments)
    query_emb = _embed(prompt or "")
    removed = 0
    with _lock:
        entries = _load().get(tenant_id) or []
        kept: list[dict[str, Any]] = []
        for entry in entries:
            emb = entry.get("embedding") or []
            if (
                entry.get("attachment_fingerprint") == fp
                and len(emb) == len(query_emb)
                and _cosine(query_emb, emb) >= SIMILARITY_THRESHOLD
            ):
                removed += 1
                continue
            kept.append(entry)
        if removed:
            _load()[tenant_id] = kept
            _stats["invalidations"] = _stats.get("invalidations", 0) + removed
            _save_unlocked()
    return removed


def status(tenant_id: str | None = None) -> dict[str, Any]:
    cache = _load()
    with _lock:
        if tenant_id:
            entries = cache.get(tenant_id) or []
            return {
                "tenant_id":     tenant_id,
                "entry_count":   len(entries),
                "max_per_tenant": MAX_ENTRIES_PER_TENANT,
                "ttl_seconds":   TTL_SECONDS,
                "threshold":     SIMILARITY_THRESHOLD,
                "enabled":       ENABLED,
                "stats":         dict(_stats),
            }
        per_tenant = {tid: len(v) for tid, v in cache.items()}
        return {
            "enabled":         ENABLED,
            "tenant_count":    len(cache),
            "total_entries":   sum(per_tenant.values()),
            "per_tenant":      per_tenant,
            "max_per_tenant":  MAX_ENTRIES_PER_TENANT,
            "ttl_seconds":     TTL_SECONDS,
            "threshold":       SIMILARITY_THRESHOLD,
            "stats":           dict(_stats),
        }


def clear(tenant_id: str | None = None) -> int:
    """Clear cache for a tenant (or all). Returns number of entries removed."""
    cache = _load()
    with _lock:
        if tenant_id is None:
            removed = sum(len(v) for v in cache.values())
            cache.clear()
        else:
            removed = len(cache.get(tenant_id) or [])
            cache.pop(tenant_id, None)
        _save_unlocked()
    return removed


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""
