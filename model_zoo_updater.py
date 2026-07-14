#!/usr/bin/env python3
"""
Model Zoo Auto-Updater — Adaptive Green AI

Pulls model-registry entries from a trusted external source, validates each
candidate against the schema enforced by the Decision Engine, and stages
new/updated entries in ``config/model_zoo_pending.json`` for human approval.

Design constraints (intentionally conservative):
  - Off by default. Set ``MODEL_ZOO_UPDATE_ENABLED=true`` to enable.
  - Source must be allowlisted via ``MODEL_ZOO_UPDATE_SOURCE`` (exact URL).
  - Only ``https://`` URLs are accepted (or ``file://`` for local testing).
  - Every fetched entry is schema-validated and bounded (no negative power,
    plausible TDP, plausible hardware efficiency, etc.).
  - Nothing ever applies automatically. Approval is an explicit POST to
    ``/api/model-zoo/updates/{id}/approve`` (admin-gated).
  - Audit trail of fetch / approve / reject is appended to
    ``data/model_zoo_updates.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from model_zoo import get_model_zoo, MODEL_ZOO_PATH

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

# ── Config ────────────────────────────────────────────────────────────────────

UPDATE_ENABLED = os.getenv("MODEL_ZOO_UPDATE_ENABLED", "false").lower() in {"1", "true", "yes"}
UPDATE_SOURCE = os.getenv("MODEL_ZOO_UPDATE_SOURCE", "").strip()
UPDATE_INTERVAL_S = int(os.getenv("MODEL_ZOO_UPDATE_INTERVAL_S", "3600"))
UPDATE_REQUEST_TIMEOUT_S = int(os.getenv("MODEL_ZOO_UPDATE_TIMEOUT_S", "20"))

PENDING_PATH = Path(os.getenv("MODEL_ZOO_PENDING_PATH", CONFIG_DIR / "model_zoo_pending.json"))
AUDIT_PATH = Path(os.getenv("MODEL_ZOO_UPDATE_AUDIT", DATA_DIR / "model_zoo_updates.jsonl"))

# Required keys every staged entry must define before it can be approved.
_REQUIRED_FIELDS = (
    "id", "model_id", "model_variant", "architecture",
    "parameter_count_b", "flop_count_per_token",
    "hardware", "hardware_class", "region",
    "accuracy_baseline", "latency_ms_p50",
    "power_tdp_w", "hardware_efficiency", "pue",
    "mfg_carbon_kg", "device_lifetime_years",
)

# Hard bounds — entries violating these are rejected at fetch time.
_BOUNDS = {
    "parameter_count_b":      (0.001, 2000.0),
    "flop_count_per_token":   (1e6, 1e14),
    "accuracy_baseline":      (0.0, 1.0),
    "latency_ms_p50":         (1.0, 60_000.0),
    "power_tdp_w":            (1.0, 2000.0),
    "hardware_efficiency":    (0.05, 1.0),
    "pue":                    (1.0, 3.0),
    "mfg_carbon_kg":          (0.0, 5000.0),
    "device_lifetime_years":  (0.5, 20.0),
    "region_carbon_multiplier": (0.1, 5.0),
}

_ALLOWED_SCHEMES = {"https", "file"}


# ── Updater service ──────────────────────────────────────────────────────────

class ModelZooUpdater:
    """Polls a trusted source and stages candidate entries for human approval."""

    def __init__(
        self,
        source_url: str = UPDATE_SOURCE,
        pending_path: str | Path = PENDING_PATH,
        audit_path: str | Path = AUDIT_PATH,
    ):
        self._source_url = source_url
        self._pending_path = Path(pending_path)
        self._audit_path = Path(audit_path)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_check: dict[str, Any] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._load_pending()

    # ─────────────── Public API ────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled":   UPDATE_ENABLED,
                "source":    _redact_source(self._source_url),
                "interval_s": UPDATE_INTERVAL_S,
                "pending_count": len(self._pending),
                "last_check": dict(self._last_check),
                "running":   bool(self._thread and self._thread.is_alive()),
            }

    def list_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._pending.values()]

    def check_now(self) -> dict[str, Any]:
        """Trigger one fetch+validate cycle synchronously. Returns a summary."""
        return self._fetch_and_stage()

    def approve(self, update_id: str, approver: str = "admin") -> dict[str, Any]:
        """Apply a pending update to the live model zoo (admin-gated)."""
        with self._lock:
            staged = self._pending.get(update_id)
            if not staged:
                return {"status": "not_found", "id": update_id}
            entry = staged["entry"]

        zoo = get_model_zoo()
        try:
            zoo.register_model(entry)
        except Exception as exc:  # noqa: BLE001 — surface to caller
            return {"status": "error", "id": update_id, "error": str(exc)}

        with self._lock:
            self._pending.pop(update_id, None)
            self._save_pending_unlocked()

        self._append_audit({
            "event":    "approved",
            "id":       update_id,
            "model_id": entry.get("id"),
            "approver": approver,
        })
        return {"status": "approved", "id": update_id, "model_id": entry.get("id")}

    def reject(self, update_id: str, reviewer: str = "admin", reason: str = "") -> dict[str, Any]:
        """Drop a pending update without applying it."""
        with self._lock:
            staged = self._pending.pop(update_id, None)
            if staged is None:
                return {"status": "not_found", "id": update_id}
            self._save_pending_unlocked()

        self._append_audit({
            "event":    "rejected",
            "id":       update_id,
            "model_id": staged.get("entry", {}).get("id"),
            "reviewer": reviewer,
            "reason":   reason,
        })
        return {"status": "rejected", "id": update_id}

    # ─────────────── Background loop ───────────────────────────────────────

    def start(self) -> None:
        if not UPDATE_ENABLED:
            logger.info("Model Zoo updater disabled (MODEL_ZOO_UPDATE_ENABLED=false)")
            return
        if not self._source_url:
            logger.warning("Model Zoo updater enabled but MODEL_ZOO_UPDATE_SOURCE is empty")
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="zoo-updater", daemon=True)
        self._thread.start()
        logger.info("Model Zoo updater started (interval=%ss)", UPDATE_INTERVAL_S)

    def stop(self) -> None:
        self._stop.set()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._fetch_and_stage()
            except Exception as exc:  # noqa: BLE001 — never crash the loop
                logger.exception("Zoo updater tick failed: %s", exc)
            self._stop.wait(UPDATE_INTERVAL_S)

    # ─────────────── Fetch + validate ──────────────────────────────────────

    def _fetch_and_stage(self) -> dict[str, Any]:
        if not self._source_url:
            return {"status": "no_source"}
        if not _is_allowlisted(self._source_url):
            return {"status": "source_blocked", "source": _redact_source(self._source_url)}

        try:
            payload = _fetch_json(self._source_url)
        except Exception as exc:  # noqa: BLE001
            err = {"status": "fetch_failed", "error": str(exc)}
            with self._lock:
                self._last_check = {**err, "at": _utc_now_iso()}
            return err

        candidates = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(candidates, list):
            err = {"status": "invalid_payload", "detail": "missing 'models' list"}
            with self._lock:
                self._last_check = {**err, "at": _utc_now_iso()}
            return err

        zoo = get_model_zoo()
        live_by_id = {m.get("id"): m for m in zoo.list_models()}

        accepted: list[str] = []
        rejected: list[dict[str, Any]] = []
        unchanged: list[str] = []
        with self._lock:
            for raw in candidates:
                if not isinstance(raw, dict):
                    rejected.append({"id": None, "reason": "not_an_object"})
                    continue
                ok, reason = _validate_entry(raw)
                if not ok:
                    rejected.append({"id": raw.get("id"), "reason": reason})
                    continue
                model_id = str(raw["id"])
                if model_id in live_by_id and _entries_equal(raw, live_by_id[model_id]):
                    unchanged.append(model_id)
                    continue
                update_id = _stable_update_id(model_id, raw)
                self._pending[update_id] = {
                    "id":         update_id,
                    "model_id":   model_id,
                    "fetched_at": _utc_now_iso(),
                    "source":     _redact_source(self._source_url),
                    "change":     "update" if model_id in live_by_id else "new",
                    "diff":       _summarise_diff(raw, live_by_id.get(model_id, {})),
                    "entry":      raw,
                }
                accepted.append(update_id)

            self._save_pending_unlocked()

        summary = {
            "status":    "ok",
            "at":        _utc_now_iso(),
            "accepted":  accepted,
            "rejected":  rejected,
            "unchanged": unchanged,
            "pending_count": len(self._pending),
        }
        with self._lock:
            self._last_check = summary
        self._append_audit({"event": "fetch", "summary": {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "unchanged_count": len(unchanged),
        }})
        return summary

    # ─────────────── Persistence ───────────────────────────────────────────

    def _load_pending(self) -> None:
        if not self._pending_path.exists():
            self._pending = {}
            return
        try:
            payload = json.loads(self._pending_path.read_text(encoding="utf-8"))
            self._pending = {str(k): v for k, v in (payload.get("pending") or {}).items()}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Pending zoo updates corrupt, resetting: %s", exc)
            self._pending = {}

    def _save_pending_unlocked(self) -> None:
        try:
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            self._pending_path.write_text(
                json.dumps(
                    {"saved_at": _utc_now_iso(), "pending": self._pending},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to write pending zoo updates: %s", exc)

    def _append_audit(self, record: dict[str, Any]) -> None:
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"ts": _utc_now_iso(), **record}, ensure_ascii=False)
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logger.warning("Failed to append zoo update audit entry: %s", exc)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_allowlisted(url: str) -> bool:
    """Only the exact URL configured in env is honoured. Scheme limited to https/file."""
    if not url or url != UPDATE_SOURCE:
        return False
    parsed = urlparse(url)
    return parsed.scheme in _ALLOWED_SCHEMES and bool(parsed.netloc or parsed.path)


def _fetch_json(url: str) -> Any:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return json.loads(Path(parsed.path).read_text(encoding="utf-8"))
    resp = requests.get(url, timeout=UPDATE_REQUEST_TIMEOUT_S, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()


def _validate_entry(entry: dict[str, Any]) -> tuple[bool, str]:
    for key in _REQUIRED_FIELDS:
        if key not in entry:
            return False, f"missing_field:{key}"

    mid = str(entry.get("id", "")).strip()
    if not mid or len(mid) > 128 or any(ch in mid for ch in (" ", "\t", "\n")):
        return False, "invalid_id"

    for key, (lo, hi) in _BOUNDS.items():
        if key in entry:
            try:
                v = float(entry[key])
            except (TypeError, ValueError):
                return False, f"non_numeric:{key}"
            if not (lo <= v <= hi):
                return False, f"out_of_bounds:{key}={v}"

    if entry.get("moe"):
        try:
            n = int(entry.get("num_experts", 0))
            k = int(entry.get("active_experts_k", 0))
        except (TypeError, ValueError):
            return False, "moe_topology_non_int"
        if n <= 0 or k <= 0 or k > n:
            return False, "invalid_moe_topology"

    return True, "ok"


def _entries_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _canonical_hash(a) == _canonical_hash(b)


def _canonical_hash(entry: dict[str, Any]) -> str:
    serialised = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _stable_update_id(model_id: str, entry: dict[str, Any]) -> str:
    return f"{model_id}:{_canonical_hash(entry)[:12]}"


def _summarise_diff(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    if not old:
        return {"kind": "new", "fields": sorted(new.keys())}
    changed: dict[str, dict[str, Any]] = {}
    for key, new_val in new.items():
        old_val = old.get(key)
        if old_val != new_val:
            changed[key] = {"old": old_val, "new": new_val}
    return {"kind": "update", "changed": changed}


def _redact_source(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        return f"{parsed.scheme}://<redacted>@{parsed.hostname or ''}{parsed.path}"
    return url


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Module-level singleton ────────────────────────────────────────────────────

_updater_instance: ModelZooUpdater | None = None
_updater_lock = threading.Lock()


def get_zoo_updater() -> ModelZooUpdater:
    global _updater_instance
    if _updater_instance is None:
        with _updater_lock:
            if _updater_instance is None:
                _updater_instance = ModelZooUpdater()
    return _updater_instance
