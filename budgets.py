"""
Per-tenant budgets — Adaptive Green AI

Enforces token + cost caps per tenant with soft-warn and hard-block tiers.
JSON-backed config in `data/tenant_budgets.json` so admins can hot-edit it.

Usage rollups are computed on demand from the audit log (single source of
truth — no separate counter to drift). For high-throughput deployments the
caller can pass an explicit usage snapshot to avoid log scans.

Schema (data/tenant_budgets.json)
─────────────────────────────────
{
  "default_budget": {
    "monthly_token_limit":   2000000,
    "daily_token_limit":     200000,
    "monthly_cost_usd_limit": 50.0,
    "soft_warn_pct":         0.80,
    "hard_block":            true
  },
  "tenants": {
    "acme":   { "monthly_token_limit": 5000000, "hard_block": true,  ... },
    "lab":    { "monthly_token_limit": 100000,  "hard_block": false, ... }
  }
}
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FALLBACK_DATA_DIR = Path(os.getenv("FALLBACK_DATA_DIR", "/tmp/green-ai"))


def _resolve_writable(target: Path) -> Path:
    """Return target if its parent dir is writable, else FALLBACK_DATA_DIR/<name>."""
    for candidate in (target, FALLBACK_DATA_DIR / target.name):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            # Touch a probe to test write access without clobbering existing data
            probe = candidate.parent / f".write_probe_{os.getpid()}"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    return target


BUDGETS_PATH = _resolve_writable(
    Path(os.getenv("TENANT_BUDGETS_PATH", DATA_DIR / "tenant_budgets.json"))
)

# Reasonable open defaults — admins tighten per-tenant in the JSON.
_DEFAULT_BUDGET = {
    "monthly_token_limit": 2_000_000,
    "daily_token_limit":   200_000,
    "monthly_cost_usd_limit": 50.0,
    "soft_warn_pct": 0.80,
    "hard_block": True,
}

# Cloud-equivalent pricing used to dollarize tenant usage (matches observability)
COST_INPUT_USD_PER_1K = float(os.getenv("BUDGET_INPUT_USD_PER_1K", "0.0015"))
COST_OUTPUT_USD_PER_1K = float(os.getenv("BUDGET_OUTPUT_USD_PER_1K", "0.0020"))


_lock = threading.RLock()
_state: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        if BUDGETS_PATH.exists():
            try:
                _state = json.loads(BUDGETS_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Tenant budgets file corrupt, using defaults: %s", exc)
                _state = {"default_budget": dict(_DEFAULT_BUDGET), "tenants": {}}
        else:
            _state = {"default_budget": dict(_DEFAULT_BUDGET), "tenants": {}}
            _save_unlocked()
        _state.setdefault("default_budget", dict(_DEFAULT_BUDGET))
        _state.setdefault("tenants", {})
        return _state


def _save_unlocked() -> None:
    BUDGETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUDGETS_PATH.write_text(
        json.dumps(_state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_budget(tenant_id: str) -> dict[str, Any]:
    """Return the effective budget for a tenant (per-tenant overrides + defaults)."""
    state = _load()
    base = dict(state["default_budget"])
    override = state["tenants"].get(tenant_id) or {}
    base.update(override)
    base["tenant_id"] = tenant_id
    base["overridden"] = bool(override)
    return base


def list_budgets() -> dict[str, Any]:
    state = _load()
    return {
        "default_budget": dict(state["default_budget"]),
        "tenants": {
            tid: get_budget(tid) for tid in sorted(state["tenants"].keys())
        },
    }


def set_budget(tenant_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    state = _load()
    with _lock:
        existing = state["tenants"].get(tenant_id) or {}
        cleaned: dict[str, Any] = {}
        for k, v in fields.items():
            if k not in _DEFAULT_BUDGET:
                continue
            if k == "hard_block":
                cleaned[k] = bool(v)
            elif k == "soft_warn_pct":
                cleaned[k] = max(0.0, min(1.0, float(v)))
            else:
                cleaned[k] = max(0, float(v))
        existing.update(cleaned)
        state["tenants"][tenant_id] = existing
        _save_unlocked()
    return get_budget(tenant_id)


def delete_budget(tenant_id: str) -> bool:
    state = _load()
    with _lock:
        if tenant_id not in state["tenants"]:
            return False
        state["tenants"].pop(tenant_id)
        _save_unlocked()
    return True


def compute_usage(audit_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token + cost usage from audit-log entries (already filtered to scope)."""
    tokens_in = 0
    tokens_out = 0
    requests = 0
    co2_g = 0.0
    for e in audit_entries:
        toks = e.get("tokens") or {}
        tokens_in += int(_safe_num(toks.get("input"), 0))
        tokens_out += int(_safe_num(toks.get("output"), 0))
        co2_g += float(_safe_num(e.get("system_co2_g"), 0.0))
        requests += 1
    cost_usd = (
        (tokens_in / 1000.0) * COST_INPUT_USD_PER_1K
        + (tokens_out / 1000.0) * COST_OUTPUT_USD_PER_1K
    )
    return {
        "requests":     requests,
        "tokens_in":    tokens_in,
        "tokens_out":   tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "co2_g":        round(co2_g, 6),
        "cost_usd":     round(cost_usd, 6),
    }


def _safe_num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def usage_for_tenant(
    tenant_id: str,
    audit_reader,
    *,
    monthly: bool = True,
) -> dict[str, Any]:
    """Compute monthly + daily usage for `tenant_id` by scanning audit log.

    `audit_reader` is `read_audit_log` injected from decision_engine to avoid
    a circular import. It must accept (from_ts, max_entries, tenant_filter).
    """
    import time
    now_ts = time.time()
    day_ago = now_ts - 24 * 3600
    month_ago = now_ts - 30 * 24 * 3600
    month_entries = audit_reader(
        from_ts=month_ago, max_entries=20000, tenant_filter=tenant_id
    )
    day_entries = [
        e for e in month_entries
        if _epoch_of(e) >= day_ago
    ]
    return {
        "tenant_id": tenant_id,
        "as_of":     _now_iso(),
        "month":     compute_usage(month_entries),
        "day":       compute_usage(day_entries),
    }


def _epoch_of(entry: dict[str, Any]) -> float:
    try:
        return datetime.fromisoformat(
            entry.get("timestamp", "").replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_budget(
    tenant_id: str,
    estimated_input_tokens: int,
    audit_reader,
) -> dict[str, Any]:
    """Pre-inference check.

    Returns dict with keys:
        allowed       : bool   — False means block the request.
        reason        : str    — one of [ok, soft-warn, daily-token-cap,
                                 monthly-token-cap, monthly-cost-cap]
        budget        : dict   — effective limits for the tenant
        usage         : dict   — {month: {...}, day: {...}}
        projected_*   : numeric headroom estimates
    """
    budget = get_budget(tenant_id)
    usage = usage_for_tenant(tenant_id, audit_reader)
    month = usage["month"]
    day = usage["day"]

    monthly_token_limit = float(budget.get("monthly_token_limit") or 0)
    daily_token_limit = float(budget.get("daily_token_limit") or 0)
    monthly_cost_usd_limit = float(budget.get("monthly_cost_usd_limit") or 0)
    soft_pct = float(budget.get("soft_warn_pct") or 0.8)
    hard_block = bool(budget.get("hard_block", True))

    projected_monthly_tokens = month["tokens_total"] + max(estimated_input_tokens, 0)
    projected_daily_tokens = day["tokens_total"] + max(estimated_input_tokens, 0)

    pct_monthly = (
        projected_monthly_tokens / monthly_token_limit if monthly_token_limit > 0 else 0.0
    )
    pct_daily = (
        projected_daily_tokens / daily_token_limit if daily_token_limit > 0 else 0.0
    )
    pct_cost = (
        month["cost_usd"] / monthly_cost_usd_limit if monthly_cost_usd_limit > 0 else 0.0
    )

    reason = "ok"
    allowed = True
    if monthly_token_limit > 0 and projected_monthly_tokens > monthly_token_limit:
        reason = "monthly-token-cap"
        allowed = not hard_block
    elif daily_token_limit > 0 and projected_daily_tokens > daily_token_limit:
        reason = "daily-token-cap"
        allowed = not hard_block
    elif monthly_cost_usd_limit > 0 and month["cost_usd"] > monthly_cost_usd_limit:
        reason = "monthly-cost-cap"
        allowed = not hard_block
    elif max(pct_monthly, pct_daily, pct_cost) >= soft_pct:
        reason = "soft-warn"

    return {
        "allowed":   allowed,
        "reason":    reason,
        "tenant_id": tenant_id,
        "budget":    budget,
        "usage":     usage,
        "headroom": {
            "tokens_month_pct": round(pct_monthly * 100, 2),
            "tokens_day_pct":   round(pct_daily * 100, 2),
            "cost_month_pct":   round(pct_cost * 100, 2),
        },
    }
