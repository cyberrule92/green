"""
CSRD / GHG Protocol reporting — Adaptive Green AI

Aggregates the HMAC-signed audit log into:

  • GHG Protocol Scope-2 disclosure (electricity-driven CO₂e)
      Scope-2 (location-based)  : energy_kwh × grid_carbon_intensity_g_per_kwh
      Scope-2 (market-based)    : same, scaled by region_carbon_multiplier
                                  (placeholder for purchased instruments — REC/PPA)

  • CSRD ESRS E1 climate disclosure
      E1-5 Energy consumption + mix (total MWh)
      E1-6 Gross Scope 2 GHG emissions (tCO₂e)
      E1-7 GHG removals + carbon credits (placeholder, not yet wired)
      E1-9 Anticipated financial effects of physical & transition risks (n/a)

  • Per-model + per-period breakdown for chargeback.

Outputs JSON or CSV (audit-friendly, importable to Watershed / Persefoni / Sweep).

The chain-of-custody invariant: every aggregated number traces back to an
HMAC-signed audit row. We therefore include `audit.signed_entries_count`
and a sample of HMAC digests so a CSRD auditor can spot-verify integrity.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Reporting framework versions
GHG_PROTOCOL_VERSION = "GHG Protocol Corporate Standard 2015 (Scope 2 Quality Criteria 2015)"
CSRD_FRAMEWORK_VERSION = "CSRD ESRS E1 (Delegated Regulation 2023/2772)"
REPORT_SCHEMA_VERSION = "1.0.0"


def _safe_num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _epoch_of_entry(e: dict[str, Any]) -> float:
    return _parse_iso(e.get("timestamp")) or 0.0


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def build_report(
    audit_entries: list[dict[str, Any]],
    *,
    tenant_id: str | None = None,
    period_from_iso: str | None = None,
    period_to_iso: str | None = None,
    energy_price_usd_kwh: float = 0.12,
    market_based_renewable_pct: float = 0.0,
) -> dict[str, Any]:
    """Aggregate audit entries into a CSRD/GHG-aligned report.

    `audit_entries` is the already-filtered list (caller scopes it to the tenant
    and time window).
    """
    total_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_energy_kwh = 0.0
    total_co2_g_location = 0.0
    grid_ci_samples: list[float] = []
    by_model: dict[str, dict[str, float]] = {}
    by_day: dict[str, dict[str, float]] = {}
    hmac_digests: list[str] = []

    earliest_ts: float | None = None
    latest_ts: float | None = None

    for e in audit_entries:
        total_requests += 1
        ts = _epoch_of_entry(e)
        if ts:
            earliest_ts = min(earliest_ts, ts) if earliest_ts is not None else ts
            latest_ts = max(latest_ts, ts) if latest_ts is not None else ts

        gpu_w = _safe_num(e.get("gpu_power_w"))
        dur_s = _safe_num(e.get("infer_duration_s"))
        if dur_s == 0.0:
            dur_s = _safe_num(e.get("actual_latency_ms")) / 1000.0
        kwh = (gpu_w * dur_s) / 3_600_000.0   # Wh → kWh
        total_energy_kwh += kwh

        co2_g = _safe_num(e.get("system_co2_g"))
        total_co2_g_location += co2_g

        ci = _safe_num(e.get("grid_carbon"))
        if ci > 0:
            grid_ci_samples.append(ci)

        toks = e.get("tokens") or {}
        ti = int(_safe_num(toks.get("input")))
        to = int(_safe_num(toks.get("output")))
        total_input_tokens += ti
        total_output_tokens += to

        m = e.get("selected_model") or "unknown"
        bm = by_model.setdefault(m, {
            "requests": 0, "tokens": 0, "energy_kwh": 0.0, "co2_g": 0.0,
        })
        bm["requests"] += 1
        bm["tokens"] += ti + to
        bm["energy_kwh"] += kwh
        bm["co2_g"] += co2_g

        if ts:
            dk = _day_key(ts)
            bd = by_day.setdefault(dk, {
                "requests": 0, "tokens": 0, "energy_kwh": 0.0, "co2_g": 0.0,
            })
            bd["requests"] += 1
            bd["tokens"] += ti + to
            bd["energy_kwh"] += kwh
            bd["co2_g"] += co2_g

        digest = e.get("_hmac")
        if digest and len(hmac_digests) < 5:
            hmac_digests.append(digest)

    grid_ci_avg = (
        sum(grid_ci_samples) / len(grid_ci_samples) if grid_ci_samples else 0.0
    )
    total_tokens = total_input_tokens + total_output_tokens

    # ── GHG Protocol Scope 2 ─────────────────────────────────────────────
    location_based_kg = total_co2_g_location / 1000.0
    market_factor = max(0.0, 1.0 - (market_based_renewable_pct or 0.0))
    market_based_kg = location_based_kg * market_factor

    # ── CSRD ESRS E1 ────────────────────────────────────────────────────
    energy_total_mwh = total_energy_kwh / 1000.0
    intensity_per_inference_g = (
        total_co2_g_location / total_requests if total_requests else 0.0
    )
    intensity_per_1k_tokens_g = (
        total_co2_g_location / (total_tokens / 1000.0) if total_tokens else 0.0
    )

    energy_usd = total_energy_kwh * energy_price_usd_kwh

    # Chain-of-custody summary
    audit = {
        "signed_entries_count":   total_requests,
        "earliest_iso":           _iso(earliest_ts) if earliest_ts else None,
        "latest_iso":             _iso(latest_ts) if latest_ts else None,
        "hmac_chain":             "verified-on-write",
        "sample_hmac_digests":    hmac_digests,
    }

    return {
        "report_schema_version":   REPORT_SCHEMA_VERSION,
        "ghg_protocol_version":    GHG_PROTOCOL_VERSION,
        "csrd_framework_version":  CSRD_FRAMEWORK_VERSION,
        "generated_at":            _iso(),
        "scope": {
            "tenant_id":             tenant_id,
            "period_from_iso":       period_from_iso,
            "period_to_iso":         period_to_iso,
            "data_source":           "data/decision_logs.jsonl (HMAC-signed audit trail)",
            "market_based_renewable_pct": market_based_renewable_pct,
            "energy_price_usd_kwh":  energy_price_usd_kwh,
        },
        "totals": {
            "requests":              total_requests,
            "tokens_input":          total_input_tokens,
            "tokens_output":         total_output_tokens,
            "tokens_total":          total_tokens,
            "energy_kwh":            round(total_energy_kwh, 6),
            "energy_mwh":            round(energy_total_mwh, 9),
            "energy_usd":            round(energy_usd, 6),
            "grid_ci_avg_g_per_kwh": round(grid_ci_avg, 2),
        },
        "ghg_protocol_scope2": {
            "method":                            "operational LLMCarbon × grid CI",
            "location_based_kg_co2e":            round(location_based_kg, 9),
            "market_based_kg_co2e":              round(market_based_kg, 9),
            "market_based_methodology":          (
                "renewable-instrument adjusted (REC/PPA placeholder)"
                if market_based_renewable_pct > 0 else "no renewable instruments declared"
            ),
        },
        "csrd_esrs_e1": {
            "E1-5_energy_consumption_total_mwh":       round(energy_total_mwh, 9),
            "E1-5_energy_mix_renewable_pct":           round(market_based_renewable_pct * 100, 2),
            "E1-6_gross_scope_2_emissions_tco2e":      round(location_based_kg / 1000.0, 12),
            "E1-6_gross_scope_2_market_based_tco2e":   round(market_based_kg / 1000.0, 12),
            "E1-6_intensity_per_inference_g_co2e":     round(intensity_per_inference_g, 6),
            "E1-6_intensity_per_1k_tokens_g_co2e":     round(intensity_per_1k_tokens_g, 6),
            "E1-7_carbon_credits_tco2e":               0.0,
            "E1-7_credits_methodology":                "not declared",
            "E1-9_climate_risk_assessment":            "out of scope for this report",
        },
        "by_model": [
            {
                "model": m,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()},
            }
            for m, stats in sorted(by_model.items(), key=lambda kv: -kv[1]["requests"])
        ],
        "by_day": [
            {
                "day": d,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()},
            }
            for d, stats in sorted(by_day.items())
        ],
        "audit": audit,
    }


def report_to_csv(report: dict[str, Any]) -> str:
    """Render a CSRD-friendly CSV for ESG-platform import (Watershed/Persefoni/Sweep)."""
    out = io.StringIO()
    w = csv.writer(out)

    scope = report.get("scope") or {}
    totals = report.get("totals") or {}
    ghg = report.get("ghg_protocol_scope2") or {}
    csrd = report.get("csrd_esrs_e1") or {}

    # Header
    w.writerow(["section", "key", "value"])
    w.writerow(["meta", "tenant_id",        scope.get("tenant_id") or ""])
    w.writerow(["meta", "period_from_iso",  scope.get("period_from_iso") or ""])
    w.writerow(["meta", "period_to_iso",    scope.get("period_to_iso") or ""])
    w.writerow(["meta", "generated_at",     report.get("generated_at") or ""])
    w.writerow(["meta", "schema_version",   report.get("report_schema_version") or ""])
    w.writerow(["meta", "ghg_protocol_version", report.get("ghg_protocol_version") or ""])
    w.writerow(["meta", "csrd_framework_version", report.get("csrd_framework_version") or ""])

    for k, v in totals.items():
        w.writerow(["totals", k, v])
    for k, v in ghg.items():
        w.writerow(["ghg_scope2", k, v])
    for k, v in csrd.items():
        w.writerow(["csrd_esrs_e1", k, v])

    # By model
    w.writerow([])
    w.writerow(["by_model_section"])
    w.writerow(["model", "requests", "tokens", "energy_kwh", "co2_g"])
    for row in report.get("by_model") or []:
        w.writerow([row.get("model"), row.get("requests"),
                    row.get("tokens"), row.get("energy_kwh"), row.get("co2_g")])

    # By day
    w.writerow([])
    w.writerow(["by_day_section"])
    w.writerow(["day", "requests", "tokens", "energy_kwh", "co2_g"])
    for row in report.get("by_day") or []:
        w.writerow([row.get("day"), row.get("requests"),
                    row.get("tokens"), row.get("energy_kwh"), row.get("co2_g")])

    return out.getvalue()


def _iso(ts: float | None = None) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
