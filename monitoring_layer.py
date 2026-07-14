#!/usr/bin/env python3
"""
Monitoring Layer — Adaptive Green AI
Implements the Observation Layer (Section 3.2) of the paper:
- Grid Carbon Fetcher: real-time and forecast carbon intensity from Electricity Maps
  with multi-zone support (CAISO, ENTSO-E, IN-WE) (Section 3.2.2)
- 48-hour forecast array per zone (Section 3.2.2)
- System metrics sidecar: GPU/CPU utilisation and power draw (Section 3.2.1)
- Task profiler: per-request criticality, accuracy, and latency inference (Section 3.2)
"""

from __future__ import annotations

import copy
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter

# urllib3 is a transitive dependency of requests and is always present at
# runtime. Some IDEs (Pylance/Pyright) cannot resolve transitive deps, so
# we probe three locations before falling back to a minimal stub.
try:
    from urllib3.util.retry import Retry                          # standard install
except ModuleNotFoundError:
    try:
        from requests.packages.urllib3.util.retry import Retry    # requests-bundled
    except (ModuleNotFoundError, AttributeError):
        # Last-resort stub — only reached in heavily stripped environments.
        class Retry:  # type: ignore[no-redef]
            """Minimal Retry stub used when urllib3 is not importable."""
            def __init__(self, *_, **__): pass

# ── Config ──────────────────────────────────────────────────────────────────
# EMAP_TOKEN must be set in your .env file (get a free key at electricitymap.org).
# Leaving it empty causes the fetcher to return a clean "unconfigured" fallback
# rather than hitting the API with an expired/invalid token and silently degrading.
EMAP_TOKEN = os.getenv("EMAP_TOKEN", "")
EMAP_ZONE = os.getenv("EMAP_ZONE", "IN-WE")

# Additional zones for multi-region routing (Section 3.5.3)
# Format: "ZONE_ID:friendly_region_key"
_EXTRA_ZONES_ENV = os.getenv("EMAP_EXTRA_ZONES", "US-CAL-CISO:us-west,DE:eu")
EMAP_ALL_ZONES: dict[str, str] = {EMAP_ZONE: "local"}
for _entry in _EXTRA_ZONES_ENV.split(","):
    _parts = _entry.strip().split(":")
    if len(_parts) == 2:
        EMAP_ALL_ZONES[_parts[0].strip()] = _parts[1].strip()

CACHE_DURATION = int(os.getenv("EMAP_CACHE_SECONDS", "60"))
FORECAST_HORIZON_HOURS = int(os.getenv("EMAP_FORECAST_HOURS", "48"))
SYSTEM_SERVICE_URL = os.getenv("SYSTEM_SERVICE_URL", "http://127.0.0.1:9000/metrics")
GRID_CARBON_FALLBACK = float(os.getenv("GRID_CARBON_FALLBACK", "475"))

logger = logging.getLogger(__name__)


# ── Default payloads ─────────────────────────────────────────────────────────

DEFAULT_SYSTEM_METRICS: dict[str, Any] = {
    "system_gpu_utilization": 0.0,
    "system_gpu_TotalMemory": 0.0,
    "system_gpu_UsedMemory": 0.0,
    "system_gpu_MemoryFree": 0.0,
    "system_gpu_SMClockFrequency": 0.0,
    "system_gpu_MemClockFrequency": 0.0,
    "system_gpu_GraphicsClock": 0.0,
    "system_gpu_VideoClock": 0.0,
    "system_gpu_CoreTemperature": 0.0,
    "system_gpu_PowerDraw": 0.0,
    "system_gpu_PerformanceState": "unknown",
    "system_cpu_utilization": 0.0,
    "system_cpu_power": 0.0,
    "system_total_power": 0.0,
    "system_energy": 0.0,
    "system_co2_emission": 0.0,
}

DEFAULT_GRID_SIGNAL: dict[str, Any] = {
    "provider": "ElectricityMap",
    "zone": EMAP_ZONE,
    "configured": False,
    "status": "unconfigured",
    "carbon_intensity": GRID_CARBON_FALLBACK,
    "power_total_mw": None,
    "source": "fallback",
    "detail": "EMAP_TOKEN is not set. Add EMAP_TOKEN=<your_key> to .env to enable live carbon data.",
    "last_updated": None,
    "carbon_status": "fallback",
    "power_status": "unavailable",
    "carbon_source": "fallback",
    "power_source": "unavailable",
    "forecast": [],
}


# ── HTTP session ─────────────────────────────────────────────────────────────
# urllib3 <1.26 uses `method_whitelist`; >=1.26 / v2.x uses `allowed_methods`.
# We probe at import-time so the session works across all installed versions.

def _build_retry() -> Retry:
    """Build a urllib3 Retry object compatible with both v1.x and v2.x."""
    _retry_kwargs = dict(
        total=5,
        backoff_factor=0.3,
        # 429 (rate-limited) must NOT be auto-retried: the Electricity Maps
        # free tier enforces a hard 60-second window, so retrying immediately
        # burns quota without succeeding. The cache layer handles back-pressure.
        status_forcelist=[500, 502, 503, 504],
    )
    # Try the modern parameter name first (urllib3 >= 1.26)
    try:
        return Retry(
            **_retry_kwargs,
            allowed_methods={"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"},
        )
    except TypeError:
        # Fall back to the legacy parameter name (urllib3 < 1.26)
        return Retry(
            **_retry_kwargs,
            method_whitelist={"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"},
        )


def _requests_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_build_retry())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


http_sess = _requests_session()


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Per-zone cache ────────────────────────────────────────────────────────────
# Each zone has its own cached signal + forecast array
_zone_lock = threading.Lock()
_zone_cache: dict[str, dict[str, Any]] = {}   # zone -> signal dict
_zone_last_ts: dict[str, float] = {}           # zone -> epoch of last fetch

# Primary zone cache aliases (for backward-compat callers)
_grid_lock = threading.Lock()
_grid_last_ts: float = 0.0
_grid_cached: dict[str, Any] = dict(DEFAULT_GRID_SIGNAL)


# ── Grid Carbon Fetcher ───────────────────────────────────────────────────────

def _build_unconfigured_signal(zone: str) -> dict[str, Any]:
    return {
        **DEFAULT_GRID_SIGNAL,
        "zone": zone,
        "configured": False,
        "status": "unconfigured",
        "carbon_intensity": GRID_CARBON_FALLBACK,
        "source": "fallback",
        "detail": "EMAP_TOKEN is not set. Add EMAP_TOKEN=<your_key> to .env to enable live carbon data.",
        "last_updated": utc_now_iso(),
        "forecast": [],
    }


def _build_partial_signal(zone: str, detail: str) -> dict[str, Any]:
    return {
        **DEFAULT_GRID_SIGNAL,
        "zone": zone,
        "configured": bool(EMAP_TOKEN),
        "status": "degraded",
        "carbon_intensity": GRID_CARBON_FALLBACK,
        "source": "fallback",
        "detail": detail,
        "last_updated": utc_now_iso(),
        "forecast": [],
    }


def _fetch_zone_signal(zone: str) -> dict[str, Any]:
    """Fetch live carbon intensity + forecast for a single Electricity Maps zone."""
    if not EMAP_TOKEN:
        return _build_unconfigured_signal(zone)

    # Electricity Maps v3 accepts either "auth-token" (legacy) or
    # "Authorization: Bearer <token>" (newer keys, length > 40 or explicit prefix).
    if EMAP_TOKEN.startswith("Bearer ") or len(EMAP_TOKEN) > 40:
        headers = {"Authorization": f"Bearer {EMAP_TOKEN}"}
    else:
        headers = {"auth-token": EMAP_TOKEN}
    carbon_url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
    forecast_url = f"https://api.electricitymap.org/v3/carbon-intensity/forecast?zone={zone}"
    power_url = f"https://api.electricitymap.org/v3/power-breakdown/latest?zone={zone}"

    try:
        # ── Live carbon intensity ──
        carbon_resp = http_sess.get(carbon_url, headers=headers, timeout=5)
        carbon_resp.raise_for_status()
        carbon_payload = carbon_resp.json()
        carbon_intensity = safe_float(carbon_payload.get("carbonIntensity"), GRID_CARBON_FALLBACK)

        # ── 48-hour forecast array (Section 3.2.2) ──
        forecast: list[dict[str, Any]] = []
        try:
            fc_resp = http_sess.get(forecast_url, headers=headers, timeout=8)
            fc_resp.raise_for_status()
            fc_data = fc_resp.json()
            raw_fc = fc_data.get("forecast", [])
            for point in raw_fc:
                dt_str = point.get("datetime") or ""
                ci = safe_float(point.get("carbonIntensity"), GRID_CARBON_FALLBACK)
                ts = _iso_to_epoch(dt_str)
                if ts and ci > 0:
                    forecast.append({"timestamp": ts, "carbon_intensity": ci, "datetime": dt_str})
            forecast = forecast[: FORECAST_HORIZON_HOURS * 4]  # 15-min intervals max
        except Exception as exc:
            logger.debug("Forecast fetch failed for zone %s: %s", zone, exc)

        # ── Grid power breakdown ──
        power_total = None
        power_status = "unavailable"
        power_source = "unavailable"
        try:
            pwr_resp = http_sess.get(power_url, headers=headers, timeout=5)
            pwr_resp.raise_for_status()
            pwr_payload = pwr_resp.json()
            power_total = safe_float(
                pwr_payload.get("powerConsumptionTotal") or pwr_payload.get("powerProductionTotal"),
                0.0,
            )
            power_status = "live"
            power_source = "electricitymap"
        except Exception as exc:
            logger.debug("Power breakdown fetch failed for zone %s: %s", zone, exc)

        detail = "Live ElectricityMap carbon intensity"
        if forecast:
            detail += f" + {len(forecast)}-point forecast"
        if power_status == "live":
            detail += " + grid power available."
        else:
            detail += " (power breakdown unavailable)."

        return {
            "provider": "ElectricityMap",
            "zone": zone,
            "configured": True,
            "status": "live" if power_status == "live" else "degraded",
            "carbon_intensity": carbon_intensity,
            "power_total_mw": power_total,
            "source": "electricitymap",
            "detail": detail,
            "last_updated": utc_now_iso(),
            "carbon_status": "live",
            "power_status": power_status,
            "carbon_source": "electricitymap",
            "power_source": power_source,
            "forecast": forecast,
        }

    except Exception as exc:
        logger.warning("Carbon intensity fetch failed for zone %s: %s", zone, exc)
        return _build_partial_signal(zone, str(exc))


def fetch_grid_signal(zone: str | None = None) -> dict[str, Any]:
    """
    Fetch grid signal for the primary (or specified) zone.
    Uses a per-zone cache with CACHE_DURATION expiry.
    """
    target_zone = zone or EMAP_ZONE
    now = time.time()

    with _zone_lock:
        last_ts = _zone_last_ts.get(target_zone, 0.0)
        if now - last_ts < CACHE_DURATION and target_zone in _zone_cache:
            return copy.deepcopy(_zone_cache[target_zone])

    signal = _fetch_zone_signal(target_zone)

    with _zone_lock:
        _zone_cache[target_zone] = signal
        _zone_last_ts[target_zone] = now

    # Also keep primary-zone alias for backward compat
    if target_zone == EMAP_ZONE:
        global _grid_cached, _grid_last_ts
        with _grid_lock:
            _grid_cached = signal
            _grid_last_ts = now

    return copy.deepcopy(signal)


def fetch_all_zone_signals() -> dict[str, dict[str, Any]]:
    """
    Fetch live signals for all configured zones in parallel.
    Returns dict: region_key -> signal.
    Used by the Decision Engine for multi-region carbon routing (Section 3.5.3).
    """
    import concurrent.futures

    results: dict[str, dict[str, Any]] = {}
    zone_items = list(EMAP_ALL_ZONES.items())

    def _fetch(zone_tuple: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        zone_id, region_key = zone_tuple
        return region_key, fetch_grid_signal(zone_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(zone_items), 4)) as pool:
        futures = {pool.submit(_fetch, item): item for item in zone_items}
        for future in concurrent.futures.as_completed(futures, timeout=12):
            try:
                region_key, signal = future.result()
                results[region_key] = signal
            except Exception as exc:
                logger.warning("Zone fetch failed: %s", exc)

    return results


def get_zone_carbon_map(zone_signals: dict[str, dict[str, Any]] | None = None) -> dict[str, float]:
    """
    Build a {zone_id: carbon_intensity} lookup for regional routing scoring.
    If zone_signals not provided, fetches fresh data.
    """
    signals = zone_signals or fetch_all_zone_signals()
    carbon_map: dict[str, float] = {}
    for zone_id, region_key in EMAP_ALL_ZONES.items():
        signal = signals.get(region_key, {})
        carbon_map[zone_id] = safe_float(signal.get("carbon_intensity"), GRID_CARBON_FALLBACK)
    return carbon_map


def get_zone_forecast(zone: str | None = None) -> list[dict[str, Any]]:
    """Return the forecast array for a zone (empty list if unavailable)."""
    signal = fetch_grid_signal(zone or EMAP_ZONE)
    return signal.get("forecast", [])


def find_low_carbon_window(
    forecast: list[dict[str, Any]],
    deferral_budget_ms: int,
    threshold_percentile: float = 0.25,
) -> dict[str, Any] | None:
    """
    Identify the best low-carbon dispatch window within the deferral budget.
    Paper Section 3.5.2:
        t* = argmin_{t ∈ [0, budget]} forecast[t]
    Returns the best forecast point or None if no window found.
    """
    if not forecast or deferral_budget_ms <= 0:
        return None

    now = time.time()
    deadline_ts = now + (deferral_budget_ms / 1000.0)

    eligible = [p for p in forecast if p.get("timestamp", 0) <= deadline_ts]
    if not eligible:
        return None

    intensities = sorted(p["carbon_intensity"] for p in eligible)
    threshold = intensities[int(len(intensities) * threshold_percentile)]

    best = min(eligible, key=lambda p: p["carbon_intensity"])
    if best["carbon_intensity"] <= threshold:
        return best
    return None


def fetch_carbon_intensity(zone: str | None = None) -> float:
    return safe_float(fetch_grid_signal(zone).get("carbon_intensity"), GRID_CARBON_FALLBACK)


def fetch_power_breakdown(zone: str | None = None) -> float:
    return safe_float(fetch_grid_signal(zone).get("power_total_mw"), 0.0)


# ── System metrics via host sidecar (Section 3.2.1) ──────────────────────────
_system_lock = threading.Lock()
_system_last_ts: float = 0.0
_system_cached: dict[str, Any] = {}


def fetch_system_metrics() -> dict[str, Any]:
    global _system_last_ts, _system_cached

    now = time.time()
    with _system_lock:
        if now - _system_last_ts < CACHE_DURATION and _system_cached:
            return _system_cached

    try:
        response = http_sess.get(SYSTEM_SERVICE_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        merged = {**DEFAULT_SYSTEM_METRICS, **data}
        with _system_lock:
            _system_cached = merged
            _system_last_ts = now
        return merged
    except Exception as exc:
        logger.warning("System metrics fetch failed: %s", exc)
        return _system_cached or dict(DEFAULT_SYSTEM_METRICS)


# ── Task Profiler (Section 3.2) ───────────────────────────────────────────────

def infer_task_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute criticality, accuracy requirement, and latency SLA per request."""
    criticality = 0.5
    accuracy_req = 0.75
    latency_sla = 250
    latency_min = 50
    latency_max = 500

    task_profile = payload.get("task_profile", {})
    if isinstance(task_profile, dict) and task_profile:
        criticality = safe_float(task_profile.get("criticality"), criticality)
        accuracy_req = safe_float(task_profile.get("accuracy_req"), accuracy_req)
        latency_sla = safe_float(task_profile.get("latency_sla"), latency_sla)
    else:
        criticality = safe_float(payload.get("criticality"), criticality)
        accuracy_req = safe_float(payload.get("accuracy_req"), accuracy_req)
        latency_sla = safe_float(payload.get("latency_sla"), latency_sla)

        priority = str(payload.get("priority", "")).strip().lower()
        if priority == "urgent":
            criticality = max(criticality, 1.0)
            accuracy_req = max(accuracy_req, 0.9)
            latency_sla = min(latency_sla, 100)
        elif priority == "high":
            criticality = max(criticality, 0.8)
            accuracy_req = max(accuracy_req, 0.82)
            latency_sla = min(latency_sla, 160)
        elif priority == "medium":
            criticality = max(criticality, 0.5)
            accuracy_req = max(accuracy_req, 0.74)
            latency_sla = min(latency_sla, 240)
        elif priority in {"low", "casual"}:
            criticality = min(criticality, 0.25)
            accuracy_req = min(accuracy_req, 0.65)
            latency_sla = max(latency_sla, 320)

        mode = str(payload.get("mode", "")).strip().lower()
        if mode == "fast":
            latency_sla = min(latency_sla, 150)
        elif mode == "accurate":
            accuracy_req = max(accuracy_req, 0.88)

    query = str(payload.get("query", "")).lower()
    if query:
        if re.search(r"\b(urgent|immediately|asap|important|emergency|now|incident|outage)\b", query):
            criticality = max(criticality, 0.9)
            latency_sla = min(latency_sla, 120)
        if len(query.split()) > 50:
            accuracy_req = max(accuracy_req, 0.88)
            latency_sla = min(latency_sla, 220)
        if re.search(r"\b(help|assist|support|question|problem|debug|fix)\b", query):
            criticality = max(criticality, 0.78)
            accuracy_req = max(accuracy_req, 0.8)
            latency_sla = min(latency_sla, 180)

    criticality = min(max(criticality, 0.0), 1.0)
    accuracy_req = min(max(accuracy_req, 0.0), 1.0)
    latency_sla = min(max(latency_sla, latency_min), latency_max)

    return {
        "criticality": round(criticality, 3),
        "accuracy_req": round(accuracy_req, 3),
        "latency_sla": int(latency_sla),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iso_to_epoch(iso_str: str) -> float | None:
    """Parse ISO-8601 string to epoch float; returns None on failure."""
    if not iso_str:
        return None
    try:
        # Strip trailing Z and parse
        clean = iso_str.rstrip("Z").replace("+00:00", "")
        dt = datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None
