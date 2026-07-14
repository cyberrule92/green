#!/usr/bin/env python3
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_SCRIPT = os.getenv("SYSTEM_SCRIPT", str(BASE_DIR / "system_metrics.sh"))
CACHE_DURATION = int(os.getenv("CACHE_DURATION", "60"))
logger = logging.getLogger(__name__)

DEFAULT_METRICS = {
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

_metrics_cache = dict(DEFAULT_METRICS)
_metrics_lock = threading.Lock()
_last_ts = 0.0
_last_error = ""


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _normalize_metrics(raw_payload: dict | None) -> dict:
    payload = dict(DEFAULT_METRICS)
    if isinstance(raw_payload, dict):
        payload.update(raw_payload)
    payload["timestamp"] = int(time.time())
    return payload


def fetch_system_metrics(force_refresh: bool = False) -> dict:
    global _metrics_cache, _last_ts, _last_error

    now = time.time()
    with _metrics_lock:
        if not force_refresh and now - _last_ts < CACHE_DURATION:
            return dict(_metrics_cache)

        if not os.path.exists(SYSTEM_SCRIPT):
            _last_error = f"System metrics script not found at {SYSTEM_SCRIPT}"
            logger.warning(_last_error)
            _last_ts = now
            _metrics_cache = _normalize_metrics(_metrics_cache)
            return dict(_metrics_cache)

        try:
            result = subprocess.run(
                ["bash", SYSTEM_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=8,
            )
            payload = json.loads(result.stdout)
            _metrics_cache = _normalize_metrics(payload)
            _last_error = ""
        except Exception as exc:
            _last_error = str(exc)
            logger.warning("System metrics fetch failed: %s", exc)
            _metrics_cache = _normalize_metrics(_metrics_cache)

        _last_ts = now
        return dict(_metrics_cache)


app = FastAPI(title="Green Metrics Service", version="1.1.0")


@app.get("/metrics")
def get_metrics():
    return fetch_system_metrics()


@app.get("/health")
def health():
    script_exists = os.path.exists(SYSTEM_SCRIPT)
    fetch_system_metrics()
    return {
        "status": "ok",
        "script_exists": script_exists,
        "system_script": SYSTEM_SCRIPT,
        "cache_duration_seconds": CACHE_DURATION,
        "last_refresh_epoch": int(_last_ts),
        "last_error": _last_error,
    }


if __name__ == "__main__":
    _configure_logging()
    uvicorn.run(app, host="0.0.0.0", port=9000)
