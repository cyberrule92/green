#!/usr/bin/env python3
"""
Deferred Execution Queue — Adaptive Green AI
Implements EcoServe carbon-aware batching (Section 3.5.2 of the paper).

When grid carbon intensity is above threshold and a request has sufficient
deferral tolerance, the request is held in a priority queue and dispatched
when a low-carbon window arrives (or the deferral budget expires).

Low-carbon window identification (Section 3.5.2):
    t* = argmin_{t ∈ [0, H]} forecast[t]   subject to t ≤ deferral_budget
    If forecast[t*] < threshold: defer to t*; else: dispatch immediately.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Carbon threshold above which deferral is considered (gCO2/kWh)
HIGH_CARBON_THRESHOLD = float(450)
# Percentile of recent history used as the "low-carbon" target
LOW_CARBON_PERCENTILE = 0.25
# Background dispatch interval (seconds)
DISPATCH_INTERVAL_S = 10
# Maximum deferred requests kept in queue (backpressure)
MAX_QUEUE_SIZE = 500


@dataclass(order=True)
class _DeferredItem:
    """Priority queue entry; lower priority_ts = dispatched first."""
    priority_ts: float
    enqueued_at: float = field(compare=False)
    deadline_ts: float = field(compare=False)
    request_id: str = field(compare=False)
    request_payload: dict[str, Any] = field(compare=False)
    dispatch_fn: Callable[[dict[str, Any]], Any] = field(compare=False)
    callback: Callable[[Any], None] | None = field(compare=False, default=None)


class DeferredQueue:
    """
    Carbon-aware deferred execution queue.

    Usage:
        queue = DeferredQueue(forecast_provider=my_carbon_fetcher)
        queue.start()
        queue.enqueue(request_id, payload, dispatch_fn, deferral_ms=1800000)
        queue.stop()
    """

    def __init__(
        self,
        forecast_provider: Callable[[], dict[str, Any]] | None = None,
        high_carbon_threshold: float = HIGH_CARBON_THRESHOLD,
        dispatch_interval_s: float = DISPATCH_INTERVAL_S,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        self._heap: list[_DeferredItem] = []
        self._lock = threading.Lock()
        self._forecast_provider = forecast_provider
        self._high_carbon_threshold = high_carbon_threshold
        self._dispatch_interval_s = dispatch_interval_s
        self._max_queue_size = max_queue_size
        self._running = False
        self._thread: threading.Thread | None = None
        self._dispatched_count = 0
        self._expired_count = 0
        self._enqueued_count = 0
        self._current_carbon: float = high_carbon_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background dispatch thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="deferred-queue")
        self._thread.start()
        logger.info("DeferredQueue background dispatcher started")

    def stop(self) -> None:
        """Stop background dispatch thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("DeferredQueue stopped (dispatched=%d, expired=%d)", self._dispatched_count, self._expired_count)

    def enqueue(
        self,
        request_id: str,
        payload: dict[str, Any],
        dispatch_fn: Callable[[dict[str, Any]], Any],
        deferral_ms: int = 900_000,
        callback: Callable[[Any], None] | None = None,
    ) -> bool:
        """
        Enqueue a request for deferred dispatch.
        Returns False if the queue is full (backpressure — caller should dispatch immediately).
        """
        with self._lock:
            if len(self._heap) >= self._max_queue_size:
                logger.warning("DeferredQueue full; request %s will be dispatched immediately", request_id)
                return False

            now = time.time()
            deadline_ts = now + (deferral_ms / 1000.0)

            # Find best dispatch time within deferral budget
            priority_ts = self._find_best_dispatch_time(now, deadline_ts)

            item = _DeferredItem(
                priority_ts=priority_ts,
                enqueued_at=now,
                deadline_ts=deadline_ts,
                request_id=request_id,
                request_payload=payload,
                dispatch_fn=dispatch_fn,
                callback=callback,
            )
            heapq.heappush(self._heap, item)
            self._enqueued_count += 1
            logger.info(
                "Deferred request %s; target_dispatch=%s deadline=%s queue_size=%d",
                request_id,
                _ts_to_iso(priority_ts),
                _ts_to_iso(deadline_ts),
                len(self._heap),
            )
            return True

    def should_defer(
        self,
        grid_carbon: float,
        deferral_budget_ms: int,
        supports_batching: bool,
    ) -> bool:
        """
        Quick check: should this request be deferred?
        Mirrors the paper's deferral decision workflow (Section 3.5.2).
        """
        if deferral_budget_ms <= 0:
            return False
        if not supports_batching:
            return False
        return grid_carbon >= self._high_carbon_threshold

    def update_carbon(self, carbon_g_per_kwh: float) -> None:
        """
        Called by the monitoring layer when grid carbon signal updates.
        Triggers a re-evaluation pass to dispatch any newly eligible requests.
        """
        self._current_carbon = carbon_g_per_kwh
        if carbon_g_per_kwh < self._high_carbon_threshold:
            with self._lock:
                self._dispatch_eligible(force_low_carbon=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            pending = [
                {
                    "request_id": item.request_id,
                    "enqueued_at": _ts_to_iso(item.enqueued_at),
                    "target_dispatch": _ts_to_iso(item.priority_ts),
                    "deadline": _ts_to_iso(item.deadline_ts),
                    "seconds_until_deadline": round(item.deadline_ts - now, 1),
                }
                for item in sorted(self._heap)
            ]
        return {
            "queue_size": len(pending),
            "pending_requests": pending,
            "dispatched_total": self._dispatched_count,
            "expired_total": self._expired_count,
            "enqueued_total": self._enqueued_count,
            "current_carbon_g_per_kwh": self._current_carbon,
            "high_carbon_threshold": self._high_carbon_threshold,
            "dispatch_interval_s": self._dispatch_interval_s,
        }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                with self._lock:
                    self._dispatch_eligible()
            except Exception as exc:
                logger.error("DeferredQueue dispatch error: %s", exc, exc_info=True)
            time.sleep(self._dispatch_interval_s)

    def _dispatch_eligible(self, force_low_carbon: bool = False) -> None:
        """Called under self._lock. Dispatches all items whose time has come."""
        now = time.time()
        dispatched: list[_DeferredItem] = []
        expired: list[_DeferredItem] = []
        remaining: list[_DeferredItem] = []

        while self._heap:
            item = self._heap[0]
            # Expired — dispatch regardless
            if now >= item.deadline_ts:
                heapq.heappop(self._heap)
                expired.append(item)
            # Target time reached or low-carbon window arrived
            elif now >= item.priority_ts or force_low_carbon:
                heapq.heappop(self._heap)
                dispatched.append(item)
            else:
                break  # heap ordered; rest are not due yet

        for item in dispatched + expired:
            label = "low-carbon-window" if item in dispatched else "deadline-expired"
            logger.info("Dispatching deferred request %s (reason=%s)", item.request_id, label)
            try:
                result = item.dispatch_fn(item.request_payload)
                if item.callback:
                    item.callback(result)
                self._dispatched_count += 1
            except Exception as exc:
                logger.error("Deferred dispatch failed for %s: %s", item.request_id, exc)

        self._expired_count += len(expired)

    def _find_best_dispatch_time(self, now: float, deadline_ts: float) -> float:
        """
        Identify the best (lowest carbon) dispatch time within the deferral budget.
        Uses the grid forecast array from forecast_provider if available.
        Falls back to scheduling at deadline (conservative).
        """
        if self._forecast_provider is None:
            return now  # no forecast — dispatch at target time = now (best effort)

        try:
            grid_signal = self._forecast_provider()
            forecast: list[dict[str, Any]] = grid_signal.get("forecast", [])
            if not forecast:
                return now

            best_ts = now
            best_ci = float("inf")
            for point in forecast:
                ts = float(point.get("timestamp", now))
                ci = float(point.get("carbon_intensity", 9999))
                if ts > deadline_ts:
                    break
                if ci < best_ci:
                    best_ci = ci
                    best_ts = ts

            if best_ci < self._high_carbon_threshold:
                return best_ts
            return now  # no low-carbon window found; dispatch now
        except Exception as exc:
            logger.warning("Forecast lookup failed in deferred queue: %s", exc)
            return now


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# Module-level singleton
_queue_instance: DeferredQueue | None = None
_queue_lock = threading.Lock()


def get_deferred_queue(forecast_provider: Callable | None = None) -> DeferredQueue:
    global _queue_instance
    if _queue_instance is None:
        with _queue_lock:
            if _queue_instance is None:
                _queue_instance = DeferredQueue(forecast_provider=forecast_provider)
                _queue_instance.start()
    return _queue_instance
