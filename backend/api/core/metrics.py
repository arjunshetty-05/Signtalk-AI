"""
core/metrics.py — SignTalk AI / Person C, Prompt C1

Thread-safe in-memory counters backing GET /metrics: total predictions
served, average inference latency, and active WebSocket connection count.
"""

from __future__ import annotations

import threading


class MetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._total_predictions = 0
        self._total_latency_ms = 0.0
        self._active_connections = 0

    def record_prediction(self, latency_ms: float) -> None:
        with self._lock:
            self._total_predictions += 1
            self._total_latency_ms += latency_ms

    def connection_opened(self) -> None:
        with self._lock:
            self._active_connections += 1

    def connection_closed(self) -> None:
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_predictions
                if self._total_predictions > 0
                else 0.0
            )
            return {
                "total_predictions": self._total_predictions,
                "average_latency_ms": round(avg_latency, 2),
                "active_connections": self._active_connections,
            }


metrics = MetricsTracker()
