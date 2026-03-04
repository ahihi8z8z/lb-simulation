"""Metrics collection and summary reporting."""

import statistics
from typing import Dict, List

from .utils import percentile


class MetricsCollector:
    """Collect latency, throughput, utilization, and queue/load samples."""

    def __init__(self, num_workers: int) -> None:
        self.latencies: List[float] = []
        self.latencies_by_class: Dict[int, List[float]] = {}
        self.worker_busy_time: List[float] = [0.0 for _ in range(num_workers)]
        self.queue_samples: List[int] = []
        self.global_inflight_samples: List[int] = []
        self.dispatch_count = 0
        self.completion_count = 0

    def record_dispatch(self, queue_len: int, global_inflight: int) -> None:
        self.dispatch_count += 1
        self.queue_samples.append(queue_len)
        self.global_inflight_samples.append(global_inflight)

    def record_completion(
        self,
        worker_id: int,
        class_id: int,
        latency: float,
        service_time: float,
    ) -> None:
        self.completion_count += 1
        self.latencies.append(latency)
        self.worker_busy_time[worker_id] += service_time
        self.latencies_by_class.setdefault(class_id, []).append(latency)

    def summarize(self, sim_time: float, active_time: float) -> Dict[str, object]:
        """Return aggregate metrics for the full simulation run."""

        mean_latency = statistics.fmean(self.latencies) if self.latencies else 0.0
        median_latency = statistics.median(self.latencies) if self.latencies else 0.0
        p95 = percentile(self.latencies, 95)
        p99 = percentile(self.latencies, 99)

        utilization = [
            (busy / sim_time) if sim_time > 0 else 0.0 for busy in self.worker_busy_time
        ]
        by_class = {
            class_id: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else 0.0,
                "p95": percentile(values, 95),
            }
            for class_id, values in sorted(self.latencies_by_class.items())
        }

        return {
            "dispatched": self.dispatch_count,
            "completed": self.completion_count,
            "throughput": (self.completion_count / active_time) if active_time > 0 else 0.0,
            "mean_latency": mean_latency,
            "median_latency": median_latency,
            "p95_latency": p95,
            "p99_latency": p99,
            "avg_queue_len": statistics.fmean(self.queue_samples) if self.queue_samples else 0.0,
            "avg_global_inflight": (
                statistics.fmean(self.global_inflight_samples)
                if self.global_inflight_samples
                else 0.0
            ),
            "avg_utilization": statistics.fmean(utilization) if utilization else 0.0,
            "utilization_by_worker": utilization,
            "latency_by_class": by_class,
        }
