"""Metrics collection and summary reporting."""

import logging
import statistics
from typing import Dict, List, Optional

from .utils import percentile

logger = logging.getLogger(__name__)


def _population_stddev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def _max_gap(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return max(values) - min(values)


class MetricsCollector:
    """Collect latency, throughput, utilization, and queue/load samples."""

    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers
        self.latencies: List[float] = []
        self.latencies_by_class: Dict[int, List[float]] = {}
        self.latencies_by_worker: Dict[int, List[float]] = {}
        self.total_job_size = 0
        self.total_job_size_by_class: Dict[int, int] = {}
        self.worker_busy_time: List[float] = [0.0 for _ in range(num_workers)]
        self.queue_samples: List[int] = []
        self.global_inflight_samples: List[int] = []
        self.dispatch_count = 0
        self.dispatch_count_by_class: Dict[int, int] = {}
        self.dispatch_count_by_worker: List[int] = [0 for _ in range(num_workers)]
        self.completion_count = 0
        self.drop_count = 0
        self.drop_count_by_class: Dict[int, int] = {}
        self.drop_count_by_worker: List[int] = [0 for _ in range(num_workers)]
        logger.info("MetricsCollector initialized for %d workers", num_workers)

    def record_dispatch(
        self,
        worker_id: int,
        class_id: int,
        queue_len: int,
        global_inflight: int,
    ) -> None:
        self.dispatch_count += 1
        self.dispatch_count_by_worker[worker_id] += 1
        self.dispatch_count_by_class[class_id] = self.dispatch_count_by_class.get(class_id, 0) + 1
        self.queue_samples.append(queue_len)
        self.global_inflight_samples.append(global_inflight)
        logger.debug(
            "Dispatch metric recorded worker=%d class=%d queue_len=%d global_inflight=%d",
            worker_id,
            class_id,
            queue_len,
            global_inflight,
        )

    def record_drop(self, worker_id: int, class_id: int) -> None:
        self.drop_count += 1
        self.drop_count_by_worker[worker_id] += 1
        self.drop_count_by_class[class_id] = self.drop_count_by_class.get(class_id, 0) + 1
        logger.debug(
            "Drop metric recorded worker=%d class=%d dropped=%d",
            worker_id,
            class_id,
            self.drop_count,
        )

    def record_completion(
        self,
        worker_id: int,
        class_id: int,
        job_size: int,
        latency: float,
        service_time: float,
        busy_time: Optional[float] = None,
    ) -> None:
        self.completion_count += 1
        self.latencies.append(latency)
        busy_time_value = service_time if busy_time is None else busy_time
        self.worker_busy_time[worker_id] += max(0.0, float(busy_time_value))
        self.latencies_by_class.setdefault(class_id, []).append(latency)
        self.latencies_by_worker.setdefault(worker_id, []).append(latency)
        self.total_job_size += int(job_size)
        self.total_job_size_by_class[class_id] = (
            self.total_job_size_by_class.get(class_id, 0) + int(job_size)
        )
        logger.debug(
            "Completion metric recorded worker=%d class=%d job_size=%d latency=%.4f",
            worker_id,
            class_id,
            job_size,
            latency,
        )

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
                "median": statistics.median(values) if values else 0.0,
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
            }
            for class_id, values in sorted(self.latencies_by_class.items())
        }
        by_worker: Dict[int, Dict[str, object]] = {}
        for worker_id in range(len(self.worker_busy_time)):
            values = self.latencies_by_worker.get(worker_id, [])
            by_worker[worker_id] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else 0.0,
                "median": statistics.median(values) if values else 0.0,
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
            }

        worker_latency_means = [float(stats.get("mean", 0.0)) for stats in by_worker.values()]
        service_latency_means = [float(stats.get("mean", 0.0)) for stats in by_class.values()]
        utilization_float = [float(value) for value in utilization]
        worker_latency_mean_stddev = _population_stddev(worker_latency_means)
        worker_latency_mean_max_gap = _max_gap(worker_latency_means)
        service_latency_mean_stddev = _population_stddev(service_latency_means)
        service_latency_mean_max_gap = _max_gap(service_latency_means)
        worker_utilization_stddev = _population_stddev(utilization_float)
        worker_utilization_max_gap = _max_gap(utilization_float)
        drop_rate = (
            self.drop_count / self.dispatch_count if self.dispatch_count > 0 else 0.0
        )
        drop_by_class = {
            class_id: {
                "dispatched": dispatched,
                "dropped": self.drop_count_by_class.get(class_id, 0),
                "drop_rate": (
                    self.drop_count_by_class.get(class_id, 0) / dispatched
                    if dispatched > 0
                    else 0.0
                ),
            }
            for class_id, dispatched in sorted(self.dispatch_count_by_class.items())
        }
        drop_by_worker = {
            worker_id: {
                "dispatched": self.dispatch_count_by_worker[worker_id],
                "dropped": self.drop_count_by_worker[worker_id],
                "drop_rate": (
                    self.drop_count_by_worker[worker_id] / self.dispatch_count_by_worker[worker_id]
                    if self.dispatch_count_by_worker[worker_id] > 0
                    else 0.0
                ),
            }
            for worker_id in range(self.num_workers)
        }
        logger.info(
            "Summarized metrics dispatched=%d completed=%d dropped=%d mean_latency=%.4f",
            self.dispatch_count,
            self.completion_count,
            self.drop_count,
            mean_latency,
        )

        return {
            "dispatched": self.dispatch_count,
            "completed": self.completion_count,
            "dropped": self.drop_count,
            "drop_rate": drop_rate,
            "total_job_size": self.total_job_size,
            "total_job_size_by_class": dict(sorted(self.total_job_size_by_class.items())),
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
            "worker_latency_mean_stddev": worker_latency_mean_stddev,
            "worker_latency_mean_max_gap": worker_latency_mean_max_gap,
            "service_latency_mean_stddev": service_latency_mean_stddev,
            "service_latency_mean_max_gap": service_latency_mean_max_gap,
            "worker_utilization_stddev": worker_utilization_stddev,
            "worker_utilization_max_gap": worker_utilization_max_gap,
            "utilization_by_worker": utilization,
            "latency_by_class": by_class,
            "latency_by_worker": by_worker,
            "drop_by_class": drop_by_class,
            "drop_by_worker": drop_by_worker,
        }
