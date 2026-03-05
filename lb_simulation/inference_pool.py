"""Worker pool and service-time model."""

import random
from typing import Callable, Dict, List, Optional

import simpy

from .load_balancer import LoadBalancer
from .metrics import MetricsCollector
from .models import Request
from .worker_models import ServiceTimeContext
from .workers import WorkerSpec


class InferencePool:
    """FCFS single-server worker pool."""

    def __init__(
        self,
        env: simpy.Environment,
        worker_specs: List[WorkerSpec],
        metrics: MetricsCollector,
        on_complete: Optional[Callable[[Request, int, float, bool], None]] = None,
        on_request_done: Optional[Callable[[Dict[str, object]], None]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env = env
        if not worker_specs:
            raise ValueError("InferencePool requires at least one worker.")
        self.worker_specs = worker_specs
        self.metrics = metrics
        self.on_complete = on_complete
        self.on_request_done = on_request_done
        self.rng = rng or random.Random()

        self.resources: List[simpy.Resource] = [
            simpy.Resource(env, capacity=1) for _ in worker_specs
        ]
        self.num_workers = len(worker_specs)
        self.global_inflight = 0

    def _service_time(self, worker_id: int, job_size: int, n_local: int, n_global: int) -> float:
        worker_spec = self.worker_specs[worker_id]
        context = ServiceTimeContext(job_size=job_size, n_local=n_local, n_global=n_global)
        return worker_spec.service_model_impl.sample_service_time(context, self.rng)

    def dispatch(
        self,
        request: Request,
        worker_id: int,
        lb: LoadBalancer,
        latency_tracked: bool = False,
    ) -> None:
        self.global_inflight += 1
        self.env.process(self._serve(request, worker_id, lb, latency_tracked))

    def _serve(
        self,
        request: Request,
        worker_id: int,
        lb: LoadBalancer,
        latency_tracked: bool,
    ):
        resource = self.resources[worker_id]
        worker_spec = self.worker_specs[worker_id]

        # Sample queue pressure at dispatch time.
        queue_len_on_dispatch = len(resource.queue) + resource.count
        self.metrics.record_dispatch(queue_len_on_dispatch, self.global_inflight)

        with resource.request() as slot:
            yield slot

            # At service start, resource.count is 1 for this request.
            n_local = len(resource.queue)
            n_global = self.global_inflight
            t_start = self.env.now
            service_time = self._service_time(worker_id, request.job_size, n_local, n_global)

            yield self.env.timeout(service_time)

            t_done = self.env.now
            latency = t_done - request.t_arrival
            lb.on_complete(worker_id)
            self.global_inflight = max(0, self.global_inflight - 1)
            self.metrics.record_completion(
                worker_id=worker_id,
                class_id=request.class_id,
                latency=latency,
                service_time=service_time,
            )
            if self.on_complete:
                self.on_complete(request, worker_id, latency, latency_tracked)

            if self.on_request_done:
                self.on_request_done(
                    {
                        "rid": request.rid,
                        "class_id": request.class_id,
                        "worker_id": worker_id,
                        "worker_class_id": worker_spec.worker_class_id,
                        "worker_service_model": worker_spec.service_model,
                        "job_size": request.job_size,
                        "model": request.model,
                        "log_type": request.log_type,
                        "t_arrival": request.t_arrival,
                        "t_start": t_start,
                        "t_done": t_done,
                        "queue_len_on_dispatch": queue_len_on_dispatch,
                        "n_local_at_start": n_local,
                        "n_global_at_start": n_global,
                        "latency_tracked": latency_tracked,
                        "service_time": service_time,
                        "latency": latency,
                    }
                )
