"""Worker pool and service-time model."""

import random
from typing import Callable, Dict, List, Optional

import simpy

from .load_balancer import LoadBalancer
from .metrics import MetricsCollector
from .models import Request, ServiceTimeParams


class InferencePool:
    """FCFS single-server worker pool."""

    def __init__(
        self,
        env: simpy.Environment,
        num_workers: int,
        st_params: ServiceTimeParams,
        metrics: MetricsCollector,
        on_request_done: Optional[Callable[[Dict[str, object]], None]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env = env
        self.st_params = st_params
        self.metrics = metrics
        self.on_request_done = on_request_done
        self.rng = rng or random.Random()

        self.resources: List[simpy.Resource] = [
            simpy.Resource(env, capacity=1) for _ in range(num_workers)
        ]
        self.num_workers = num_workers
        self.global_inflight = 0

    def _service_time(self, hidden_size: int, n_local: int, n_global: int) -> float:
        # S_base(z) = a + b * z
        base = self.st_params.a + self.st_params.b * hidden_size
        # h(n) = 1 + c * n
        local_factor = 1.0 + self.st_params.c * n_local
        # g(N) = 1 + d * max(0, N - N0)
        global_factor = 1.0 + self.st_params.d * max(0, n_global - self.st_params.n0)
        # epsilon ~ LogNormal(0, sigma)
        noise = self.rng.lognormvariate(0.0, self.st_params.sigma)
        service = base * local_factor * global_factor * noise
        return max(self.st_params.min_s, service)

    def dispatch(self, request: Request, worker_id: int, lb: LoadBalancer) -> None:
        self.global_inflight += 1
        self.env.process(self._serve(request, worker_id, lb))

    def _serve(self, request: Request, worker_id: int, lb: LoadBalancer):
        resource = self.resources[worker_id]

        # Sample queue pressure at dispatch time.
        queue_len_on_dispatch = len(resource.queue) + resource.count
        self.metrics.record_dispatch(queue_len_on_dispatch, self.global_inflight)

        with resource.request() as slot:
            yield slot

            # At service start, resource.count is 1 for this request.
            n_local = len(resource.queue)
            n_global = self.global_inflight
            t_start = self.env.now
            service_time = self._service_time(request.hidden_size, n_local, n_global)

            yield self.env.timeout(service_time)

            t_done = self.env.now
            latency = t_done - request.t_arrival
            lb.on_complete(worker_id, latency)
            self.global_inflight = max(0, self.global_inflight - 1)
            self.metrics.record_completion(
                worker_id=worker_id,
                class_id=request.class_id,
                latency=latency,
                service_time=service_time,
            )

            if self.on_request_done:
                self.on_request_done(
                    {
                        "rid": request.rid,
                        "class_id": request.class_id,
                        "worker_id": worker_id,
                        "hidden_size": request.hidden_size,
                        "t_arrival": request.t_arrival,
                        "t_start": t_start,
                        "t_done": t_done,
                        "queue_len_on_dispatch": queue_len_on_dispatch,
                        "n_local_at_start": n_local,
                        "n_global_at_start": n_global,
                        "service_time": service_time,
                        "latency": latency,
                    }
                )
