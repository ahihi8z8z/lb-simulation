"""Worker pool and service-time model runtime."""

import json
import logging
import random
from typing import Callable, Dict, List, Optional

import simpy

from .load_balancer import LoadBalancer
from .metrics import MetricsCollector
from .models import Request
from .worker_models import LimitedProcessorSharingModel, ServiceTimeContext
from .workers import WorkerSpec

logger = logging.getLogger(__name__)


class InferencePool:
    """
    Worker pool based on SimPy Resource workers.

    - Most worker models use `capacity=1` with configurable queue policy:
      FCFS (`Resource`) or SJF (`PriorityResource`).
    - `limited_processor_sharing` is approximated as a `k`-server worker by
      setting `capacity=max_concurrency` and serving each request with isolated
      service time `job_size / (processing_rate / max_concurrency)`.
    """

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

        self.resources: List[simpy.Resource] = []
        for worker_id, worker_spec in enumerate(worker_specs):
            capacity = 1
            service_model = worker_spec.service_model_impl
            if isinstance(service_model, LimitedProcessorSharingModel):
                capacity = max(1, int(service_model.max_concurrency))
                logger.info(
                    (
                        "Initialized k-server worker_id=%d total_rate=%.6f "
                        "per_slot_rate=%.6f max_concurrency=%d"
                    ),
                    worker_id,
                    service_model.processing_rate,
                    service_model.slot_processing_rate,
                    capacity,
                )
            if worker_spec.queue_policy == "sjf":
                resource = simpy.PriorityResource(env, capacity=capacity)
            else:
                resource = simpy.Resource(env, capacity=capacity)
            self.resources.append(resource)
            logger.debug(
                (
                    "Worker queue configured worker_id=%d queue_policy=%s "
                    "capacity=%d queue_timeout_seconds=%s"
                ),
                worker_id,
                worker_spec.queue_policy,
                capacity,
                worker_spec.queue_timeout_seconds,
            )

        self.num_workers = len(worker_specs)
        self.global_inflight = 0
        logger.info("InferencePool initialized with %d workers", self.num_workers)

    def _service_time(self, worker_id: int, job_size: int, n_local: int, n_global: int) -> float:
        worker_spec = self.worker_specs[worker_id]
        context = ServiceTimeContext(job_size=job_size, n_local=n_local, n_global=n_global)
        return worker_spec.service_model_impl.sample_service_time(context, self.rng)

    def _worker_load_snapshot(self, worker_id: int) -> int:
        """Return local worker load snapshot used by detail metrics."""

        resource = self.resources[worker_id]
        return len(resource.queue) + resource.count

    def _request_worker_slot(self, worker_id: int, request: Request):
        """Create one resource request with queue-policy-aware priority."""

        resource = self.resources[worker_id]
        worker_spec = self.worker_specs[worker_id]
        if worker_spec.queue_policy == "sjf":
            # PriorityResource serves lower values first.
            return resource.request(priority=(int(request.job_size), int(request.rid)))
        return resource.request()

    def _finalize_completion(
        self,
        request: Request,
        worker_id: int,
        worker_spec: WorkerSpec,
        lb: LoadBalancer,
        latency_tracked: bool,
        lb_completion_worker_ids: List[int],
        lb_selected_worker_id: Optional[int],
        routed_via_latency_tracker: bool,
        detail_state: Optional[Dict[str, object]],
        queue_len_on_dispatch: int,
        n_local_at_start: int,
        n_global_at_start: int,
        t_start: float,
        service_time: float,
        busy_time: Optional[float] = None,
    ) -> None:
        """Finalize one completed request and emit metrics/callbacks."""

        t_done = self.env.now
        latency = t_done - request.t_arrival
        queueing_latency = max(0.0, latency - service_time)
        completion_worker_ids = lb_completion_worker_ids or [worker_id]
        for completion_worker_id in completion_worker_ids:
            lb.on_complete(completion_worker_id)
        self.global_inflight = max(0, self.global_inflight - 1)
        self.metrics.record_completion(
            worker_id=worker_id,
            class_id=request.class_id,
            job_size=request.job_size,
            latency=latency,
            service_time=service_time,
            busy_time=busy_time,
        )
        if self.on_complete:
            self.on_complete(request, worker_id, latency, latency_tracked)
        logger.debug(
            "Completed request rid=%d worker=%d latency=%.4f tracked=%s",
            request.rid,
            worker_id,
            latency,
            latency_tracked,
        )

        if self.on_request_done:
            lb_state_json = None
            lb_control_state_json = None
            queue_snapshot = None
            if detail_state is not None:
                if "lb_state" in detail_state:
                    lb_state_json = json.dumps(detail_state["lb_state"], separators=(",", ":"))
                if "lb_control_state" in detail_state:
                    lb_control_state_json = json.dumps(
                        detail_state["lb_control_state"], separators=(",", ":")
                    )
                if "queue_snapshot" in detail_state:
                    queue_snapshot = json.dumps(detail_state["queue_snapshot"])

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
                    "n_local_at_start": n_local_at_start,
                    "n_global_at_start": n_global_at_start,
                    "lb_selected_worker_id": (
                        lb_selected_worker_id
                        if lb_selected_worker_id is not None
                        else worker_id
                    ),
                    "routed_via_latency_tracker": routed_via_latency_tracker,
                    "latency_tracked": latency_tracked,
                    "service_time": service_time,
                    "queueing_latency": queueing_latency,
                    "latency": latency,
                    "lb_state": lb_state_json,
                    "lb_control_state": lb_control_state_json,
                    "queue_snapshot": queue_snapshot,
                }
            )

    def _finalize_drop(
        self,
        request: Request,
        worker_id: int,
        lb: LoadBalancer,
        lb_completion_worker_ids: List[int],
        waited_in_queue: float,
    ) -> None:
        """Finalize one request dropped after exceeding queue wait timeout."""

        completion_worker_ids = lb_completion_worker_ids or [worker_id]
        for completion_worker_id in completion_worker_ids:
            lb.on_complete(completion_worker_id)
        self.global_inflight = max(0, self.global_inflight - 1)
        self.metrics.record_drop(worker_id=worker_id, class_id=request.class_id)
        logger.info(
            (
                "Dropped request rid=%d class_id=%d worker=%d "
                "queue_wait=%.6f reason=queue_timeout"
            ),
            request.rid,
            request.class_id,
            worker_id,
            waited_in_queue,
        )

    def dispatch(
        self,
        request: Request,
        worker_id: int,
        lb: LoadBalancer,
        latency_tracked: bool = False,
        lb_completion_worker_ids: Optional[List[int]] = None,
        lb_selected_worker_id: Optional[int] = None,
        routed_via_latency_tracker: bool = False,
        detail_state: Optional[Dict[str, object]] = None,
    ) -> None:
        if detail_state is not None:
            detail_state["queue_snapshot"] = [
                self._worker_load_snapshot(idx) for idx in range(self.num_workers)
            ]

        self.global_inflight += 1
        logger.debug(
            "Dispatch request rid=%d worker=%d tracked=%s global_inflight=%d",
            request.rid,
            worker_id,
            latency_tracked,
            self.global_inflight,
        )

        self.env.process(
            self._serve_with_resource(
                request=request,
                worker_id=worker_id,
                lb=lb,
                latency_tracked=latency_tracked,
                lb_completion_worker_ids=lb_completion_worker_ids,
                lb_selected_worker_id=lb_selected_worker_id,
                routed_via_latency_tracker=routed_via_latency_tracker,
                detail_state=detail_state,
            )
        )

    def _serve_with_resource(
        self,
        request: Request,
        worker_id: int,
        lb: LoadBalancer,
        latency_tracked: bool,
        lb_completion_worker_ids: Optional[List[int]],
        lb_selected_worker_id: Optional[int],
        routed_via_latency_tracker: bool,
        detail_state: Optional[Dict[str, object]],
    ):
        resource = self.resources[worker_id]
        worker_spec = self.worker_specs[worker_id]

        queue_len_on_dispatch = len(resource.queue) + resource.count
        self.metrics.record_dispatch(
            worker_id=worker_id,
            class_id=request.class_id,
            queue_len=queue_len_on_dispatch,
            global_inflight=self.global_inflight,
        )

        slot = self._request_worker_slot(worker_id, request)
        acquired = False
        try:
            queue_timeout_seconds = worker_spec.queue_timeout_seconds
            if queue_timeout_seconds is None:
                yield slot
            else:
                wait_result = yield slot | self.env.timeout(queue_timeout_seconds)
                if slot not in wait_result:
                    slot.cancel()
                    self._finalize_drop(
                        request=request,
                        worker_id=worker_id,
                        lb=lb,
                        lb_completion_worker_ids=(
                            lb_completion_worker_ids
                            if lb_completion_worker_ids is not None
                            else [worker_id]
                        ),
                        waited_in_queue=max(0.0, self.env.now - request.t_arrival),
                    )
                    return

            acquired = True

            # Exclude the current request from local pressure.
            n_local = len(resource.queue) + max(0, resource.count - 1)
            n_global = self.global_inflight
            t_start = self.env.now
            service_time = self._service_time(worker_id, request.job_size, n_local, n_global)
            yield self.env.timeout(service_time)
            busy_time = service_time / float(max(1, resource.capacity))

            self._finalize_completion(
                request=request,
                worker_id=worker_id,
                worker_spec=worker_spec,
                lb=lb,
                latency_tracked=latency_tracked,
                lb_completion_worker_ids=(
                    lb_completion_worker_ids if lb_completion_worker_ids is not None else [worker_id]
                ),
                lb_selected_worker_id=lb_selected_worker_id,
                routed_via_latency_tracker=routed_via_latency_tracker,
                detail_state=detail_state,
                queue_len_on_dispatch=queue_len_on_dispatch,
                n_local_at_start=n_local,
                n_global_at_start=n_global,
                t_start=t_start,
                service_time=service_time,
                busy_time=busy_time,
            )
        finally:
            if acquired:
                resource.release(slot)
