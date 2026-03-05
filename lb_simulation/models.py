"""Core data models used by the simulator."""

from dataclasses import dataclass


@dataclass
class Request:
    """A request flowing through generator, load balancer, and worker pool."""

    rid: int
    t_arrival: float
    class_id: int
    job_size: int
    model: str
    log_type: str
