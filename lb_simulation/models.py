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


@dataclass
class ServiceTimeParams:
    """Parameters for service-time model: S = S_base * h(local) * g(global) * noise."""

    a: float = 0.03
    b: float = 0.002
    c: float = 0.12
    d: float = 0.015
    n0: int = 32
    sigma: float = 0.20
    min_s: float = 0.001
