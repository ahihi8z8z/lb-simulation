"""Core data models used by the simulator."""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """A request flowing through generator, load balancer, and worker pool."""

    rid: int
    t_arrival: float
    class_id: int
    job_size: int
    model: str
    log_type: str

    def __post_init__(self) -> None:
        if self.rid == 0:
            logger.info(
                "First request observed class=%d model=%s",
                self.class_id,
                self.model,
            )
        logger.debug(
            "Request created rid=%d class=%d job_size=%d",
            self.rid,
            self.class_id,
            self.job_size,
        )
