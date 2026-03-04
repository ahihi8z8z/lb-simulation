"""Simulation assembly, CLI parsing, and summary printing."""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import simpy

from .inference_pool import InferencePool
from .load_balancer import LoadBalancer
from .metrics import MetricsCollector
from .models import Request, ServiceTimeParams
from .request_csv_logger import RequestCsvLogger
from .traffic import TrafficGenerator, default_gamma_windows, load_trace_csv


def run_simulation(
    t_end: float = 60 * 60,
    num_workers: int = 8,
    policy: str = "latency_only",
    arrival_mode: str = "modeled_gamma",
    service_classes: int = 1,
    ewma_gamma: float = 0.10,
    seed: int = 42,
    trace_file: Optional[Path] = None,
    full_log: bool = False,
    full_log_file: Optional[Path] = None,
) -> Dict[str, object]:
    """Run one simulation and return aggregate metrics."""

    rng = random.Random(seed)
    env = simpy.Environment()

    trace_timestamps: List[float] = []
    if arrival_mode == "trace_replay":
        if trace_file is None:
            raise ValueError("trace_file is required when arrival_mode='trace_replay'")
        trace_timestamps = load_trace_csv(trace_file)

    metrics = MetricsCollector(num_workers=num_workers)
    logger: Optional[RequestCsvLogger] = None
    log_path: Optional[Path] = None

    if full_log:
        log_path = full_log_file or Path("request_full_log.csv")
        logger = RequestCsvLogger(log_path)
        logger.open()

    load_balancer = LoadBalancer(
        num_workers=num_workers,
        policy=policy,
        ewma_gamma=ewma_gamma,
        rng=rng,
    )
    inference_pool = InferencePool(
        env=env,
        num_workers=num_workers,
        st_params=ServiceTimeParams(n0=max(1, num_workers * 4)),
        metrics=metrics,
        on_request_done=logger.write if logger else None,
        rng=rng,
    )

    def on_arrival(request: Request) -> None:
        worker_id = load_balancer.choose_worker(request)
        load_balancer.on_dispatch(worker_id)
        inference_pool.dispatch(request, worker_id, load_balancer)

    traffic_generator = TrafficGenerator(
        env=env,
        t_end=t_end,
        arrival_mode=arrival_mode,
        on_request=on_arrival,
        rng=rng,
        service_classes=service_classes,
        gamma_windows=default_gamma_windows(t_end),
        trace_timestamps=trace_timestamps,
    )

    try:
        env.process(traffic_generator.run())
        # Keep running until all pending requests are drained.
        env.run()
    finally:
        if logger:
            logger.close()

    summary = metrics.summarize(sim_time=env.now, active_time=t_end)
    summary["sim_time_total"] = env.now
    summary["drain_time"] = max(0.0, env.now - t_end)
    summary["policy"] = policy
    summary["workers"] = num_workers
    summary["arrival_mode"] = arrival_mode
    summary["full_log_enabled"] = full_log
    summary["full_log_file"] = str(log_path) if log_path else ""
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Event-driven latency-only LB simulator (SimPy)")
    parser.add_argument("--t-end", type=float, default=3600.0, help="Simulation horizon in seconds")
    parser.add_argument("--workers", type=int, default=8, help="Number of backend workers")
    parser.add_argument(
        "--policy",
        type=str,
        default="latency_only",
        choices=["latency_only", "peak_ewma", "least_inflight", "round_robin", "random"],
        help="Load balancing policy",
    )
    parser.add_argument(
        "--arrival-mode",
        type=str,
        default="modeled_gamma",
        choices=["modeled_gamma", "trace_replay"],
        help="Arrival process mode",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="Trace CSV path (required for trace_replay mode)",
    )
    parser.add_argument(
        "--service-classes", type=int, default=1, help="Number of service classes/tenants"
    )
    parser.add_argument("--ewma-gamma", type=float, default=0.10, help="EWMA smoothing factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--full-log",
        action="store_true",
        help="Write every completed request to a CSV file",
    )
    parser.add_argument(
        "--full-log-file",
        type=Path,
        default=Path("request_full_log.csv"),
        help="CSV path used when --full-log is enabled",
    )
    return parser


def print_summary(summary: Dict[str, object]) -> None:
    """Pretty-print simulation summary."""

    print("=== Simulation Summary ===")
    print(f"policy                : {summary['policy']}")
    print(f"arrival_mode          : {summary['arrival_mode']}")
    print(f"workers               : {summary['workers']}")
    print(f"dispatched            : {summary['dispatched']}")
    print(f"completed             : {summary['completed']}")
    print(f"throughput (req/s)    : {summary['throughput']:.4f}")
    print(f"mean latency (s)      : {summary['mean_latency']:.4f}")
    print(f"median latency (s)    : {summary['median_latency']:.4f}")
    print(f"p95 latency (s)       : {summary['p95_latency']:.4f}")
    print(f"p99 latency (s)       : {summary['p99_latency']:.4f}")
    print(f"avg queue len         : {summary['avg_queue_len']:.4f}")
    print(f"avg global inflight   : {summary['avg_global_inflight']:.4f}")
    print(f"avg utilization       : {summary['avg_utilization']:.4f}")
    print(f"sim time total (s)    : {summary['sim_time_total']:.4f}")
    print(f"drain time (s)        : {summary['drain_time']:.4f}")
    print(f"full log enabled      : {summary['full_log_enabled']}")
    if summary["full_log_enabled"]:
        print(f"full log file         : {summary['full_log_file']}")

    by_class = summary.get("latency_by_class", {})
    if isinstance(by_class, dict) and by_class:
        print("\nLatency by class:")
        for class_id, stats in by_class.items():
            print(
                f"  class {class_id}: count={stats['count']} "
                f"mean={stats['mean']:.4f}s p95={stats['p95']:.4f}s"
            )


def main() -> None:
    """CLI main function."""

    args = build_arg_parser().parse_args()
    summary = run_simulation(
        t_end=args.t_end,
        num_workers=args.workers,
        policy=args.policy,
        arrival_mode=args.arrival_mode,
        service_classes=args.service_classes,
        ewma_gamma=args.ewma_gamma,
        seed=args.seed,
        trace_file=args.trace_file,
        full_log=args.full_log,
        full_log_file=args.full_log_file,
    )
    print_summary(summary)
