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
from .traffic import (
    TrafficGenerator,
    load_service_class_config,
)


def run_simulation(
    t_end: float = 60 * 60,
    num_workers: int = 8,
    policy: str = "latency_only",
    service_class_config: Optional[Path] = None,
    ewma_gamma: float = 0.10,
    seed: int = 42,
    full_log: bool = False,
    full_log_file: Optional[Path] = None,
) -> Dict[str, object]:
    """Run one simulation and return aggregate metrics."""

    rng = random.Random(seed)
    env = simpy.Environment()

    if service_class_config is None:
        raise ValueError("service_class_config is required.")

    class_specs = load_service_class_config(service_class_config, t_end=t_end)
    if not class_specs:
        raise ValueError("service_class_config does not contain any class entry.")
    effective_service_classes = len(class_specs)

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

    rid_counter = 0

    def next_rid() -> int:
        nonlocal rid_counter
        rid = rid_counter
        rid_counter += 1
        return rid

    traffic_generators: List[TrafficGenerator] = []
    for spec in class_specs:
        class_rng = random.Random(rng.randrange(1, 2**31))
        traffic_generators.append(
            TrafficGenerator(
                env=env,
                t_end=t_end,
                arrival_mode=spec.arrival_mode,
                on_request=on_arrival,
                rng=class_rng,
                service_classes=1,
                zipf_s=spec.zipf_s,
                zipf_xmin=spec.zipf_xmin,
                zipf_max=spec.zipf_max,
                gamma_windows=spec.gamma_windows,
                trace_records=spec.trace_records,
                model=spec.model,
                log_type=spec.log_type,
                fixed_class_id=spec.class_id,
                next_rid=next_rid,
            )
        )

    try:
        for traffic_generator in traffic_generators:
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
    summary["arrival_mode"] = "per_class_config"
    summary["service_classes"] = effective_service_classes
    summary["service_class_config_file"] = str(service_class_config)
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
        "--service-class-config",
        type=Path,
        required=True,
        help="Required JSON config for per-class traffic.",
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
    print(f"service classes       : {summary['service_classes']}")
    if summary["service_class_config_file"]:
        print(f"service class config  : {summary['service_class_config_file']}")
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
        service_class_config=args.service_class_config,
        ewma_gamma=args.ewma_gamma,
        seed=args.seed,
        full_log=args.full_log,
        full_log_file=args.full_log_file,
    )
    print_summary(summary)
