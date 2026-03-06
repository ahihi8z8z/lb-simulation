"""Simulation assembly, CLI parsing, and summary printing."""

import argparse
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import simpy

from .controller import LoadBalancerController, load_controller_config
from .inference_pool import InferencePool
from .load_balancer import (
    LoadBalancer,
    supported_policies,
)
from .metrics import MetricsCollector
from .models import Request
from .request_csv_logger import RequestCsvLogger
from .traffic import (
    TrafficGenerator,
    load_service_class_config,
)
from .workers import expand_worker_specs, load_worker_class_config


_DURATION_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smhdSMHD]?)\s*$"
)


def parse_duration_seconds(raw: str) -> float:
    """
    Parse duration text to seconds.

    Supported examples:
    - "300" (seconds)
    - "90s"
    - "1m"
    - "2h"
    - "3d"
    """

    match = _DURATION_PATTERN.match(str(raw))
    if not match:
        raise ValueError(
            f"Invalid --t-end value: {raw}. Supported forms: 300, 90s, 1m, 2h, 3d."
        )

    value = float(match.group("value"))
    unit = match.group("unit").lower()
    multiplier = {
        "": 1.0,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }[unit]
    seconds = value * multiplier
    if seconds <= 0:
        raise ValueError("--t-end must be > 0.")
    return seconds


def _create_run_dir(logs_root: Path) -> Path:
    """Create a unique run directory under logs_root."""

    logs_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = logs_root / run_stamp
    suffix = 1
    while run_dir.exists():
        run_dir = logs_root / f"{run_stamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write a JSON file with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_simulation(
    t_end: float = 60 * 60,
    policy: str = "latency_only",
    service_class_config: Optional[Path] = None,
    worker_class_config: Optional[Path] = None,
    controller_config: Optional[Path] = None,
    seed: int = 42,
    full_log: bool = False,
    logs_root: Path = Path("logs"),
) -> Dict[str, object]:
    """Run one simulation and return aggregate metrics."""

    rng = random.Random(seed)
    env = simpy.Environment()

    if service_class_config is None:
        raise ValueError("service_class_config is required.")
    if worker_class_config is None:
        raise ValueError("worker_class_config is required.")

    class_specs = load_service_class_config(service_class_config, t_end=t_end)
    if not class_specs:
        raise ValueError("service_class_config does not contain any class entry.")
    effective_service_classes = len(class_specs)
    worker_class_specs = load_worker_class_config(worker_class_config)
    effective_worker_classes = len(worker_class_specs)
    worker_specs = expand_worker_specs(worker_class_specs)
    num_workers = len(worker_specs)
    controller_cfg = load_controller_config(controller_config)
    controller = LoadBalancerController(
        policy=policy,
        num_workers=num_workers,
        config=controller_cfg,
        rng=random.Random(rng.randrange(1, 2**31)),
    )

    run_dir = _create_run_dir(logs_root)
    service_config_snapshot_path = run_dir / "service_class_config.json"
    worker_config_snapshot_path = run_dir / "worker_class_config.json"
    controller_config_snapshot_path: Optional[Path] = None
    shutil.copy2(service_class_config, service_config_snapshot_path)
    shutil.copy2(worker_class_config, worker_config_snapshot_path)
    if controller_config is not None:
        controller_config_snapshot_path = run_dir / "controller_config.json"
        shutil.copy2(controller_config, controller_config_snapshot_path)
    run_config_path = run_dir / "run_config.json"
    _write_json(
        run_config_path,
        {
            "controller_config": str(controller_config) if controller_config else "",
            "controller_config_snapshot_file": (
                str(controller_config_snapshot_path) if controller_config_snapshot_path else ""
            ),
            "controller_mode": controller_cfg.mode,
            "full_log": full_log,
            "policy": policy,
            "seed": seed,
            "service_class_config": str(service_class_config),
            "service_class_config_snapshot_file": str(service_config_snapshot_path),
            "t_end": t_end,
            "worker_class_config": str(worker_class_config),
            "worker_class_config_snapshot_file": str(worker_config_snapshot_path),
            "worker_classes": effective_worker_classes,
            "workers": num_workers,
            "lb_workers": num_workers + (1 if controller.latency_tracker_enabled else 0),
        },
    )

    metrics = MetricsCollector(num_workers=num_workers)
    logger: Optional[RequestCsvLogger] = None
    log_path: Optional[Path] = None

    if full_log:
        log_path = run_dir / "request_full_log.csv"
        logger = RequestCsvLogger(log_path)
        logger.open()

    load_balancer = LoadBalancer(
        num_workers=num_workers,
        policy=policy,
        rng=rng,
    )
    controller.initialize(load_balancer)

    def on_complete(
        request: Request,
        worker_id: int,
        latency: float,
        latency_tracked: bool,
    ) -> None:
        controller.on_request_complete(
            request=request,
            worker_id=worker_id,
            latency=latency,
            latency_tracked=latency_tracked,
            lb=load_balancer,
        )

    inference_pool = InferencePool(
        env=env,
        worker_specs=worker_specs,
        metrics=metrics,
        on_complete=on_complete,
        on_request_done=logger.write if logger else None,
        rng=rng,
    )

    def on_arrival(request: Request) -> None:
        lb_selected_worker_id = load_balancer.choose_worker(request)
        if controller.is_latency_tracker_worker(lb_selected_worker_id):
            # The latency-tracker worker itself has zero service time and only forwards
            # to a real worker (RR or selected-worker mode, depending on policy).
            selected_worker_id = load_balancer.consume_redirect_target(request.rid)
            forwarded_worker_id = controller.forward_via_latency_tracker(
                request,
                selected_worker_id=selected_worker_id,
            )
            load_balancer.on_dispatch(lb_selected_worker_id)
            load_balancer.on_dispatch(forwarded_worker_id)
            inference_pool.dispatch(
                request,
                forwarded_worker_id,
                load_balancer,
                latency_tracked=True,
                lb_completion_worker_ids=[
                    forwarded_worker_id,
                    lb_selected_worker_id,
                ],
                lb_selected_worker_id=(
                    selected_worker_id
                    if selected_worker_id is not None
                    else lb_selected_worker_id
                ),
                routed_via_latency_tracker=True,
            )
            return

        load_balancer.on_dispatch(lb_selected_worker_id)
        inference_pool.dispatch(
            request,
            lb_selected_worker_id,
            load_balancer,
            latency_tracked=False,
            lb_completion_worker_ids=[lb_selected_worker_id],
            lb_selected_worker_id=lb_selected_worker_id,
            routed_via_latency_tracker=False,
        )

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
    summary["lb_workers"] = num_workers + (1 if controller.latency_tracker_enabled else 0)
    summary["worker_classes"] = effective_worker_classes
    summary["arrival_mode"] = "per_class_config"
    summary["service_classes"] = effective_service_classes
    summary["service_class_config_file"] = str(service_class_config)
    summary["worker_class_config_file"] = str(worker_class_config)
    summary["controller_config_file"] = str(controller_config) if controller_config else ""
    summary["full_log_enabled"] = full_log
    summary["full_log_file"] = str(log_path) if log_path else ""
    summary["run_config_file"] = str(run_config_path)
    summary["run_dir"] = str(run_dir)
    summary["service_class_config_snapshot_file"] = str(service_config_snapshot_path)
    summary["worker_class_config_snapshot_file"] = str(worker_config_snapshot_path)
    summary["controller_config_snapshot_file"] = (
        str(controller_config_snapshot_path) if controller_config_snapshot_path else ""
    )
    summary["controller"] = controller.summarize(load_balancer)
    summary_file = run_dir / "summary.json"
    summary["summary_file"] = str(summary_file)
    _write_json(summary_file, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Event-driven latency-only LB simulator (SimPy)")
    policy_choices = supported_policies()
    parser.add_argument(
        "--t-end",
        type=str,
        default="1h",
        help="Simulation horizon: seconds or duration suffix (e.g. 300, 90s, 1m, 2h, 3d)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="latency_only",
        choices=policy_choices,
        help=f"Load balancing policy ({', '.join(policy_choices)})",
    )
    parser.add_argument(
        "--service-class-config",
        type=Path,
        required=True,
        help="Required JSON config for per-class traffic.",
    )
    parser.add_argument(
        "--worker-class-config",
        type=Path,
        required=True,
        help="Required JSON config for worker classes and service models.",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help="Optional JSON config for controller (weight control / latency tracker tuning).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--full-log",
        action="store_true",
        help="Write every completed request to a CSV file",
    )
    return parser


def print_summary(summary: Dict[str, object]) -> None:
    """Pretty-print simulation summary."""

    print("=== Simulation Summary ===")
    print(f"policy                : {summary['policy']}")
    print(f"arrival_mode          : {summary['arrival_mode']}")
    print(f"workers               : {summary['workers']}")
    print(f"lb workers            : {summary.get('lb_workers', summary['workers'])}")
    print(f"worker classes        : {summary['worker_classes']}")
    print(f"service classes       : {summary['service_classes']}")
    if summary["service_class_config_file"]:
        print(f"service class config  : {summary['service_class_config_file']}")
    if summary["worker_class_config_file"]:
        print(f"worker class config   : {summary['worker_class_config_file']}")
    if summary["controller_config_file"]:
        print(f"controller config     : {summary['controller_config_file']}")
    print(f"run dir               : {summary['run_dir']}")
    print(f"run config file       : {summary['run_config_file']}")
    print(f"summary file          : {summary['summary_file']}")
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
    controller = summary.get("controller")
    if isinstance(controller, dict):
        print(f"controller mode       : {controller.get('mode', '')}")
        print(f"latency tracker       : {controller.get('latency_tracker_enabled', False)}")
        print(f"latency samples       : {controller.get('latency_samples_total', 0)}")
        print(
            "latency redirects     : "
            f"{controller.get('track_redirected', 0)} / {controller.get('track_decisions', 0)}"
        )
        redirect_policy = controller.get("latency_redirect_policy", {})
        if isinstance(redirect_policy, dict) and redirect_policy.get("name"):
            print(f"redirect policy       : {redirect_policy.get('name')}")
        print(f"wrr control mode      : {controller.get('wrr_control_mode', 'none')}")

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
    try:
        t_end_seconds = parse_duration_seconds(args.t_end)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    summary = run_simulation(
        t_end=t_end_seconds,
        policy=args.policy,
        service_class_config=args.service_class_config,
        worker_class_config=args.worker_class_config,
        controller_config=args.controller_config,
        seed=args.seed,
        full_log=args.full_log,
    )
    print_summary(summary)
