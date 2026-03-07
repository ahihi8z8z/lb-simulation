"""Simulation assembly, CLI parsing, and summary printing."""

import argparse
import json
import logging
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import simpy

from .controller import LATENCY_AWARE_POLICIES, LoadBalancerController, load_controller_config
from .inference_pool import InferencePool
from .logging_utils import (
    configure_logging,
    normalize_log_mode,
    set_simulation_time_provider,
)
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

logger = logging.getLogger(__name__)


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
    detail: bool = False,
    logger_mode: str = "INFO",
    logs_root: Path = Path("logs"),
) -> Dict[str, object]:
    """Run one simulation and return aggregate metrics."""
    wall_clock_start = time.perf_counter()
    set_simulation_time_provider(None)

    if service_class_config is None:
        raise ValueError("service_class_config is required.")
    if worker_class_config is None:
        raise ValueError("worker_class_config is required.")
    normalized_policy = policy.strip().lower()
    if (normalized_policy in LATENCY_AWARE_POLICIES) and (controller_config is None):
        raise ValueError(
            f"Policy '{normalized_policy}' requires --controller-config with "
            "latency_tracker.enabled=true."
        )

    run_dir = _create_run_dir(logs_root)
    runtime_log_path = configure_logging(run_dir, mode=logger_mode)
    normalized_logger_mode = normalize_log_mode(logger_mode)
    logger.info(
        "Starting simulation policy=%s t_end=%.3f service_config=%s worker_config=%s",
        policy,
        t_end,
        service_class_config,
        worker_class_config,
    )
    logger.debug("Run directory prepared at %s", run_dir)

    rng = random.Random(seed)
    env = simpy.Environment()
    set_simulation_time_provider(lambda: env.now)

    class_specs = load_service_class_config(service_class_config, t_end=t_end)
    if not class_specs:
        raise ValueError("service_class_config does not contain any class entry.")
    effective_service_classes = len(class_specs)
    service_class_descriptions = {
        str(spec.class_id): spec.description
        for spec in class_specs
        if spec.description
    }
    worker_class_specs = load_worker_class_config(worker_class_config)
    effective_worker_classes = len(worker_class_specs)
    worker_class_descriptions = {
        str(spec.class_id): spec.description
        for spec in worker_class_specs
        if spec.description
    }
    worker_specs = expand_worker_specs(worker_class_specs)
    num_workers = len(worker_specs)
    controller_cfg = load_controller_config(controller_config)
    controller = LoadBalancerController(
        policy=normalized_policy,
        num_workers=num_workers,
        config=controller_cfg,
        rng=random.Random(rng.randrange(1, 2**31)),
    )
    logger.info(
        "Loaded %d service classes, %d worker classes, %d workers",
        effective_service_classes,
        effective_worker_classes,
        num_workers,
    )

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
            "detail": detail,
            "logger_mode": normalized_logger_mode,
            "runtime_log_file": str(runtime_log_path),
            "policy": policy,
            "seed": seed,
            "service_class_config": str(service_class_config),
            "service_class_config_snapshot_file": str(service_config_snapshot_path),
            "service_class_descriptions": service_class_descriptions,
            "t_end": t_end,
            "worker_class_config": str(worker_class_config),
            "worker_class_config_snapshot_file": str(worker_config_snapshot_path),
            "worker_class_descriptions": worker_class_descriptions,
            "worker_classes": effective_worker_classes,
            "workers": num_workers,
            "lb_workers": num_workers + (1 if controller.latency_tracker_enabled else 0),
        },
    )

    metrics = MetricsCollector(num_workers=num_workers)
    detail_writer: Optional[RequestCsvLogger] = None
    detail_path: Optional[Path] = None

    if detail:
        detail_path = run_dir / "request_detail_metrics.csv"
        detail_writer = RequestCsvLogger(detail_path)
        detail_writer.open()
        logger.info("Detail metrics enabled at %s", detail_path)

    load_balancer = LoadBalancer(
        num_workers=num_workers,
        policy=normalized_policy,
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
        on_request_done=detail_writer.write if detail_writer else None,
        rng=rng,
    )

    def _build_detail_state() -> Optional[Dict[str, object]]:
        if not detail:
            return None
        lb_state = {
            "inflight": list(load_balancer.inflight),
            "lat_ewma": list(load_balancer.lat_ewma),
            "worker_weights": list(load_balancer.worker_weights),
            "penalty": list(load_balancer.penalty),
            "feedback_count": list(load_balancer.feedback_count),
        }
        lb_control_state: Dict[str, object] = {}
        if controller.lb_control_module is not None and controller.lb_control_module.name != "none":
            lb_control_state = controller.lb_control_module.summarize(load_balancer)
        return {"lb_state": lb_state, "lb_control_state": lb_control_state}

    def on_arrival(request: Request) -> None:
        detail_state = _build_detail_state()
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
                detail_state=detail_state,
            )
            logger.debug(
                "Request rid=%d routed via tracker -> worker=%d",
                request.rid,
                forwarded_worker_id,
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
            detail_state=detail_state,
        )
        logger.debug(
            "Request rid=%d routed directly -> worker=%d",
            request.rid,
            lb_selected_worker_id,
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
                trace_traffic_scale=spec.trace_traffic_scale,
                zipf_s=spec.zipf_s,
                zipf_xmin=spec.zipf_xmin,
                zipf_max=spec.zipf_max,
                response_slope=spec.response_slope,
                response_intercept=spec.response_intercept,
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
        if detail_writer:
            detail_writer.close()

    summary = metrics.summarize(sim_time=env.now, active_time=t_end)
    summary["wall_time_total"] = time.perf_counter() - wall_clock_start
    summary["sim_time_total"] = env.now
    summary["drain_time"] = max(0.0, env.now - t_end)
    summary["policy"] = normalized_policy
    summary["workers"] = num_workers
    summary["lb_workers"] = num_workers + (1 if controller.latency_tracker_enabled else 0)
    summary["worker_classes"] = effective_worker_classes
    summary["arrival_mode"] = "per_class_config"
    summary["service_classes"] = effective_service_classes
    summary["service_class_config_file"] = str(service_class_config)
    summary["worker_class_config_file"] = str(worker_class_config)
    summary["controller_config_file"] = str(controller_config) if controller_config else ""
    summary["detail_enabled"] = detail
    summary["detail_metrics_file"] = str(detail_path) if detail_path else ""
    summary["logger_mode"] = normalized_logger_mode
    summary["runtime_log_file"] = str(runtime_log_path)
    summary["run_config_file"] = str(run_config_path)
    summary["run_dir"] = str(run_dir)
    summary["service_class_config_snapshot_file"] = str(service_config_snapshot_path)
    summary["service_class_descriptions"] = service_class_descriptions
    summary["worker_class_config_snapshot_file"] = str(worker_config_snapshot_path)
    summary["worker_class_descriptions"] = worker_class_descriptions
    summary["controller_config_snapshot_file"] = (
        str(controller_config_snapshot_path) if controller_config_snapshot_path else ""
    )
    summary["controller"] = controller.summarize(load_balancer)
    summary_file = run_dir / "summary.json"
    summary["summary_file"] = str(summary_file)
    _write_json(summary_file, summary)
    logger.info(
        "Simulation completed dispatched=%s completed=%s mean_latency=%.4f wall_time=%.3fs",
        summary["dispatched"],
        summary["completed"],
        summary["mean_latency"],
        summary["wall_time_total"],
    )
    set_simulation_time_provider(None)
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
        help=(
            "JSON config for controller (weight control / latency tracker tuning). "
            "Required for latency-aware policies."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Write per-request detail metrics to CSV",
    )
    parser.add_argument(
        "--logger-mode",
        type=str,
        default="INFO",
        help="Logger mode for console/file output (DEBUG or INFO)",
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
    service_desc = summary.get("service_class_descriptions", {})
    worker_desc = summary.get("worker_class_descriptions", {})
    if isinstance(service_desc, dict) and service_desc:
        print(f"service class desc    : {len(service_desc)} configured")
    if isinstance(worker_desc, dict) and worker_desc:
        print(f"worker class desc     : {len(worker_desc)} configured")
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
    print(f"wall time total (s)   : {summary.get('wall_time_total', 0.0):.4f}")
    print(f"drain time (s)        : {summary['drain_time']:.4f}")
    print(f"detail enabled        : {summary['detail_enabled']}")
    if summary["detail_enabled"]:
        print(f"detail metrics file   : {summary['detail_metrics_file']}")
    print(f"logger mode           : {summary.get('logger_mode', 'INFO')}")
    if summary.get("runtime_log_file"):
        print(f"runtime log file      : {summary['runtime_log_file']}")
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

    by_worker = summary.get("latency_by_worker", {})
    if isinstance(by_worker, dict) and by_worker:
        print("\nLatency by worker:")
        for worker_id, stats in by_worker.items():
            print(
                f"  worker {worker_id}: count={stats['count']} "
                f"mean={stats['mean']:.4f}s p95={stats['p95']:.4f}s"
            )


def main() -> None:
    """CLI main function."""

    args = build_arg_parser().parse_args()
    try:
        t_end_seconds = parse_duration_seconds(args.t_end)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        normalized_mode = normalize_log_mode(args.logger_mode)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    summary = run_simulation(
        t_end=t_end_seconds,
        policy=args.policy,
        service_class_config=args.service_class_config,
        worker_class_config=args.worker_class_config,
        controller_config=args.controller_config,
        seed=args.seed,
        detail=args.detail,
        logger_mode=normalized_mode,
    )
    print_summary(summary)
