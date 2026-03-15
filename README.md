# LB Simulation

An event-driven load-balancing simulator built on SimPy for LLM-like workloads. It supports trace replay, gamma-modeled traffic, multiple worker classes, per-service-class topology, and controller-assisted latency-aware policies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the plotting and analysis scripts under `tools/`:

```bash
pip install -r tools/requirements.txt
```

## Supported Policies

- `random`
- `static-wrr`
- `lp-wrr`
- `sp-wrr`
- `least_connection`
- `power_of_two_choices`
- `min_ema_latency`
- `latency_p2c`

Common CLI aliases:

- `swrr` -> `static-wrr`
- `lpwrr` -> `lp-wrr`
- `spwrr` -> `sp-wrr`
- `lc` -> `least_connection`
- `p2c` -> `power_of_two_choices`
- `mema` -> `min_ema_latency`
- `lp2c` -> `latency_p2c`

## Run a Single Simulation

Example:

```bash
.venv/bin/python simulator.py \
  -f configs/simulation.example.json \
  -p swrr \
  -t 30m
```

Main options:

- `-f`, `--config`: required unified simulation config.
- `-p`, `--policy`: policy name or alias.
- `-t`, `--t-end`: simulation duration; accepts `300`, `90s`, `1m`, `2h`, `3d`.
- `-S`, `--seed`: global random seed, default `42`.
- `-d`, `--detail`: write `request_detail_metrics.csv`.
- `-l`, `--log-level`: `DEBUG` or `INFO`.

Each run creates `logs/run-YYYYMMDD-HHMMSS/` with:

- `run_config.json`
- `service_class_config.json`
- `worker_class_config.json`
- `topology_config.json` when configured
- `controller_config.json` when configured
- `simulation_config.json`
- `summary.json`
- `runtime.log`
- `request_detail_metrics.csv` when `-d` is enabled

## Run a Sweep

Example:

```bash
.venv/bin/python tools/sweep_controller_configs.py \
  -f configs/sweep_plan.example.json \
  -p swrr,lpwrr,spwrr \
  -j 8 \
  -t 30m
```

Main options:

- `-f`, `--config`: required sweep plan JSON.
- `-p`, `--policy`: one policy or a comma-separated policy list.
- `-t`, `--t-end`: simulation duration for each case.
- `-j`, `--jobs`: parallel worker processes; `1` is sequential, `0` uses all CPU cores.
- `-S`, `--seed`: random seed.
- `-d`, `--detail`: write detail CSVs for each child run.
- `-l`, `--log-level`: `DEBUG` or `INFO`.
- `--report-dir`: output directory for sweep artifacts.
- `--resume-report-dir`: resume an unfinished sweep from an existing report directory.
- `--output-csv`: result CSV path; default is `<report_dir>/controller_sweep_results.csv`.
- `--continue-on-error`: continue remaining cases after a failure.

Sweep output is written to `logs/sweep-YYYYMMDD-HHMMSS/` by default.

## Main Simulation Config

Example file: `configs/simulation.example.json`

Top-level keys:

- `service_class`: required.
- `worker_class`: required.
- `topology`: optional.
- `controller`: optional, but required by some policies.

### `service_class`

Schema:

```json
{
  "service_class": {
    "classes": [
      {
        "class_id": 0,
        "arrival_mode": "trace_replay",
        "description": "optional",
        "model": "ChatGPT",
        "log_type": "Conversation log",
        "trace_file": "../traces/BurstGPT_without_fails_1.csv",
        "traffic_scale": 1
      }
    ]
  }
}
```

Fields in `service_class.classes[]`:

- `class_id`: unique integer; defaults to the list index when omitted.
- `description`: free-form description.
- `arrival_mode`: `trace_replay` or `modeled_gamma`; default is `modeled_gamma`.
- `model`: traffic-class model label. In `trace_replay` mode it filters matching trace rows.
- `log_type`: traffic-class log type label. In `trace_replay` mode it filters matching trace rows.
- `seed`: optional integer seed for modeled traffic.

When `arrival_mode = "trace_replay"`:

- `trace_file`: required path to the trace CSV.
- `traffic_scale`: positive integer multiplier for the trace load.
- Do not set `gamma_params_file`, `zipf_params_file`, `scale_gamma`, `scale_beta`, or `response_linear`.

When `arrival_mode = "modeled_gamma"`:

- `gamma_params_file`: required gamma-window parameter file.
- `zipf_params_file`: required Zipf parameter file.
- `scale_gamma`: gamma scale multiplier, default `1.0`, must be > `0`.
- `scale_beta`: beta scale multiplier, default `1.0`, must be > `0`.
- `response_linear`: optional object for converting sampled request length to response length.
- `response_linear.slope`: default `1.0`, must be >= `0`.
- `response_linear.intercept`: default `0.0`.
- Do not set `traffic_scale`.

Notes:

- Paths inside `service_class` are resolved relative to the config file directory.
- The old per-service `worker_ids` field is no longer supported; use `topology.service_class_worker_ids`.

### `worker_class`

Schema:

```json
{
  "worker_class": {
    "classes": [
      {
        "class_id": 0,
        "count": 4,
        "service_model": "contention_lognormal",
        "queue_policy": "fcfs",
        "queue_timeout_seconds": null,
        "params": {}
      }
    ]
  }
}
```

Fields in `worker_class.classes[]`:

- `class_id`: unique integer; defaults to the list index when omitted.
- `description`: free-form description.
- `count`: number of worker instances in the class; must be > `0`.
- `service_model`: worker service-time model.
- `queue_policy`: `fcfs` or `sjf`; aliases such as `fifo`, `shortest_job_first`, and `shortest_request_first` are accepted.
- `queue_timeout_seconds`: `null` or a number >= `0`; requests waiting longer than this are dropped from that worker queue.
- `params`: model-specific parameter object.

Supported `service_model` values:

- `contention_lognormal`
- `linear_lognormal`
- `fixed`
- `fixed_linear`
- `limited_processor_sharing`

`params` fields by `service_model`:

- `contention_lognormal`: `a`, `b`, `c`, `d`, `n0`, `sigma`, `min_s`
- `linear_lognormal`: `a`, `b`, `sigma`, `min_s`
- `fixed`: `service_time`
- `fixed_linear`: `a`, `b`, `min`, `max`
- `limited_processor_sharing`: `processing_rate`, `max_concurrency`

### `topology`

`topology` is optional. Use it to restrict which workers each service class can reach or to provide initial WRR weights per class.

Supported fields:

- `service_class_worker_ids`: map `class_id -> [worker_id, ...]`
- `service_class_worker_weights`: map `class_id -> { worker_id: weight }`

Constraints:

- `service_class_worker_ids[class_id]` must be a non-empty list with no duplicate `worker_id` values.
- `worker_id` must be within the real worker range after expanding `worker_class`.
- If `service_class_worker_weights` is present, every weight must be > `0`.
- If `service_class_worker_weights` is present, it must provide weights for every connected worker of that class.
- If a class has no `service_class_worker_ids`, it can reach all workers by default.
- `service_class_worker_weights` is most useful with WRR policies: `static-wrr`, `lp-wrr`, `sp-wrr`.

### `controller`

Policies that use `controller`:

- `random`, `static-wrr`, `least_connection`, `power_of_two_choices`: ignore `controller`.
- `min_ema_latency`, `latency_p2c`: require `controller.latency_tracker`.
- `lp-wrr`: require `controller.latency_tracker` and `controller.lp-wrr`.
- `sp-wrr`: require `controller.latency_tracker` and `controller.sp-wrr`.

#### `controller.latency_tracker`

Supported fields:

- `init_estimate`: initial estimate, default `0.5`, clamped to a value > `0`.
- `ewma_gamma`: EWMA coefficient in `0..1`, default `0.1`.
- `redirect_policy`: redirect policy name or redirect policy config object.

`redirect_policy` can be written as:

```json
"redirect_policy": "track_all"
```

or:

```json
"redirect_policy": {
  "name": "fixed_rate",
  "params": {
    "rate": 0.05
  }
}
```

Available redirect policies:

- `fixed_rate`: redirect a fixed fraction of requests. Use `params.rate` in the range `0..1`.
- `track_all`: route every tracked request through the tracker.

If `redirect_policy` is omitted, the code uses `fixed_rate` with the default `rate = 0.05`, or reads the legacy `sample_rate` field when present.

#### `controller.lp-wrr` and `controller.sp-wrr`

These two blocks share the same schema, but only the block matching the active policy is used. Do not configure both in the same controller payload.

Supported fields:

- `update_interval_seconds`: weight-update interval, default `60.0`.
- `min_weight`: minimum weight, default `0.1`.
- `max_weight`: maximum weight, default `10.0`, must be >= `min_weight`.
- `lp_balance_tolerance`: controller balance tolerance, default `0.25`.
- `lp_ewma_gamma`: controller EWMA coefficient in `0..1`, default `0.1`.

## Sweep Config

Example file: `configs/sweep_plan.example.json`

Top-level keys:

- `base_config`: required base config merged into every scenario. Its schema matches the main simulation config.
- `scenarios`: required scenario list.
- `sweeps`: optional map from `scenario_name` to that scenario's sweep variables.

### `scenarios[]`

Each entry in `scenarios` has:

- `name`: required unique scenario name.
- `override`: object merged on top of `base_config` via deep merge.

Example:

```json
{
  "name": "split_topology",
  "override": {
    "topology": {
      "service_class_worker_ids": {
        "0": [0, 1, 2, 3],
        "1": [4, 5, 6, 7]
      }
    }
  }
}
```

### `sweeps`

`sweeps` has the form:

```json
{
  "sweeps": {
    "baseline": {
      "service_class.classes[0].traffic_scale": [1, 2, 3],
      "controller.latency_tracker.ewma_gamma": {
        "min": 0.05,
        "max": 0.20,
        "step": 0.05
      }
    }
  }
}
```

Each entry in `sweeps[scenario_name]` is:

- key: the field path to sweep.
- value: either a value list or a range object `{ "min": ..., "max": ..., "step": ... }`.

Supported path syntax:

- object keys separated by `.`
- list indexes written as `[i]`

Valid examples:

- `worker_class.classes[0].params.a`
- `service_class.classes[1].scale_gamma`
- `controller.lp-wrr.update_interval_seconds`
- `topology.service_class_worker_ids.0`

Behavior and constraints:

- Names inside `sweeps` must match an existing `scenario.name`.
- Every sweep variable must provide a non-empty list or a valid range with `step > 0`.
- When `-p` includes multiple policies, the tool automatically drops controller fields that do not apply to the current policy.
- For policies that do not use a controller, the entire `controller` block is removed from the runtime payload.
- Paths inside `base_config.service_class` are resolved relative to the sweep-plan directory.

## Utility Scripts

The scripts in `tools/` can still be used directly. The fastest way to inspect script options is:

```bash
.venv/bin/python tools/<script_name>.py --help
```

Common scripts:

- `tools/sweep_controller_configs.py`
- `tools/plot_detail_metrics.py`
- `tools/plot_log_comparison.py`
- `tools/plot_csv_chart.py`
- `tools/extract_lp_weights_csv.py`
