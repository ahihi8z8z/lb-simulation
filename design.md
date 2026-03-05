# Latency-Only Load Balancing Simulator for LLM-like Workloads

This document describes the **system design for a simulator** used to evaluate a **latency-feedback load balancer** under **LLM-like workloads** derived from BurstGPT patterns.

The simulator models a **black-box dispatching layer** that observes only **end-to-end latency feedback**, without access to request size, token length, or KV-cache information.

---

# 1. High-Level Architecture

The system consists of four main components.


TrafficGenerator (BurstGPT-pattern)
→ LoadBalancer / Dispatcher (latency-only)
→ InferencePool (shared backend workers)
→ Response / Completion
→ Latency feedback to LoadBalancer


Key idea:

- The **load balancer sees only latency feedback**
- Backend service time depends on **hidden job size and contention**
- Multiple dispatchers can compete for the same backend pool

---

# 2. TrafficGenerator

## Purpose

Generates request arrivals that follow **bursty traffic patterns similar to BurstGPT workloads**.

## Inputs / Configuration

| Parameter | Description |
|---|---|
| `T_end` | Total simulation time |
| `arrival_mode` | `TraceReplay` or `ModeledGamma` |
| `service_classes` | Optional number of services / tenants |
| `size_model` | Distribution of hidden job sizes |

### Arrival Models

#### TraceReplay
Replay timestamps directly from BurstGPT traces.

#### ModeledGamma

Generate inter-arrival times using Gamma distributions:


inter_arrival ~ Gamma(α(t), β(t))


Parameters can change every **20-minute window** (as observed in BurstGPT).

### Hidden Request Size (Optional)

To simulate LLM workloads:


prompt_tokens ~ Zipf(s, xmin)
gen_tokens ~ Zipf / LogNormal


The hidden job size determines backend service time but **is not visible to the load balancer**.

## Output

Each arrival generates:


Request(
rid,
t_arrival,
class_id,
job_size
)


---

# 3. LoadBalancer / Dispatcher

## Purpose

Dispatch incoming requests to backend workers using **only latency feedback**.

The load balancer does **not observe**:

- token length
- KV cache
- queue lengths inside the worker

It only observes:

- request completion latency
- optional inflight request counts

## State Maintained per Worker

| Variable | Meaning |
|---|---|
| `lat_ewma[i]` | EWMA latency estimate |
| `inflight[i]` | number of outstanding requests |
| `penalty[i]` | optional temporary slowdown penalty |

## Policy Interface


choose_worker(request, state) → worker_id


## Baseline Policies

### Random


worker = random(worker_pool)


### Round Robin


worker = next(worker_pool)


### Least Inflight


worker = argmin_i inflight[i]


### Peak-EWMA Style

Inspired by Envoy:


score_i = lat_ewma[i] * (1 + inflight[i])
worker = argmin(score_i)


### Latency-Only Policy (Your Contribution)

A policy based purely on observed latency signals.

## Feedback Update

When a request completes:


latency = t_done - t_arrival

lat_ewma[i] =
(1 - γ) * lat_ewma[i] + γ * latency

inflight[i] -= 1


---

# 4. InferencePool

## Purpose

Represents a shared pool of backend workers (e.g., GPU replicas).

Each worker processes requests and introduces **queueing delay and contention**.


InferencePool
├── Worker 1
├── Worker 2
├── ...
└── Worker M


---

# 5. Worker Model

Each worker maintains a **local request queue**.

Two possible models are supported.

---

## Option A: FCFS Single-Server Queue

Simplest model.


worker_i:
queue_i
processing_request


Process:

1. Request arrives → enqueue
2. If worker idle → start service
3. When service finishes → start next request

---

## Option B: Processor Sharing (LLM-style)

Worker has capacity:


μ tokens / second


Multiple active requests share the capacity simultaneously.

This approximates **token generation interleaving** in LLM serving.

---

# 6. Service Time Model

Service time is **not visible to the load balancer**.

It depends on hidden request size and system contention.

General model:


S = S_base(z) * h(n_i(t)) * g(N_global(t)) + ε


Where:

| Term | Meaning |
|---|---|
| `z` | hidden request size |
| `n_i(t)` | local queue length at worker i |
| `N_global(t)` | total inflight requests across pool |
| `ε` | noise |

Example instantiation:


S_base(z) = a + b * z
h(n) = 1 + c * n
g(N) = 1 + d * max(0, N - N0)
ε ~ LogNormal(0, σ)


Interpretation:

- Larger prompts take longer
- Local queue increases delay
- Global contention introduces cross-worker slowdown

---

# 7. Multi-Service / Multi-LB Extension

To simulate **multiple services competing for the same backend**:


TrafficGenerator_k
↓
LoadBalancer_k
↓
Shared InferencePool


Properties:

- Each service has its own dispatcher
- Each dispatcher observes only its own request latency
- Backend contention affects all services

---

# 8. Metrics Collector

Tracks system performance.

## Latency Metrics

- Mean latency
- Median latency
- P95 latency
- P99 latency

## Throughput


completed_requests / simulation_time


## Resource Metrics

- Worker utilization
- Queue length distribution
- Global inflight requests

## Fairness (optional)

Latency across service classes.

---

# 9. Simulation Engine

The simulator is implemented as a **discrete-event simulation**.

## Event Types

| Event | Description |
|---|---|
| `ARRIVAL` | request enters system |
| `DISPATCH` | LB selects worker |
| `SERVICE_START` | worker begins processing |
| `SERVICE_DONE` | request completes |

## Main Loop


while event_queue not empty:

event = pop_next_event()

update_system_state()

generate_new_events()

Events are processed in chronological order.

---

# 10. Suggested Default Parameters

| Parameter | Example |
|---|---|
| workers | 8–64 |
| Zipf exponent | 1.1–1.5 |
| Gamma burst windows | 20 minutes |
| EWMA γ | 0.05–0.2 |
| contention threshold N0 | depends on pool size |

---

# 11. Key Modeling Principles

1. The **load balancer is black-box** and sees only latency.
2. Backend service time depends on **hidden job size and contention**.
3. BurstGPT-like traffic introduces **burstiness and heavy-tail job sizes**.
4. Multiple dispatchers can compete for the same backend pool.

This allows evaluation of **latency-feedback load balancing policies under realistic LLM-like workloads
