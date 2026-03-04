# ⚡ LB Simulation (Event-Driven with SimPy)

Mô phỏng **load balancing dựa trên latency feedback** cho workload kiểu LLM, theo thiết kế trong `design.md`.

## 🎯 Mục tiêu
- Đánh giá policy LB khi **không biết hidden request size**.
- Backend service time phụ thuộc:
  - hidden size
  - local queue contention
  - global contention
- Hỗ trợ traffic bursty và trace replay.

## 🧱 Kiến trúc nhanh
`TrafficGenerator -> LoadBalancer -> InferencePool (workers) -> Completion -> Latency feedback`

## 📦 Cấu trúc mã nguồn
```text
lb_simulation/
  __init__.py
  __main__.py
  models.py
  utils.py
  metrics.py
  load_balancer.py
  inference_pool.py
  traffic.py
  request_csv_logger.py
  runner.py
simulator.py               # CLI entrypoint mỏng
design.md
requirements.txt
```

## 🚀 Cài đặt
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Cách chạy
### Chạy mặc định
```bash
python3 simulator.py
```

### Chọn policy / thời gian / số worker
```bash
python3 simulator.py \
  --t-end 1800 \
  --workers 16 \
  --policy latency_only
```

### Trace replay mode
```bash
python3 simulator.py \
  --arrival-mode trace_replay \
  --trace-file ./trace.csv
```

## 🧠 Chính sách load balancer hỗ trợ
- `random`
- `round_robin`
- `least_inflight`
- `peak_ewma`
- `latency_only` (EWMA + exploration nhẹ)

## 📊 Metrics đầu ra
- Mean / Median / P95 / P99 latency
- Throughput
- Avg queue length
- Avg global inflight
- Worker utilization
- Latency theo class (nếu dùng nhiều class)

## 📝 Full request log (CSV)
Khi bật `--full-log`, mọi request hoàn thành sẽ được ghi ngay ra CSV.

```bash
python3 simulator.py --full-log
python3 simulator.py --full-log --full-log-file logs/requests.csv
```

CSV columns:
- `rid`, `class_id`, `worker_id`, `hidden_size`
- `t_arrival`, `t_start`, `t_done`
- `queue_len_on_dispatch`, `n_local_at_start`, `n_global_at_start`
- `service_time`, `latency`

## 🧪 Dev quick check
```bash
python3 -m py_compile simulator.py lb_simulation/*.py
```
