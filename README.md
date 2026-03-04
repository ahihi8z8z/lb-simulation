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
configs/
  service_classes.example.json
  service_classes_2class.example.json
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
### Bắt buộc truyền file config service class
```bash
python3 simulator.py \
  --service-class-config configs/service_classes_2class.example.json
```

### Chọn policy / thời gian / số worker
```bash
python3 simulator.py \
  --service-class-config configs/service_classes.example.json \
  --t-end 1800 \
  --workers 16 \
  --policy latency_only
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
python3 simulator.py \
  --service-class-config configs/service_classes.example.json \
  --full-log

python3 simulator.py \
  --service-class-config configs/service_classes.example.json \
  --full-log --full-log-file logs/requests.csv
```

CSV columns:
- `rid`, `class_id`, `worker_id`, `hidden_size`
- `t_arrival`, `t_start`, `t_done`
- `queue_len_on_dispatch`, `n_local_at_start`, `n_global_at_start`
- `service_time`, `latency`

## 🧩 Service class config (JSON)
Khi dùng `--service-class-config`, mỗi class có thể tự chọn:
- `trace_replay`: trỏ tới `trace_file`
- `modeled_gamma`: dùng `gamma_windows` hoặc `gamma` cố định

File config hỗ trợ 2 dạng:
- object có key `classes`
- hoặc list trực tiếp các class

Ví dụ:
```json
{
  "classes": [
    {
      "class_id": 0,
      "arrival_mode": "trace_replay",
      "trace_file": "traces/class0.csv",
      "zipf": { "s": 1.2, "xmin": 16, "max": 2048 }
    },
    {
      "class_id": 1,
      "arrival_mode": "modeled_gamma",
      "gamma_windows": [
        { "window_end": 1200, "alpha": 2.5, "beta": 0.3 },
        { "window_end": 2400, "alpha": 2.0, "beta": 0.65 }
      ]
    },
    {
      "class_id": 2,
      "arrival_mode": "modeled_gamma",
      "gamma": { "alpha": 2.3, "beta": 0.4, "window_size": 1200 }
    }
  ]
}
```

Ghi chú:
- `trace_file` có thể là path tuyệt đối hoặc path tương đối theo thư mục chứa file config.
- `--service-class-config` là option bắt buộc.

## 🧪 Dev quick check
```bash
python3 -m py_compile simulator.py lb_simulation/*.py
```
