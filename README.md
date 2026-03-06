# ⚡ LB Simulation (Event-Driven with SimPy)

Mô phỏng **load balancing dựa trên latency feedback** cho workload kiểu LLM, theo thiết kế trong `design.md`.

## 🎯 Mục tiêu
- Đánh giá policy LB khi **không biết hidden job size**.
- Backend service time phụ thuộc:
  - job size
  - local queue contention
  - global contention
- Hỗ trợ traffic bursty và trace replay.

## 🧱 Kiến trúc nhanh
`TrafficGenerator -> LoadBalancer -> InferencePool (workers) -> Completion -> Controller`

## 📦 Cấu trúc mã nguồn
```text
lb_simulation/
  __init__.py
  __main__.py
  models.py
  utils.py
  metrics.py
  controller.py
  latency_redirect_policies.py
  load_balancer.py
  lb_policies.py
  inference_pool.py
  workers.py
  worker_models.py
  traffic.py
  request_csv_logger.py
  runner.py
configs/
  service_classes.example.json
  service_classes_2class.example.json
  worker_classes.example.json
  worker_classes_heterogeneous.example.json
  controller_wrr_inverse.example.json
  controller_track_all.example.json
tools/
  plot_full_log.py
  requirements.txt
  README.md
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
### Bắt buộc truyền file config service class + worker class
```bash
python3 simulator.py \
  --service-class-config configs/service_classes_2class.example.json \
  --worker-class-config configs/worker_classes.example.json
```

### Chọn policy / thời gian mô phỏng
```bash
python3 simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes_heterogeneous.example.json \
  --t-end 2h \
  --policy latency_only
```

## 🧠 Chính sách load balancer hỗ trợ
- `random`
- `round_robin`
- `weighted_round_robin`
- `least_inflight`
- `peak_ewma`
- `latency_only` (EWMA + exploration nhẹ)

Mỗi policy được module hóa trong `lb_simulation/lb_policies.py` qua cơ chế registry.
Để thêm policy mới, chỉ cần:
1. Tạo class kế thừa `LoadBalancingPolicy`
2. Gắn `@register_policy`
3. Đặt `name` duy nhất

CLI `--policy` sẽ tự nhận policy mới mà không cần sửa `runner.py`.

`--t-end` hỗ trợ cả đơn vị thời gian: `300`, `90s`, `1m`, `2h`, `3d`.

## 🎛️ Controller
`Controller` là module riêng để:
- Điều khiển tham số policy (ví dụ: `worker_weights` của `weighted_round_robin`)
- Theo dõi latency theo kiểu sampling một phần traffic

Mặc định (không truyền `--controller-config`) controller ở chế độ no-op:
- Không điều khiển tham số policy

Latency tracking đã tách khỏi Load Balancer:
- LB không tự học latency từ toàn bộ completion.
- Với policy cần latency (`peak_ewma`, `latency_only`), controller tự bật latency tracker.
- Latency tracker được mô hình như một worker đặc biệt: service time = 0, rồi forward request tới worker thật.
- Controller gửi cho LB policy redirect để quyết định tỉ lệ request đi qua latency tracker.
- Redirect policy có thể điều khiển cách forward (`round_robin` hoặc forward về đúng worker LB đã chọn).
- Chỉ request đi qua latency tracker mới được dùng để cập nhật latency estimate.

## 📊 Metrics đầu ra
- Mean / Median / P95 / P99 latency
- Throughput
- Avg queue length
- Avg global inflight
- Worker utilization
- Latency theo class (nếu dùng nhiều class)

## 📝 Full request log (CSV)
Mỗi lần chạy sẽ tự tạo một thư mục con trong `./logs` theo dạng:
- `logs/run-YYYYMMDD-HHMMSS/`

Artifacts luôn có trong mỗi run:
- `run_config.json` (tham số lần chạy)
- `service_class_config.json` (snapshot config đầu vào)
- `worker_class_config.json` (snapshot config worker)
- `summary.json` (kết quả tổng hợp)

Khi bật `--full-log`, mọi request hoàn thành sẽ được ghi vào:
- `request_full_log.csv` trong chính thư mục run đó.

```bash
python3 simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes.example.json \
  --controller-config configs/controller_wrr_inverse.example.json \
  --policy weighted_round_robin \
  --t-end 30m
```

Hoặc chạy không controller config (mặc định no-op):
```bash
python3 simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes.example.json \
  --full-log
```

CSV columns:
- `rid`, `class_id`, `worker_id`, `worker_class_id`, `worker_service_model`
- `job_size`, `model`, `log_type`
- `t_arrival`, `t_start`, `t_done`
- `queue_len_on_dispatch`, `n_local_at_start`, `n_global_at_start`
- `lb_selected_worker_id`, `routed_via_latency_tracker`
- `latency_tracked` (request này có được sample vào latency tracker hay không)
- `service_time`, `latency`

## 🧩 Service class config (JSON)
Khi dùng `--service-class-config`, mỗi class có thể tự chọn:
- `trace_replay`: trỏ tới `trace_file`
- `modeled_gamma`: dùng `gamma_windows` hoặc `gamma` cố định
- Có thể set mặc định `model` và `log_type` cho class (đặc biệt hữu ích với `modeled_gamma`)

### Trace CSV schema (trace_replay)
File trace được đọc theo header:
- `Timestamp`
- `Session ID` (không bắt buộc cho simulator)
- `Elapsed time` (không bắt buộc cho simulator)
- `Model`
- `Request tokens`
- `Response tokens`
- `Total tokens`
- `Log Type`

Mapping trong simulator:
- `job_size = Total tokens`
- `model = Model`
- `log_type = Log Type`

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
      "model": "GPT-4",
      "log_type": "Conversation log",
      "trace_file": "traces/class0.csv",
      "zipf": { "s": 1.2, "xmin": 16, "max": 2048 }
    },
    {
      "class_id": 1,
      "arrival_mode": "modeled_gamma",
      "model": "ChatGPT",
      "log_type": "Conversation log",
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

## 🧩 Worker class config (JSON)
Worker được khai báo theo class, mỗi class có:
- `count`: số worker trong class
- `service_model`: mô hình phục vụ (service-time model)
- `params`: tham số cho mô hình đó

Service models built-in:
- `contention_lognormal`: `S = (a+b*z)*(1+c*n_local)*(1+d*max(0,N-n0))*LogNormal(0,sigma)`
- `linear_lognormal`: `S = (a+b*z)*LogNormal(0,sigma)`

Ví dụ:
```json
{
  "classes": [
    {
      "class_id": 0,
      "count": 6,
      "service_model": "contention_lognormal",
      "params": { "a": 0.03, "b": 0.002, "c": 0.12, "d": 0.015, "sigma": 0.2 }
    },
    {
      "class_id": 1,
      "count": 2,
      "service_model": "linear_lognormal",
      "params": { "a": 0.04, "b": 0.003, "sigma": 0.1 }
    }
  ]
}
```

Ghi chú:
- `--worker-class-config` là option bắt buộc.

## 🧩 Controller config (JSON)
`--controller-config` là option optional.

Ví dụ:
```json
{
  "mode": "none",
  "latency_tracker": {
    "enabled": true,
    "init_estimate": 0.5,
    "ewma_gamma": 0.1,
    "redirect_policy": {
      "name": "fixed_rate",
      "params": {
        "rate": 0.05
      }
    }
  },
  "wrr": {
    "mode": "inverse_latency",
    "update_every_samples": 20,
    "inverse_power": 1.0,
    "min_weight": 0.2,
    "max_weight": 5.0
  }
}
```

Ghi chú:
- `wrr.weights` có thể set static weights ban đầu (độ dài phải đúng số worker).
- `latency_tracker.ewma_gamma` là hệ số EWMA để ước lượng latency từ sample.
- `latency_tracker.redirect_policy` hiện có:
  - `fixed_rate`: redirect theo tỉ lệ `rate`, tracker forward theo round-robin.
  - `track_all`: quyết định worker trên LB thật trước, sau đó 100% request đi qua tracker và tracker forward xuống đúng worker LB đã chọn.

## 🧪 Dev quick check
```bash
python3 -m py_compile simulator.py lb_simulation/*.py
```

## 🧰 Tools ngoài simulator
`tools/` chứa utility không thuộc luồng mô phỏng chính.

Ví dụ tool vẽ từ full log CSV:
```bash
pip install -r tools/requirements.txt
python3 tools/plot_full_log.py \
  --full-log-csv logs/run-YYYYMMDD-HHMMSS/request_full_log.csv
```

Output:
- `requests_over_time_total.png`
- `requests_over_time_by_service_class.png`
- `latency_histogram_total.png`
- `latency_histogram_by_service_class.png`
