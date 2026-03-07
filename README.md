# ⚡ LB Simulation (Event-Driven with SimPy)

Mô phỏng **load balancing dựa trên latency feedback** cho workload kiểu LLM.

## 🎯 Mục tiêu
- Đánh giá policy LB khi **không biết hidden job size**.
- Backend service time phụ thuộc:
  - job size
  - local queue contention
  - global contention
- Hỗ trợ traffic bursty và trace replay.

## 📚 Tài liệu
- Tài liệu tổng hợp: `docs/README.md`
- Tổng quan kiến trúc: `docs/architecture_overview.md`
- Queueing model view: `docs/queueing_model.md`
- Flow chart request: `docs/request_flow_chart.md`
- Biểu đồ luồng chi tiết: `docs/flow_diagrams.md`
- Design chi tiết: `docs/design.md`
- Tác dụng biến và class: `docs/variables_and_classes.md`
- Thuật toán theo nhóm: `docs/algorithms/README.md`

## 🚀 Cài đặt
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Cách chạy
### Chạy nhanh (copy là chạy được)
```bash
.venv/bin/python simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes.example.json \
  --t-end 30s \
  --policy round_robin
```

### Chạy với controller config mẫu
```bash
.venv/bin/python simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes.example.json \
  --controller-config configs/controller.example.json \
  --policy weighted_round_robin \
  --t-end 30m \
  --detail
```

Ghi chú:
- `configs/` chỉ giữ một file mẫu cho mỗi loại config: service class, worker class, controller.

## 🧠 Chính sách load balancer hỗ trợ
- `random`
- `round_robin`
- `weighted_round_robin`
- `least_inflight`
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
- Quản lý nhiều load balancer theo class: mỗi `service class` có một `LoadBalancer` riêng
- Khi bật tracker: mỗi class có một `LatencyTrackerWorker` riêng

Tài liệu kiến trúc controller và luồng chi tiết nằm trong thư mục `docs/`.

Ghi chú vận hành quan trọng:
- Mặc định (không truyền `--controller-config`) controller ở chế độ no-op.
- Với policy cần latency (`latency_only`) bắt buộc truyền `--controller-config` và set `latency_tracker.enabled=true`.
- Với `weighted_round_robin` khi `wrr.mode = lp_latency` hoặc `separate_lp`, cũng bắt buộc `latency_tracker.enabled=true`.

## 📊 Metrics đầu ra
- Mean / Median / P95 / P99 latency
- Throughput
- Avg queue length
- Avg global inflight
- Worker utilization
- Latency theo class (nếu dùng nhiều class)
- Latency theo worker

## 📝 Request Detail Metrics (CSV)
Mỗi lần chạy sẽ tự tạo một thư mục con trong `./logs` theo dạng:
- `logs/run-YYYYMMDD-HHMMSS/`

Artifacts luôn có trong mỗi run:
- `run_config.json` (tham số lần chạy)
- `service_class_config.json` (snapshot config đầu vào)
- `worker_class_config.json` (snapshot config worker)
- `summary.json` (kết quả tổng hợp)

Khi bật `--detail`, mọi request hoàn thành sẽ được ghi vào:
- `request_detail_metrics.csv` trong chính thư mục run đó.

```bash
.venv/bin/python simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes.example.json \
  --controller-config configs/controller.example.json \
  --policy weighted_round_robin \
  --t-end 30m
```

Hoặc chạy không controller config (mặc định no-op):
```bash
.venv/bin/python simulator.py \
  --service-class-config configs/service_classes.example.json \
  --worker-class-config configs/worker_classes.example.json \
  --detail
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
- `description`: mô tả class này mô hình hóa workload nào (optional)
- `trace_replay`: trỏ tới `trace_file`, `job_size` lấy trực tiếp từ cột `Total tokens`
- `modeled_gamma`: dùng `gamma_windows` hoặc `gamma` cố định cho inter-arrival
- Có thể set mặc định `model` và `log_type` cho class (đặc biệt hữu ích với `modeled_gamma`)
- Với `trace_replay`, có thể set `traffic_scale` (số nguyên dương, mặc định `1`) để nhân số request tại cùng timestamp.
  Ví dụ: một record tại `t=5s` và `traffic_scale=3` sẽ sinh 3 request cùng lúc tại `5s`.
- Với `modeled_gamma`, `job_size` được sinh theo:
  - `request_length ~ Zipf(s, xmin, max)` từ cấu hình `zipf`
  - `response_length = slope * request_length + intercept` từ `response_linear`
  - `job_size = request_length + response_length`

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
- Với `trace_replay`, thời gian mô phỏng được chuẩn hóa theo mốc tương đối:
  request đầu tiên của trace (sau khi lọc theo `model`/`log_type`) được xem là `t=0`.

File config hỗ trợ 2 dạng:
- object có key `classes`
- hoặc list trực tiếp các class

Ví dụ:
```json
{
  "classes": [
    {
      "class_id": 0,
      "description": "Replay traffic cho ChatGPT Conversation log từ trace BurstGPT",
      "arrival_mode": "trace_replay",
      "model": "ChatGPT",
      "log_type": "Conversation log",
      "trace_file": "../traces/BurstGPT_without_fails_1.csv",
      "traffic_scale": 3
    },
    {
      "class_id": 1,
      "description": "Traffic modeled bằng gamma + Zipf cho GPT-4 API log",
      "arrival_mode": "modeled_gamma",
      "model": "GPT-4",
      "log_type": "API log",
      "gamma_windows": [
        { "window_end": 300, "alpha": 1.8, "beta": 0.9 },
        { "window_end": 900, "alpha": 2.4, "beta": 0.45 },
        { "window_end": 1800, "alpha": 1.6, "beta": 1.1 }
      ],
      "zipf": { "s": 1.3, "xmin": 32, "max": 4096 },
      "response_linear": { "slope": 0.7, "intercept": 32.0 }
    }
  ]
}
```

Ghi chú:
- `trace_file` có thể là path tuyệt đối hoặc path tương đối theo thư mục chứa file config.
- `trace_replay` không nhận cấu hình `zipf`/`response_linear`.
- `modeled_gamma` không nhận cấu hình `traffic_scale`.
- `--service-class-config` là option bắt buộc.

## 🧩 Worker class config (JSON)
Worker được khai báo theo class, mỗi class có:
- `description`: mô tả class worker này mô hình hóa loại tài nguyên/phần cứng nào (optional)
- `count`: số worker trong class
- `service_model`: mô hình phục vụ (service-time model)
- `params`: tham số cho mô hình đó

Service models built-in:
- `contention_lognormal`: `S = (a+b*z)*(1+c*n_local)*(1+d*max(0,N-n0))*LogNormal(0,sigma)`
- `linear_lognormal`: `S = (a+b*z)*LogNormal(0,sigma)`
- `fixed`: `S = service_time` (hằng số)
- `fixed_linear`: `S = clip(a + b*job_size, min, max)` (service time tuyến tính theo `job_size` có chặn biên).

Ví dụ:
```json
{
  "classes": [
    {
      "class_id": 0,
      "description": "Worker nặng, chịu ảnh hưởng contention local/global",
      "count": 6,
      "service_model": "contention_lognormal",
      "params": { "a": 0.03, "b": 0.002, "c": 0.12, "d": 0.015, "sigma": 0.2 }
    },
    {
      "class_id": 1,
      "description": "Worker tuyến tính cho profile latency ổn định hơn",
      "count": 2,
      "service_model": "linear_lognormal",
      "params": { "a": 0.04, "b": 0.003, "sigma": 0.1 }
    },
    {
      "class_id": 2,
      "description": "Worker fixed service time để làm baseline",
      "count": 1,
      "service_model": "fixed",
      "params": { "service_time": 0.08 }
    },
    {
      "class_id": 3,
      "description": "Worker fixed_linear với service time tuyến tính theo job_size",
      "count": 1,
      "service_model": "fixed_linear",
      "params": { "a": 0.03, "b": 0.002, "min": 0.001, "max": 5.0 }
    }
  ]
}
```

Ghi chú:
- `--worker-class-config` là option bắt buộc.

## 🧩 Controller config (JSON)
`--controller-config` là option optional.
Nhưng với policy cần latency thì đây là option bắt buộc.

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
    "mode": "lp_latency",
    "weights": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    "update_interval_seconds": 60.0,
    "min_weight": 0.2,
    "max_weight": 5.0,
    "lp_balance_tolerance": 0.25,
    "lp_ewma_gamma": 0.1
  }
}
```

Ghi chú:
- `wrr.weights` có thể set static weights ban đầu (độ dài phải đúng số worker).
- `wrr.weights` luôn được chuẩn hóa để tổng bằng `1.0` khi apply vào Load Balancer.
- `wrr.weights` phải có mọi phần tử `> 0`.
- `wrr.mode` hỗ trợ: `none`, `lp_latency`, `separate_lp`.
- `wrr.mode = lp_latency`: map sang load-balancer-control module `wrr_lp_latency`; module này dùng latency estimate từ `latency_tracker` để giải LP cho ma trận phân bổ `class x worker`, rồi apply mỗi hàng thành `worker_weights` cho LB tương ứng của từng class.
- `wrr.mode = separate_lp`: map sang `wrr_separate_lp_latency`; module này giải LP riêng cho từng class để tối ưu mean latency của chính class đó.
- `wrr.update_interval_seconds`: chu kỳ cập nhật weight theo thời gian mô phỏng (ví dụ `60.0` nghĩa là mỗi 1 phút mô phỏng cập nhật một lần).
- `wrr.mode = lp_latency` và `wrr.mode = separate_lp` bắt buộc cần `scipy` (dùng `scipy.optimize.linprog`), không có fallback heuristic.
- `wrr.mode = lp_latency` và `wrr.mode = separate_lp` bắt buộc cần `latency_tracker.enabled=true`.
- `wrr.lp_balance_tolerance` điều khiển biên độ cân bằng tải mỗi worker quanh mức trung bình (chỉ áp dụng cho `wrr.mode = lp_latency`).
- `wrr.lp_ewma_gamma` là hệ số làm mượt weight: `new_weight = (1-gamma)*weight_cũ + gamma*weight_từ_LP`.
- Nếu `wrr.mode = none`, `wrr.weights` là bộ trọng số cố định suốt runtime.
- Nếu `wrr.mode = lp_latency`, `wrr.weights` là trọng số khởi tạo; sau đó module control sẽ cập nhật lại theo chu kỳ `wrr.update_interval_seconds`.
- Nếu `wrr.mode = separate_lp`, `wrr.weights` cũng chỉ là trọng số khởi tạo; sau đó mỗi class sẽ được cập nhật weight riêng theo LP của class đó.
- Ví dụ `2 class` và `3 worker`:
  - Trong file config, chỉ khai báo 1 vector cho worker: `"weights": [0.2, 0.3, 0.5]`.
  - Vì mỗi class có 1 LB riêng, lúc khởi tạo cả LB class 0 và LB class 1 đều nhận cùng vector `[0.2, 0.3, 0.5]`.
  - Nếu `wrr.mode = none`: cả 2 LB giữ nguyên vector này.
  - Nếu `wrr.mode = lp_latency`: LP sẽ sinh ma trận theo class x worker, ví dụ:
    - class 0: `[0.10, 0.30, 0.60]`
    - class 1: `[0.55, 0.35, 0.10]`
  - Nếu `wrr.mode = separate_lp`: LP cũng sinh ma trận `class x worker`, nhưng mỗi hàng được solve độc lập theo objective mean latency của class tương ứng.
  - Ma trận trên là trạng thái runtime do LP tối ưu, không phải format nhập trực tiếp vào `wrr.weights`.
- `latency_tracker.ewma_gamma` là hệ số EWMA để ước lượng latency từ sample.
- `latency_tracker.redirect_policy` hiện có:
  - `fixed_rate`: redirect theo tỉ lệ `rate`, tracker forward theo round-robin.
  - `track_all`: quyết định worker trên LB thật trước, sau đó 100% request đi qua tracker và tracker forward xuống đúng worker LB đã chọn.

## 🧪 Dev quick check
```bash
.venv/bin/python -m py_compile simulator.py lb_simulation/*.py
```

## 🧰 Tools ngoài simulator
`tools/` chứa utility không thuộc luồng mô phỏng chính.

Ví dụ tool vẽ từ detail metrics CSV:
```bash
pip install -r tools/requirements.txt
python3 tools/plot_detail_metrics.py \
  --detail-csv logs/run-YYYYMMDD-HHMMSS/request_detail_metrics.csv
```

Logger:
- `--logger-mode INFO` (mặc định): log mức INFO trở lên.
- `--logger-mode DEBUG`: bật thêm log DEBUG.
- Runtime log luôn được lưu vào `runtime.log` trong thư mục run.

Output:
- `requests_over_time_total.png`
- `requests_over_time_by_service_class.png`
- `latency_histogram_total.png`
- `latency_histogram_by_service_class.png`
- `latency_histogram_by_worker.png`
