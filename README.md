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
- Flow chart request: `docs/request_flow_chart.md`
- Biểu đồ luồng chi tiết: `docs/flow_diagrams.md`
- Design chi tiết: `docs/design.md`
- Tác dụng biến và class: `docs/variables_and_classes.md`

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

Tài liệu kiến trúc controller và luồng chi tiết nằm trong thư mục `docs/`.

Ghi chú vận hành quan trọng:
- Mặc định (không truyền `--controller-config`) controller ở chế độ no-op.
- Với policy cần latency (`peak_ewma`, `latency_only`) bắt buộc truyền `--controller-config` và set `latency_tracker.enabled=true`.
- Với `weighted_round_robin` khi `wrr.mode = lp_latency`, cũng bắt buộc `latency_tracker.enabled=true`.

## 📊 Metrics đầu ra
- Mean / Median / P95 / P99 latency
- Throughput
- Avg queue length
- Avg global inflight
- Worker utilization
- Latency theo class (nếu dùng nhiều class)

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
- `fixed_linear`: mô hình throughput thay đổi tuyến tính theo `job_size` (service time được suy ra từ throughput này).

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
    "update_interval_seconds": 60.0,
    "min_weight": 0.2,
    "max_weight": 5.0,
    "lp_balance_tolerance": 0.25,
    "lp_ewma_gamma": 0.1,
    "lp_weight_ema_decay": 0.2,
    "lp_use_tracked_only": false
  }
}
```

Ghi chú:
- `wrr.weights` có thể set static weights ban đầu (độ dài phải đúng số worker).
- `wrr.weights` luôn được chuẩn hóa để tổng bằng `1.0` khi apply vào Load Balancer.
- `wrr.mode` hỗ trợ: `none`, `lp_latency`.
- `wrr.mode = lp_latency`: map sang load-balancer-control module `wrr_lp_latency`; module này ước lượng latency theo `class_id x worker`, giải LP để phân bổ tải theo class, rồi chuyển thành `worker_weights` cho `weighted_round_robin`.
- `wrr.update_interval_seconds`: chu kỳ cập nhật weight theo thời gian mô phỏng (ví dụ `60.0` nghĩa là mỗi 1 phút mô phỏng cập nhật một lần).
- `wrr.mode = lp_latency` bắt buộc cần `scipy` (dùng `scipy.optimize.linprog`), không có fallback heuristic.
- `wrr.mode = lp_latency` bắt buộc cần `latency_tracker.enabled=true`.
- `wrr.lp_balance_tolerance` điều khiển biên độ cân bằng tải mỗi worker quanh mức trung bình.
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
