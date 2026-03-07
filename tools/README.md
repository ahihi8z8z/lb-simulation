# Tools (ngoài simulator core)

Thư mục này chứa các utility script hỗ trợ phân tích/kỹ thuật, không phải phần luồng mô phỏng chính.

## Plot từ detail metrics CSV

Script: `tools/plot_detail_metrics.py`

Vẽ 5 hình từ `request_detail_metrics.csv`:
- `requests_over_time_total.png`: số request theo thời gian (arrivals/completions, tổng)
- `requests_over_time_by_service_class.png`: số request theo thời gian, chia theo service class
- `latency_histogram_total.png`: histogram latency tổng
- `latency_histogram_by_service_class.png`: histogram latency chia theo service class
- `latency_histogram_by_worker.png`: histogram latency chia theo worker

### Cài phụ thuộc cho tools
```bash
pip install -r tools/requirements.txt
```

### Cách chạy
```bash
python3 tools/plot_detail_metrics.py \
  --detail-csv logs/run-YYYYMMDD-HHMMSS/request_detail_metrics.csv
```

Tùy chọn:
- `--output-dir`: thư mục output ảnh (mặc định: `<csv_dir>/plots`)
- `--time-bin`: kích thước bin theo giây cho biểu đồ request theo thời gian (mặc định `10`)
- `--latency-bins`: số bin cho histogram latency (mặc định `50`)
- `--dpi`: độ phân giải ảnh (mặc định `150`)

## So sánh nhiều log folder theo latency metrics

Script: `tools/plot_log_comparison.py`

Vẽ 3 biểu đồ cột so sánh giữa nhiều run:
- `latency_compare_system.png`: so sánh toàn hệ thống (mean, median, p95, p99)
- `latency_compare_by_service.png`: so sánh theo từng service class (mean, median, p95, p99)
- `latency_compare_by_worker.png`: so sánh theo từng worker (mean, median, p95, p99)

### Cách chạy
```bash
python3 tools/plot_log_comparison.py \
  --run logs/run-YYYYMMDD-HHMMSS=baseline \
  --run logs/run-YYYYMMDD-HHMMSS=candidate \
  --output-dir logs/comparison_plots
```

Hoặc để tool tự đặt label:
```bash
python3 tools/plot_log_comparison.py \
  --run logs/run-YYYYMMDD-HHMMSS \
  --run logs/run-YYYYMMDD-HHMMSS \
  --output-dir logs/comparison_plots
```

Hoặc chọn nhiều folder bằng wildcard:
```bash
python3 tools/plot_log_comparison.py \
  --run "logs/run-20260308-00*" \
  --output-dir logs/comparison_plots
```

Tùy chọn:
- `--run`: input dạng `<log_folder>` hoặc `<log_folder>=<label>`, lặp lại nhiều lần để so sánh.
  - Hỗ trợ wildcard/glob (ví dụ `logs/run*`) để chọn nhiều folder trong một `--run`.
  - Nếu không truyền label, tool tự suy ra label từ policy load balancer.
  - Riêng `weighted_round_robin`, label tự động có thêm control mode/module
    (ví dụ: `weighted_round_robin:lp_latency`, `weighted_round_robin:separate_lp`).
- `--output-dir`: thư mục output ảnh (mặc định `logs/comparison_plots`)
- `--dpi`: độ phân giải ảnh (mặc định `150`)

Ràng buộc:
- Nếu các run không có cấu hình service/worker giống nhau thì tool báo lỗi và không vẽ.
- Nếu summary của run không có đủ median/p99 theo service/worker, tool sẽ fallback đọc `request_detail_metrics.csv` (nếu có). Nếu không có detail CSV thì tool báo lỗi.

## Cắt trace theo cửa sổ thời gian

Script: `tools/extract_trace_window.py`

Ý tưởng:
- Nhập `--window` bằng đơn vị thời gian (`h`, `d`, `mo`, ...).
- Nếu không truyền `--start`: tool random một đoạn hợp lệ có đúng độ dài đó.
- Nếu truyền `--start`: tool cắt cố định từ mốc này (không random).

Ví dụ random:
```bash
python3 tools/extract_trace_window.py \
  --input traces/BurstGPT_without_fails_1.csv \
  --output traces/window_random_1h.csv \
  --window 1h \
  --seed 7
```

Ví dụ fixed start:
```bash
python3 tools/extract_trace_window.py \
  --input traces/BurstGPT_without_fails_1.csv \
  --output traces/window_fixed_2d.csv \
  --window 2d \
  --start 5d
```
