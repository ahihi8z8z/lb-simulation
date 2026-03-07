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
