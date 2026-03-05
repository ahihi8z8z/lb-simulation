# Tools (ngoài simulator core)

Thư mục này chứa các utility script hỗ trợ phân tích/kỹ thuật, không phải phần luồng mô phỏng chính.

## Plot từ full log CSV

Script: `tools/plot_full_log.py`

Vẽ 4 hình từ `request_full_log.csv`:
- `requests_over_time_total.png`: số request theo thời gian (arrivals/completions, tổng)
- `requests_over_time_by_service_class.png`: số request theo thời gian, chia theo service class
- `latency_histogram_total.png`: histogram latency tổng
- `latency_histogram_by_service_class.png`: histogram latency chia theo service class

### Cài phụ thuộc cho tools
```bash
pip install -r tools/requirements.txt
```

### Cách chạy
```bash
python3 tools/plot_full_log.py \
  --full-log-csv logs/run-YYYYMMDD-HHMMSS/request_full_log.csv
```

Tùy chọn:
- `--output-dir`: thư mục output ảnh (mặc định: `<csv_dir>/plots`)
- `--time-bin`: kích thước bin theo giây cho biểu đồ request theo thời gian (mặc định `10`)
- `--latency-bins`: số bin cho histogram latency (mặc định `50`)
- `--dpi`: độ phân giải ảnh (mặc định `150`)
