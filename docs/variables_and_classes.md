# Tác Dụng Của Biến Và Class

## Class chính

| File | Class | Tác dụng |
|---|---|---|
| `lb_simulation/models.py` | `Request` | Payload request chạy xuyên suốt pipeline (arrival, class, job_size, metadata). |
| `lb_simulation/load_balancer.py` | `LoadBalancer` | Giữ state LB runtime và chọn worker theo policy. |
| `lb_simulation/lb_policies.py` | `LoadBalancingPolicy` + subclasses | Implement thuật toán chọn worker (`random`, `rr`, `wrr`, `least_inflight`, `peak_ewma`, `latency_only`). |
| `lb_simulation/inference_pool.py` | `InferencePool` | Mô phỏng phục vụ request trên pool worker SimPy và phát completion callback. |
| `lb_simulation/controller.py` | `LoadBalancerController` | Điều phối latency tracker và LB control module. |
| `lb_simulation/latency_tracker.py` | `LatencyTrackerWorker` | Sampling latency theo redirect policy, EWMA estimate theo worker. |
| `lb_simulation/lb_control_modules.py` | `LoadBalancerControlModule` + subclasses | Điều khiển tham số LB theo control loop (hiện có `none`, `wrr_lp_latency`). |
| `lb_simulation/metrics.py` | `MetricsCollector` | Thu số liệu dispatch/completion và tổng hợp KPI cuối run. |
| `lb_simulation/traffic.py` | `TrafficGenerator`, `ServiceClassTrafficSpec` | Sinh traffic theo từng class từ trace hoặc gamma model. |
| `lb_simulation/workers.py` | `WorkerClassSpec`, `WorkerSpec` | Mô tả worker class từ config và worker runtime sau khi expand. |
| `lb_simulation/worker_models.py` | `WorkerServiceModel` + subclasses | Tính service time theo model của worker. |

## Biến runtime quan trọng theo module

### `LoadBalancer`
| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| `num_workers` | `int` | Số worker thật (không tính tracker). |
| `lat_ewma` | `List[float]` | Ước lượng latency theo worker để policy latency-aware sử dụng. |
| `inflight` | `List[int]` | Số request đang xử lý theo worker. |
| `feedback_count` | `List[int]` | Số sample latency đã nhận theo worker. |
| `worker_weights` | `List[float]` | Trọng số runtime cho WRR; luôn được normalize tổng = 1. |
| `latency_tracker_worker_id` | `Optional[int]` | Worker id ảo đại diện latency tracker (nếu bật). |
| `_redirect_target_by_rid` | `Dict[int, int]` | Map `request_id -> selected_worker` khi request bị redirect qua tracker. |

### `LoadBalancerController`
| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| `policy` | `str` | Policy LB đang chạy (lowercase). |
| `tracker_worker_id` | `int` | `num_workers` (id kế sau worker thật), dành cho tracker. |
| `latency_tracker_enabled` | `bool` | Bật/tắt tracker theo config đã validate. |
| `latency_tracker` | `Optional[LatencyTrackerWorker]` | Instance `LatencyTrackerWorker` hoặc `None`. |
| `lb_control_module` | `LoadBalancerControlModule` | Module điều khiển LB đang dùng (`none` hoặc `wrr_lp_latency`). |

### `LatencyTrackerWorker`
| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| `ewma_gamma` | `float` | Hệ số EWMA khi cập nhật latency estimate. |
| `estimates` | `List[float]` | Vector latency estimate theo worker. |
| `sample_counts` | `List[int]` | Số sample theo worker. |
| `redirect_policy` | `LatencyRedirectPolicy` | Chính sách quyết định request nào đi qua tracker. |
| `forward_mode` | `str` | Cách forward từ tracker sang worker thật (`round_robin` hoặc `selected_worker`). |
| `redirect_decisions` | `int` | Số lần đã ra quyết định redirect. |
| `redirected_requests` | `int` | Số request thực sự bị redirect. |

### `WrrLpLatencyControlModule`
| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| `params` | `WrrLpControlParams` | Bộ tham số LP/weight update (`WrrLpControlParams`). |
| `class_latency_estimates` | `Dict[int, List[float]]` | Ma trận EWMA latency theo `class_id x worker`. |
| `class_latency_samples` | `Dict[int, List[int]]` | Số mẫu tương ứng theo `class_id x worker`. |
| `class_completions_window` | `Dict[int, int]` | Demand theo class trong cửa sổ update hiện tại. |
| `next_update_time` | `float` | Mốc thời gian mô phỏng lần update kế tiếp. |
| `last_weights` | `List[float]` | Weights vòng trước (dùng cho EMA decay khi bật). |
| `lp_updates` | `int` | Số lần LP update thành công. |

### `InferencePool`
| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| `resources` | `List[simpy.Resource]` | Danh sách `simpy.Resource` (mỗi worker capacity=1). |
| `global_inflight` | `int` | Số request đang trong pool toàn cục. |
| `on_complete` | `Optional[Callable[[Request, int, float, bool], None]]` | Callback completion để controller update state/control. |
| `on_request_done` | `Optional[Callable[[Dict[str, object]], None]]` | Callback ghi CSV detail metrics (nếu bật `--detail`). |

### `TrafficGenerator`
| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| `arrival_mode` | `str` | `trace_replay` hoặc `modeled_gamma`. |
| `trace_records` | `List[TraceRecord]` | Dữ liệu trace đã parse/lọc. |
| `gamma_windows` | `List[Tuple[float, float, float]]` | Danh sách `(window_end, alpha, beta)` cho inter-arrival gamma. |
| `trace_traffic_scale` | `int` | Hệ số nhân số request ở cùng timestamp trace. |
| `fixed_class_id` | `Optional[int]` | Class id cố định khi generator phục vụ một service class cụ thể. |

## Cấu hình điều khiển và tác dụng

### `controller.latency_tracker`
| Key | Kiểu | Tác dụng |
|---|---|---|
| `enabled` | `bool` | Bật tracker worker và cơ chế redirect sampling. |
| `init_estimate` | `float` | Giá trị EWMA ban đầu cho mọi worker. |
| `ewma_gamma` | `float` | Mức phản ứng của estimate theo sample mới. |
| `redirect_policy` | `object` | Chọn chính sách sampling (`fixed_rate`, `track_all`, ...). |

### `controller.wrr`
| Key | Kiểu | Tác dụng |
|---|---|---|
| `mode` | `str` | `none` hoặc `lp_latency`. |
| `weights` | `Optional[List[float]]` | Weight ban đầu (nếu có), LB sẽ normalize về tổng 1. |
| `update_interval_seconds` | `float` | Chu kỳ cập nhật weight theo thời gian mô phỏng. |
| `lp_balance_tolerance` | `float` | Độ cho phép lệch tải mỗi worker quanh target trung bình. |
| `lp_ewma_gamma` | `float` | Gamma EWMA dùng trong bảng latency `class x worker` của module LP. |
| `lp_weight_ema_decay` | `float` | Làm mượt weights mới theo EMA với weights cũ. |
| `lp_use_tracked_only` | `bool` | Nếu `true`, chỉ dùng completion đã tracked để học latency LP. |
| `min_weight`, `max_weight` | `float` | Chặn biên độ weight trước khi normalize cuối cùng. |
