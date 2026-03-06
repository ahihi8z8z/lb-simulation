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
| Biến | Ý nghĩa |
|---|---|
| `num_workers` | Số worker thật (không tính tracker). |
| `lat_ewma` | Ước lượng latency theo worker để policy latency-aware sử dụng. |
| `inflight` | Số request đang xử lý theo worker. |
| `feedback_count` | Số sample latency đã nhận theo worker. |
| `worker_weights` | Trọng số runtime cho WRR; luôn được normalize tổng = 1. |
| `latency_tracker_worker_id` | Worker id ảo đại diện latency tracker (nếu bật). |
| `_redirect_target_by_rid` | Map `request_id -> selected_worker` khi request bị redirect qua tracker. |

### `LoadBalancerController`
| Biến | Ý nghĩa |
|---|---|
| `policy` | Policy LB đang chạy (lowercase). |
| `tracker_worker_id` | `num_workers` (id kế sau worker thật), dành cho tracker. |
| `latency_tracker_enabled` | Bật/tắt tracker theo config đã validate. |
| `latency_tracker` | Instance `LatencyTrackerWorker` hoặc `None`. |
| `lb_control_module` | Module điều khiển LB đang dùng (`none` hoặc `wrr_lp_latency`). |

### `LatencyTrackerWorker`
| Biến | Ý nghĩa |
|---|---|
| `ewma_gamma` | Hệ số EWMA khi cập nhật latency estimate. |
| `estimates` | Vector latency estimate theo worker. |
| `sample_counts` | Số sample theo worker. |
| `redirect_policy` | Chính sách quyết định request nào đi qua tracker. |
| `forward_mode` | Cách forward từ tracker sang worker thật (`round_robin` hoặc `selected_worker`). |
| `redirect_decisions` | Số lần đã ra quyết định redirect. |
| `redirected_requests` | Số request thực sự bị redirect. |

### `WrrLpLatencyControlModule`
| Biến | Ý nghĩa |
|---|---|
| `params` | Bộ tham số LP/weight update (`WrrLpControlParams`). |
| `class_latency_estimates` | Ma trận EWMA latency theo `class_id x worker`. |
| `class_latency_samples` | Số mẫu tương ứng theo `class_id x worker`. |
| `class_completions_window` | Demand theo class trong cửa sổ update hiện tại. |
| `next_update_time` | Mốc thời gian mô phỏng lần update kế tiếp. |
| `last_weights` | Weights vòng trước (dùng cho EMA decay khi bật). |
| `lp_updates` | Số lần LP update thành công. |

### `InferencePool`
| Biến | Ý nghĩa |
|---|---|
| `resources` | Danh sách `simpy.Resource` (mỗi worker capacity=1). |
| `global_inflight` | Số request đang trong pool toàn cục. |
| `on_complete` | Callback completion để controller update state/control. |
| `on_request_done` | Callback ghi CSV detail metrics (nếu bật `--detail`). |

### `TrafficGenerator`
| Biến | Ý nghĩa |
|---|---|
| `arrival_mode` | `trace_replay` hoặc `modeled_gamma`. |
| `trace_records` | Dữ liệu trace đã parse/lọc. |
| `gamma_windows` | Danh sách `(window_end, alpha, beta)` cho inter-arrival gamma. |
| `trace_traffic_scale` | Hệ số nhân số request ở cùng timestamp trace. |
| `fixed_class_id` | Class id cố định khi generator phục vụ một service class cụ thể. |

## Cấu hình điều khiển và tác dụng

### `controller.latency_tracker`
| Key | Tác dụng |
|---|---|
| `enabled` | Bật tracker worker và cơ chế redirect sampling. |
| `init_estimate` | Giá trị EWMA ban đầu cho mọi worker. |
| `ewma_gamma` | Mức phản ứng của estimate theo sample mới. |
| `redirect_policy` | Chọn chính sách sampling (`fixed_rate`, `track_all`, ...). |

### `controller.wrr`
| Key | Tác dụng |
|---|---|
| `mode` | `none` hoặc `lp_latency`. |
| `weights` | Weight ban đầu (nếu có), LB sẽ normalize về tổng 1. |
| `update_interval_seconds` | Chu kỳ cập nhật weight theo thời gian mô phỏng. |
| `lp_balance_tolerance` | Độ cho phép lệch tải mỗi worker quanh target trung bình. |
| `lp_ewma_gamma` | Gamma EWMA dùng trong bảng latency `class x worker` của module LP. |
| `lp_weight_ema_decay` | Làm mượt weights mới theo EMA với weights cũ. |
| `lp_use_tracked_only` | Nếu `true`, chỉ dùng completion đã tracked để học latency LP. |
| `min_weight`, `max_weight` | Chặn biên độ weight trước khi normalize cuối cùng. |
