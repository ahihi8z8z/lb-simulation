# Tổng Quan Kiến Trúc

## Pipeline chính
`TrafficGenerator -> LoadBalancer -> InferencePool -> Completion Callback -> Controller`

## Thành phần và vai trò
- `lb_simulation/traffic.py`
  - Sinh request theo `trace_replay` hoặc `modeled_gamma`.
  - Đẩy request sang callback `on_arrival`.
- `lb_simulation/load_balancer.py`
  - Giữ state LB runtime: `inflight`, `lat_ewma`, `worker_weights`, ...
  - Gọi policy tương ứng trong `lb_policies.py` để chọn worker.
  - Có cơ chế redirect request qua latency tracker.
- `lb_simulation/inference_pool.py`
  - Mô hình pool worker (SimPy Resource capacity=1/worker).
  - Tính service time theo `worker_models.py`.
  - Ghi metrics, gọi callback completion.
- `lb_simulation/controller.py`
  - Lắp 2 module:
  - `latency_tracker`: thu mẫu latency, EWMA.
  - `lb_control_module`: điều khiển tham số LB (ví dụ WRR LP latency).
- `lb_simulation/lb_control_modules.py`
  - Module điều khiển LB có thể cắm thêm bằng registry.
  - Hiện có `none` và `wrr_lp_latency`.

## Kiến trúc module hóa
- Policy LB: registry trong `lb_policies.py` (`@register_policy`).
- Redirect policy cho latency tracker: registry trong `latency_redirect_policies.py`.
- LB control module: registry trong `lb_control_modules.py`.
- Worker service model: registry trong `worker_models.py`.

## Invariants quan trọng
- `LoadBalancer.worker_weights` luôn được chuẩn hóa tổng bằng `1.0`.
- `wrr.mode=lp_latency` yêu cầu:
  - policy LB là `weighted_round_robin`.
  - `latency_tracker.enabled=true`.
  - có `scipy.optimize.linprog`.
- Policy cần latency (`latency_only`) bắt buộc bật latency tracker qua `controller-config`.

## Queueing model
- Mô tả chi tiết dưới góc nhìn hàng đợi nằm trong `queueing_model.md`.
- Ở mức kiến trúc: hệ thống là nhiều queue song song (`G/G/1` mỗi worker) với router động và vòng feedback từ controller.
