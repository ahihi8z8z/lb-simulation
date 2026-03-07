# Thuật Toán Load Balancer Control

Nguồn implementation: `lb_simulation/lb_control_modules.py`.

`LoadBalancerControlModule` là abstraction cho control logic chạy ở controller side.

## `none`
- No-op module.
- Không thay đổi tham số LB trong runtime.
- Biến implement chính:
  - `num_workers` (kế thừa từ base class): lưu số worker, không dùng để điều khiển.
  - `initialize(...)`, `on_request_complete(...)`: phương thức no-op.

## `wrr_lp_latency`
- Mục tiêu: điều chỉnh `worker_weights` cho policy `weighted_round_robin` dựa trên latency và nhu cầu theo class.
- Điều kiện sử dụng:
  - policy LB phải là `weighted_round_robin`.
  - bật `latency_tracker.enabled=true`.
  - dùng `scipy.optimize.linprog` (không có fallback heuristic).

### Dữ liệu module duy trì
- `class_latency_estimates[class_id][worker]`: EWMA latency theo class-worker.
- `class_completions_window[class_id]`: demand theo class trong cửa sổ hiện tại.
- `next_update_time`: mốc cập nhật kế tiếp theo giây mô phỏng.
- Các biến runtime quan trọng khác:
  - `class_latency_samples[class_id][worker]`: đếm số sample latency theo class-worker.
  - `last_weights`: weight vòng trước (phục vụ EMA smoothing).
  - `latency_sampled_total`: tổng số sample đã dùng cho module.
  - `lp_updates`: số lần update weight thành công.

### Vòng lặp thuật toán
1. Mỗi completion:
  - tăng demand của `request.class_id`.
  - cập nhật EWMA latency class-worker (nếu hợp lệ theo cấu hình tracked-only).
2. Nếu `completion_time >= next_update_time`:
  - gọi `_maybe_update_weights`.
  - tăng `next_update_time += update_interval_seconds`.
- Lưu ý bám theo code hiện tại:
  - Khi `lp_use_tracked_only=true` và completion hiện tại là `latency_tracked=false`,
    module `return` sớm ngay sau khi tăng `class_completions_window`.
  - Nghĩa là check mốc thời gian update chỉ chạy ở completion được phép dùng để học latency.
- Biến implement chính trong bước update:
  - `completion_time = request.t_arrival + latency`.
  - `self.params.lp_use_tracked_only` và `latency_tracked`: cờ lọc sample.
  - `self.params.lp_ewma_gamma`: hệ số EWMA update latency.

### Bài toán LP
- Biến quyết định: `x[c,w]` = tỉ lệ traffic class `c` gán cho worker `w`.
- Mục tiêu:
  - minimize `sum_c sum_w demand[c] * latency_cost[c,w] * x[c,w]`.
- Ràng buộc:
  - `sum_w x[c,w] = 1` cho mọi class `c`.
  - Cân bằng tải worker theo tolerance:
    - `lower <= sum_c demand[c] * x[c,w] <= upper`.
  - `0 <= x[c,w] <= 1`.
- Biến implement chính trong solver (`_solve_lp`):
  - `demand_by_class`: demand từng class trong cửa sổ.
  - `cost_by_class`: chi phí latency class-worker.
  - `c`: vector objective.
  - `a_eq`, `b_eq`: ràng buộc tổng phân phối theo class.
  - `a_ub`, `b_ub`: ràng buộc cân bằng tải worker.
  - `result = linprog(...)`, `result.x`: nghiệm LP từ SciPy.

### Từ nghiệm LP sang weight WRR
1. Tính `worker_load[w] = sum_c demand[c] * x[c,w]`.
2. Normalize sang candidate weight.
3. Clip theo `[min_weight, max_weight]`.
4. (Tuỳ chọn) làm mượt EMA với `last_weights`.
5. Gọi `lb.set_worker_weights(weights)` để apply (LB sẽ normalize tổng = 1).
- Biến implement chính trong bước apply:
  - `worker_loads`: tải worker suy ra từ nghiệm LP.
  - `weights`: vector weight sau normalize/clip/smoothing.
  - `self.params.min_weight`, `self.params.max_weight`: biên clip.
  - `self.params.lp_weight_ema_decay`: hệ số làm mượt.

Quay lại: [algorithms index](README.md) · [docs/README](../README.md)
