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
- `class_load_balancers[class_id]`: map class -> LB instance.
- `class_latency_estimates[class_id][worker]`: EWMA latency theo class-worker.
- `class_completions_window[class_id]`: demand theo class trong cửa sổ hiện tại.
- `next_update_time`: mốc cập nhật kế tiếp theo giây mô phỏng.
- Các biến runtime quan trọng khác:
  - `class_latency_samples[class_id][worker]`: đếm số sample latency theo class-worker.
  - `latency_sampled_total`: tổng số sample đã dùng cho module.
  - `lp_updates`: số lần update weight thành công.
  - `last_lp_class_order`: thứ tự class trong lần solve LP gần nhất.
  - `last_lp_weight_matrix`: ma trận weight sau normalize/clip theo từng class.

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
  - minimize `sum_c sum_w p[c] * latency_cost[c,w] * x[c,w]`, với `p[c] = demand[c]/sum_k demand[k]`.
  - Đây chính là tối thiểu hóa mean latency toàn hệ thống trong cửa sổ cập nhật.
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
1. Nhận nghiệm LP dạng ma trận `allocation[c,w] = x[c,w]`.
2. Với mỗi class `c`, lấy 1 hàng `allocation[c,:]`.
3. Normalize + clip hàng đó theo `[min_weight, max_weight]`.
4. Apply vào LB của class tương ứng: `class_load_balancers[class_id].set_worker_weights(row_weights)`.
- Biến implement chính trong bước apply:
  - `allocation`: ma trận nghiệm LP.
  - `row_weights`: vector weight cho một class.
  - `class_load_balancers`: LB map theo class.
  - `self.params.min_weight`, `self.params.max_weight`: biên clip.

Quay lại: [algorithms index](README.md) · [docs/README](../README.md)
