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
- `class_completions_window[class_id]`: demand theo class trong cửa sổ hiện tại.
- `next_update_time`: mốc cập nhật kế tiếp theo giây mô phỏng.
- Các biến runtime quan trọng khác:
  - `lp_updates`: số lần update weight thành công.
  - `last_lp_class_order`: thứ tự class trong lần solve LP gần nhất.
  - `last_lp_weight_matrix`: ma trận weight sau normalize/clip theo từng class.

### Vòng lặp thuật toán
1. Mỗi completion:
  - tăng demand của `request.class_id`.
  - cập nhật mốc thời gian completion để quyết định có cần chạy LP update không.
2. Nếu `completion_time >= next_update_time`:
  - gọi `_maybe_update_weights`.
  - tăng `next_update_time += update_interval_seconds`.
- Biến implement chính trong bước update:
  - `completion_time = request.t_arrival + latency`.
  - `self.params.lp_ewma_gamma`: hệ số EMA làm mượt weight mới từ LP với weight cũ.

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
  - `cost_by_class`: chi phí latency class-worker lấy từ `LoadBalancer.lat_ewma` (được tracker cập nhật bằng `latency_tracker.ewma_gamma`).
  - `c`: vector objective.
  - `a_eq`, `b_eq`: ràng buộc tổng phân phối theo class.
  - `a_ub`, `b_ub`: ràng buộc cân bằng tải worker.
  - `result = linprog(...)`, `result.x`: nghiệm LP từ SciPy.

## `wrr_separate_lp_latency`
- Mục tiêu: điều chỉnh `worker_weights` cho từng class bằng LP riêng, tối ưu mean latency của từng class.
- Điều kiện sử dụng:
  - policy LB phải là `weighted_round_robin`.
  - bật `latency_tracker.enabled=true`.
  - dùng `scipy.optimize.linprog` (không có fallback heuristic).
- Bài toán LP cho mỗi class `c`:
  - biến quyết định: `x[c,w]` (tỉ lệ traffic class `c` qua worker `w`).
  - objective: minimize `sum_w latency_cost[c,w] * x[c,w]`.
  - ràng buộc: `sum_w x[c,w] = 1`, `0 <= x[c,w] <= 1`.
- Khác với `wrr_lp_latency`:
  - `wrr_lp_latency` tối ưu mean latency toàn hệ thống trong một LP chung cho mọi class.
  - `wrr_separate_lp_latency` solve độc lập từng class, không có coupling liên-class trong objective.
  - `wrr_separate_lp_latency` không dùng ràng buộc cân bằng tải worker kiểu `lp_balance_tolerance`.

### Từ nghiệm LP sang weight WRR
1. Nhận nghiệm LP dạng ma trận `allocation[c,w] = x[c,w]`.
2. Với mỗi class `c`, lấy 1 hàng `allocation[c,:]` và clip theo `[min_weight, max_weight]`.
3. Làm mượt với weight cũ: `smoothed = (1 - lp_ewma_gamma) * old + lp_ewma_gamma * solved`.
4. Clip lại theo `[min_weight, max_weight]`.
5. Apply vào LB của class tương ứng: `class_load_balancers[class_id].set_worker_weights(row_weights)`.
- Biến implement chính trong bước apply:
  - `allocation`: ma trận nghiệm LP.
  - `row_weights`: vector weight cuối cùng sau EMA smoothing.
  - `class_load_balancers`: LB map theo class.
  - `self.params.min_weight`, `self.params.max_weight`: biên clip.
  - `self.params.lp_ewma_gamma`: hệ số làm mượt weight.

### Ví dụ `2 class` và `3 worker`
- Cấu hình controller chỉ nhận `wrr.weights` theo worker, ví dụ:
  - `"weights": [0.2, 0.3, 0.5]`
- Với design hiện tại (mỗi class một LB):
  - LB class 0 khởi tạo với `[0.2, 0.3, 0.5]`.
  - LB class 1 khởi tạo với `[0.2, 0.3, 0.5]`.
- Ở `wrr.mode=lp_latency`, sau khi solve LP có thể ra ma trận:
  - class 0 -> `[0.10, 0.30, 0.60]`
  - class 1 -> `[0.55, 0.35, 0.10]`
- Nghĩa là input config là 1 vector khởi tạo, còn ma trận `class x worker` là output runtime do LP cập nhật.

Quay lại: [algorithms index](README.md) · [docs/README](../README.md)
