# Thuật Toán Load Balancer

Nguồn implementation: `lb_simulation/lb_policies.py`.

Lưu ý runtime hiện tại:
- Mỗi service class có một `LoadBalancer` riêng, nên mỗi policy instance hoạt động trên state của class đó.

## `random`
- Ý tưởng: chọn worker ngẫu nhiên đều.
- Công thức: `worker_id ~ Uniform({0..N-1})`.
- Khi dùng: baseline đơn giản, không dùng feedback.
- Biến implement chính:
  - `lb.num_workers`: số worker để lấy miền chọn.
  - `lb.rng`: random generator dùng trong `randrange`.

## `round_robin`
- Ý tưởng: quay vòng tuần tự worker.
- State nội bộ: `_next_worker`.
- Quy tắc:
1. Chọn worker hiện tại.
2. Tăng `_next_worker = (_next_worker + 1) mod N`.
- Biến implement chính:
  - `self._next_worker`: con trỏ worker kế tiếp trong policy instance.
  - `lb.num_workers`: modulo để wrap vòng.

## `weighted_round_robin`
- Ý tưởng: smooth WRR theo điểm tích lũy.
- Input runtime: `lb.worker_weights` (đã normalize tổng = 1 trong `LoadBalancer.set_worker_weights`).
- State nội bộ: `_current_weights[i]`.
- Quy tắc mỗi request:
1. Với mọi worker `i`: `current[i] += weight[i]`.
2. Chọn worker có `current[i]` lớn nhất (tie-break ngẫu nhiên).
3. Trừ `sum(weight)` khỏi `current[selected]`.
- Tác dụng: phân phối dài hạn theo tỷ lệ weight nhưng vẫn mượt theo thời gian.
- Biến implement chính:
  - `self._current_weights: List[float]`: điểm tích lũy hiện tại theo worker.
  - `lb.worker_weights`: vector weight runtime do controller/LB cung cấp.
  - `total_weight`: tổng weight mỗi vòng (thực tế gần bằng 1).
  - `candidates`: danh sách worker đồng điểm max để tie-break.
  - `lb.rng`: chọn ngẫu nhiên khi có nhiều worker đồng điểm.

## `least_inflight`
- Ý tưởng: chọn worker đang xử lý ít request nhất.
- Score: `score[i] = inflight[i]`.
- Chọn `argmin(score)` với tie-break ngẫu nhiên.
- Biến implement chính:
  - `lb.inflight`: số request đang chạy theo worker.
  - `lb.argmin_score(...)`: helper chọn min + random tie-break.

## `latency_only`
- Ý tưởng: tối ưu latency estimate có exploration.
- Phần exploration:
  - Epsilon-greedy: với xác suất `epsilon` chọn random worker.
- Phần exploit:
  - Base score: `lat_ewma[i] * (1 + inflight[i]) + penalty[i]`.
  - Optimism bonus: `explore_coef / sqrt(feedback_count[i] + 1)`.
  - Final score: `base - bonus`.
  - Chọn worker có score nhỏ nhất.
- Ghi chú: policy này cần latency tracker bật để có feedback.
- Biến implement chính:
  - `lb.epsilon`: xác suất exploration ngẫu nhiên.
  - `lb.explore_coef`: cường độ optimism bonus.
  - `lb.lat_ewma`: latency estimate theo worker.
  - `lb.inflight`: tải tức thời theo worker.
  - `lb.penalty`: phần phạt bổ sung theo worker.
  - `lb.feedback_count`: số feedback đã nhận (để giảm bonus dần).
  - `scores`: danh sách score cuối cùng đưa vào `lb.argmin_score`.

Quay lại: [algorithms index](README.md) · [docs/README](../README.md)
