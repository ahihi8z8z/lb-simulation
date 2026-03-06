# Hệ Thống Dưới Góc Nhìn Queueing Model

## Mapping thành phần sang queueing model
- Mỗi worker trong `InferencePool` là một server đơn (`capacity=1`), tương ứng hàng đợi `G/G/1`.
- Toàn cụm worker là hệ `N` queue song song, có router ở trước:
  - Router = `LoadBalancer` policy.
  - Arrival stream = hợp của các luồng service class từ `TrafficGenerator`.

## Arrival process
- `trace_replay`: arrival theo timestamp trace, có thể burst tại cùng thời điểm (`traffic_scale`).
- `modeled_gamma`: inter-arrival lấy từ Gamma theo từng window thời gian.
- Với nhiều class, hệ là superposition của nhiều arrival process không đồng nhất.

## Service process
- Service time không i.i.d đơn giản; phụ thuộc trạng thái hệ thống:
  - `job_size`
  - `n_local`
  - `n_global`
- Do đó phù hợp nhất khi nhìn như state-dependent queueing network thay vì M/M/1 cổ điển.

## Routing discipline
- Routing động theo policy:
  - `random`, `round_robin`, `weighted_round_robin`, `least_inflight`, `latency_only`.
- `weighted_round_robin` với control module tạo closed-loop:
  - Completion -> latency estimate/control -> cập nhật `worker_weights` -> đổi routing.

## Latency tracker trong queueing view
- Tracker có thể xem như một nút quan sát (measurement node), không phải server thực phục vụ request.
- Request qua tracker vẫn được phục vụ ở worker thật; tracker dùng để lấy sample latency có kiểm soát.

## Controller như feedback control trong queueing network
- Forward path: arrival -> routing -> queue -> service -> completion.
- Feedback path:
  - completion sample -> EWMA latency estimate -> solve LP -> update routing weight.
- Đây là adaptive routing trên hệ queue song song có service time phụ thuộc trạng thái.

## Chỉ số queueing có thể theo dõi
- Mean/median/p95/p99 latency (response time).
- Avg queue length.
- Avg global inflight.
- Utilization theo worker.
- Throughput.

## Ghi chú về phân tích lý thuyết
- Vì routing và service time đều phụ thuộc trạng thái, mô hình khó có nghiệm đóng.
- Simulator phù hợp cho đánh giá thực nghiệm (what-if, sensitivity) hơn là closed-form analysis.
