# Design Ghi Chú

## Mục tiêu thiết kế
- Cho phép thay policy/load model mà không sửa lõi runner.
- Tách phần thu tín hiệu latency khỏi LB policy để kiểm soát sampling rõ ràng.
- Tách logic điều khiển tham số LB thành module cắm được.

## Quyết định kiến trúc chính
- `Controller` là orchestration layer:
  - nạp config.
  - tạo `LatencyTrackerWorker` theo từng `class_id` (nếu bật).
  - tạo `LoadBalancerControlModule`.
- Mỗi service class có một `LoadBalancer` riêng; LB chỉ giữ state và gọi policy, không tự học từ completion stream.
- `WrrLpLatencyControlModule` (`wrr_lp_latency`):
  - lấy latency estimate từ `LatencyTrackerWorker` (`LoadBalancer.lat_ewma`).
  - gom demand theo class trong cửa sổ thời gian.
  - giải LP cho allocation matrix `class x worker`.
  - làm mượt weight LP bằng `lp_ewma_gamma`, rồi apply vào LB của class tương ứng.
- `WrrSeparateLpLatencyControlModule` (`wrr_separate_lp_latency`):
  - lấy latency estimate từ `LatencyTrackerWorker` tương tự.
  - giải LP độc lập cho từng class để tối ưu mean latency của class đó.
  - làm mượt weight LP bằng `lp_ewma_gamma`, rồi apply vào LB của class tương ứng.

## Lý do update theo interval thời gian
- Tránh phụ thuộc tốc độ request (cao/thấp) khi quyết định thời điểm cập nhật.
- Dễ map với vận hành thực tế kiểu control-loop định kỳ (ví dụ mỗi 60 giây).
- Ổn định hơn trong burst traffic vì control update không bị kích dồn theo count.

## Ràng buộc và validate
- `wrr.mode=lp_latency` và `wrr.mode=separate_lp` chỉ dùng với `weighted_round_robin`.
- `lp_latency` và `separate_lp` bắt buộc bật `latency_tracker.enabled=true`.
- Không có fallback heuristic cho LP: nếu `linprog` fail thì raise lỗi runtime.
- Weight input/output luôn hợp lệ:
  - mỗi phần tử > 0.
  - khi apply vào LB luôn được normalize tổng = `1.0`.

## Điểm mở rộng
- Thêm LB policy mới: kế thừa `LoadBalancingPolicy`, dùng `@register_policy`.
- Thêm redirect policy mới: kế thừa `LatencyRedirectPolicy`, dùng `@register_latency_redirect_policy`.
- Thêm LB control module mới: kế thừa `LoadBalancerControlModule`, dùng `@register_load_balancer_control_module`.
- Thêm worker model mới: kế thừa `WorkerServiceModel`, dùng `@register_worker_model`.
