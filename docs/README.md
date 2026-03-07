# Tài Liệu Hệ Thống

Thư mục này gom toàn bộ tài liệu kiến trúc và thiết kế của simulator.

## Danh mục
- [`architecture_overview.md`](architecture_overview.md): tổng quan kiến trúc, module chính và điểm mở rộng.
- [`request_flow_chart.md`](request_flow_chart.md): flow chart xử lý request từ arrival đến completion.
- [`flow_diagrams.md`](flow_diagrams.md): các biểu đồ luồng chi tiết (runtime/control loop).
- [`design.md`](design.md): quyết định thiết kế, ràng buộc và nguyên tắc vận hành.
- [`variables_and_classes.md`](variables_and_classes.md): tác dụng các biến quan trọng và các class chính.
- [`queueing_model.md`](queueing_model.md): mô tả hệ thống dưới góc nhìn queueing model.
- [`algorithms/`](algorithms/README.md): mô tả thuật toán theo 3 nhóm:
  - [load balancer](algorithms/load_balancer.md)
  - [service model](algorithms/service_models.md)
  - [load balancer control](algorithms/load_balancer_control.md)

## Liên kết nhanh
- Kiến trúc và luồng: [architecture_overview](architecture_overview.md) · [request_flow_chart](request_flow_chart.md) · [flow_diagrams](flow_diagrams.md)
- Thuật toán: [algorithms index](algorithms/README.md) · [LB](algorithms/load_balancer.md) · [Service models](algorithms/service_models.md) · [LB control](algorithms/load_balancer_control.md)
- Queueing view: [queueing_model](queueing_model.md)
- Biến & class: [variables_and_classes](variables_and_classes.md)

## Phạm vi tài liệu
- Tài liệu phản ánh code hiện tại trong `lb_simulation/`.
- Mọi mô tả policy/control đều bám theo implementation thực tế, không phải giả định.
