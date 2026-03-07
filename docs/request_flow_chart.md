# Flow Chart Xử Lý Request

```mermaid
flowchart TD
    A["TrafficGenerator tạo Request"] --> B["on_arrival(request)"]
    B --> B1["Chọn LB theo class_id: LoadBalancer[class_id]"]
    B1 --> C["LoadBalancer[class_id].choose_worker"]
    C --> D{"Worker được chọn<br/>có phải tracker worker?"}

    D -- Không --> E["Dispatch trực tiếp tới worker thật"]
    E --> F["InferencePool._serve"]
    F --> G["Service time theo worker model"]
    G --> H["Completion callback"]

    D -- Có --> I["consume_redirect_target(rid)"]
    I --> J["Controller.forward_via_latency_tracker<br/>(theo class_id)"]
    J --> K["Dispatch tracker + worker thật"]
    K --> L["InferencePool._serve trên worker thật"]
    L --> H

    H --> M["Controller.on_request_complete"]
    M --> N{latency_tracked?}
    N -- Có --> O["LatencyTracker[class_id].observe<br/>+ update LB[class_id].lat_ewma"]
    N -- Không --> P["Bỏ qua tracker update"]
    O --> Q["LB control module on_request_complete"]
    P --> Q
    Q --> R["WrrLpLatencyControlModule<br/>có thể update worker_weights theo interval"]
```

## Ghi chú
- Tracker không xử lý service time thực tế; tracker chỉ làm bước redirect/sampling.
- Mỗi service class có LB riêng và tracker state riêng.
- Completion luôn phát sinh tại worker thật; callback controller quyết định có dùng sample đó để update latency hay không.

## Xem thêm
- Kiến trúc: [architecture_overview](architecture_overview.md)
- Luồng chi tiết: [flow_diagrams](flow_diagrams.md)
