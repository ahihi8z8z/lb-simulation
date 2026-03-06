# Flow Chart Xử Lý Request

```mermaid
flowchart TD
    A["TrafficGenerator tạo Request"] --> B["on_arrival(request)"]
    B --> C["LoadBalancer.choose_worker"]
    C --> D{"Worker được chọn<br/>có phải tracker worker?"}

    D -- Không --> E["Dispatch trực tiếp tới worker thật"]
    E --> F["InferencePool._serve"]
    F --> G["Service time theo worker model"]
    G --> H["Completion callback"]

    D -- Có --> I["consume_redirect_target(rid)"]
    I --> J["Controller.forward_via_latency_tracker"]
    J --> K["Dispatch tracker + worker thật"]
    K --> L["InferencePool._serve trên worker thật"]
    L --> H

    H --> M["Controller.on_request_complete"]
    M --> N{latency_tracked?}
    N -- Có --> O["LatencyTracker.observe + update LB lat_ewma"]
    N -- Không --> P["Bỏ qua tracker update"]
    O --> Q["LB control module on_request_complete"]
    P --> Q
    Q --> R["WrrLpLatencyControlModule<br/>có thể update worker_weights theo interval"]
```

## Ghi chú
- Tracker không xử lý service time thực tế; tracker chỉ làm bước redirect/sampling.
- Completion luôn phát sinh tại worker thật; callback controller quyết định có dùng sample đó để update latency hay không.
