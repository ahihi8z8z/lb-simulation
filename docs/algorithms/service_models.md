# Thuật Toán Service Model

Nguồn implementation: `lb_simulation/worker_models.py`.

Các model đều implement `sample_service_time(context, rng)` với:
- `job_size`: kích thước request.
- `n_local`: queue cục bộ tại worker lúc start service.
- `n_global`: inflight toàn cục lúc start service.

## `contention_lognormal`
- Công thức:
  - `S = (a + b*z) * (1 + c*n_local) * (1 + d*max(0, n_global - n0)) * LogNormal(0, sigma)`
  - `S = max(min_s, S)`.
- Ý nghĩa:
  - Thành phần theo `job_size` (`a + b*z`).
  - Thành phần contention local (`n_local`).
  - Thành phần contention global (`n_global` vượt ngưỡng `n0`).
  - Nhiễu lognormal để tạo độ phân tán.
- Biến implement chính:
  - `context.job_size`, `context.n_local`, `context.n_global`: input runtime cho mỗi request.
  - `self.a`, `self.b`, `self.c`, `self.d`, `self.n0`, `self.sigma`, `self.min_s`: tham số model.
  - `base`, `local_factor`, `global_factor`, `noise`: các thành phần trung gian để tính `service`.
  - `rng.lognormvariate(0.0, self.sigma)`: nguồn nhiễu ngẫu nhiên.

## `linear_lognormal`
- Công thức:
  - `S = (a + b*z) * LogNormal(0, sigma)`
  - `S = max(min_s, S)`.
- Ý nghĩa:
  - Tăng tuyến tính theo `job_size`.
  - Không phụ thuộc trực tiếp `n_local`, `n_global`.
- Biến implement chính:
  - `context.job_size`: biến tải chính ảnh hưởng service time.
  - `self.a`, `self.b`, `self.sigma`, `self.min_s`: tham số model.
  - `base`, `noise`: biến trung gian trong hàm `sample_service_time`.

## `fixed`
- Công thức: `S = service_time`.
- Ý nghĩa: service time hằng, dùng cho baseline/đối chứng.
- Biến implement chính:
  - `self.service_time`: giá trị service time cố định lấy từ config.
  - `context`, `rng`: không dùng trong implement (đã `del`).

## `fixed_linear`
- Công thức trong code:
  - `den = clip(a + b*job_size, min, max)`
  - `S = 1 / den`
- Ý nghĩa:
  - Mô hình thông lượng tuyến tính theo `job_size`, rồi đổi sang service time bằng nghịch đảo.
  - Có chặn biên để tránh cực trị.
- Biến implement chính:
  - `context.job_size`: input để tính `denominator`.
  - `self.a`, `self.b`, `self.min_s`, `self.max_s`: tham số model.
  - `denominator`: giá trị sau khi clip, dùng để tính `1.0 / denominator`.
