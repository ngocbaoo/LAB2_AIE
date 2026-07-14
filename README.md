# Actor-Critic (A2C) — Baseline &amp; Cải tiến GAE

Triển khai và so sánh trực quan **Actor-Critic (A2C)** cho bài toán **tối ưu
chiến lược giao dịch** (`TradingEnv` — môi trường giao dịch tổng hợp tự cài
đặt, không phụ thuộc gym: state là cửa sổ log-return gần nhất + vị thế hiện
tại, action là Sell/Hold/Buy, reward là % lãi/lỗ mỗi bước), theo hai giai đoạn:

1. **Baseline** (`backend/algorithms/actor_critic.py`) — Actor chọn hành
   động, Critic học V(s) để đánh giá hành động đó qua advantage 1-bước, cập
   nhật online sau mỗi bước.
2. **Improved** (`backend/algorithms/actor_critic_gae.py`) — **Actor-Critic +
   GAE(λ=0.95)**, cân bằng bias/variance tốt hơn advantage 1-bước, cộng
   entropy bonus và gradient clipping.

Cả hai cấu hình được huấn luyện trên **5 seed** để so sánh đáng tin cậy hơn
một lần chạy đơn lẻ.

Chi tiết kiến trúc, chỉ số đánh giá và phân tích kết quả nằm trong
[`docs/baseline/`](./docs/baseline) và [`docs/improved/`](./docs/improved).

## Yêu cầu

- Python 3.10+ (đã test với 3.11)
- Trình duyệt bất kỳ cho phần frontend

## Cài đặt

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
torch>=2.0
numpy>=1.24
```

> Trên Windows nếu `python`/`pip` không nhận, dùng launcher `py`:
> `py -3.11 -m pip install -r requirements.txt`

## Chạy huấn luyện (backend)

```bash
cd backend
python train_all.py
```

Script này huấn luyện **cả 2 cấu hình** (Actor-Critic baseline / GAE), mỗi
cấu hình trên **5 seed** × 400 episode, rồi ghi kết quả đã gộp (mean/std qua
seed) ra `frontend/data/results.json` — file này là dữ liệu duy nhất mà
frontend đọc.

Muốn kiểm tra nhanh (chỉ 1 seed):

```bash
python train.py
```

Hoặc chạy thử từng thuật toán riêng lẻ (train 50 episode, in moving-average
cuối cùng ra console, không ghi JSON):

```bash
python algorithms/actor_critic.py
python algorithms/actor_critic_gae.py
```

## Chạy frontend

Frontend là site tĩnh (HTML/CSS/JS thuần, không cần cài gì thêm), nhưng
**phải phục vụ qua HTTP** vì trang dùng `fetch()` để tải
`data/results.json` — mở trực tiếp bằng `file://` sẽ bị trình duyệt chặn CORS.

```bash
cd frontend
python -m http.server 8000
```

Sau đó mở `http://localhost:8000` trên trình duyệt.

Trang gồm:

1. Sơ đồ giải thích Actor-Critic (Actor / Critic / advantage)
2. Tóm tắt cải tiến GAE
3. Biểu đồ minh họa &amp; so sánh huấn luyện — có thể tua lại theo episode
   (Play/Pause, scrubber, tốc độ phát), ghép baseline (nét đứt) và improved
   (nét liền) trên cùng biểu đồ, dải mờ ±1 std giữa 5 seed, bảng chỉ số

## Cấu trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py                    # seeding + RewardTracker (moving average, solved_at)
│   ├── trading_env.py                # TradingEnv — môi trường giao dịch dùng chung cho 2 cấu hình
│   ├── algorithms/
│   │   ├── actor_critic.py          # Actor-Critic 1-step (baseline)
│   │   └── actor_critic_gae.py      # Actor-Critic + GAE (improved)
│   ├── train.py                      # 1 seed — kiểm tra nhanh
│   └── train_all.py                  # 2 cấu hình x 5 seed — nguồn dữ liệu chính thức
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── data/results.json             # sinh ra bởi backend/train_all.py
└── docs/
    ├── baseline/
    │   ├── 01_system_overview.md
    │   ├── 02_evaluation_metrics.md
    │   └── 03_results.md
    └── improved/
        ├── 01_system_overview.md     # lý thuyết GAE
        ├── 02_evaluation_metrics.md
        └── 03_results.md             # kết quả baseline vs GAE
```

## Tài liệu

- [`docs/baseline/01_system_overview.md`](./docs/baseline/01_system_overview.md) — tổng quan hệ thống, kiến trúc, luồng dữ liệu
- [`docs/baseline/02_evaluation_metrics.md`](./docs/baseline/02_evaluation_metrics.md) — giải thích các chỉ số đánh giá
- [`docs/baseline/03_results.md`](./docs/baseline/03_results.md) — kết quả baseline và phân tích
- [`docs/improved/01_system_overview.md`](./docs/improved/01_system_overview.md) — lý thuyết GAE
- [`docs/improved/02_evaluation_metrics.md`](./docs/improved/02_evaluation_metrics.md) — cách so sánh baseline/improved
- [`docs/improved/03_results.md`](./docs/improved/03_results.md) — kết quả improved và so sánh trước/sau
