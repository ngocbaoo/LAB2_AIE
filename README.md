# Actor-Critic (A2C) cho tối ưu chiến lược giao dịch

Triển khai **Actor-Critic (A2C)** cho bài toán **tối ưu chiến lược giao
dịch** (`TradingEnv` — môi trường giao dịch tự cài đặt, không phụ thuộc gym,
dùng **giá đóng cửa thật của SPY (S&amp;P 500 ETF), 2015–2024**
(`backend/data/spy.csv`, tải qua `backend/fetch_data.py`): state là cửa sổ
log-return gần nhất + vị thế hiện tại, action là Sell/Hold/Buy, reward là %
lãi/lỗ mỗi bước; mỗi episode bootstrap 1 đoạn 200 phiên liên tục ngẫu nhiên
trong lịch sử giá thật).

**Actor** (`backend/algorithms/actor_critic.py`) chọn hành động, **Critic**
học V(s) để đánh giá hành động đó qua advantage 1-bước, cập nhật online sau
mỗi bước (`build_networks` dựng 2 mạng, `custom_loss` kết hợp log-likelihood
của Actor với Advantage và MSE của Critic).

Huấn luyện trên **5 seed** để kết quả đáng tin cậy hơn một lần chạy đơn lẻ.

Chi tiết kiến trúc, chỉ số đánh giá và phân tích kết quả nằm trong
[`docs/baseline/`](./docs/baseline).

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
yfinance>=0.2   # chi can cho fetch_data.py (tai gia SPY 1 lan)
```

> Trên Windows nếu `python`/`pip` không nhận, dùng launcher `py`:
> `py -3.11 -m pip install -r requirements.txt`

## Chạy huấn luyện (backend)

Dữ liệu giá SPY đã được cache sẵn ở `backend/data/spy.csv`. Muốn tải lại/làm
mới:

```bash
cd backend
python fetch_data.py
```

Sau đó huấn luyện:

```bash
python train_all.py
```

Script này huấn luyện Actor-Critic trên **5 seed** × 400 episode, rồi ghi kết
quả đã gộp (mean/std qua seed) ra `frontend/data/results.json` — file này là
dữ liệu duy nhất mà frontend đọc.

Muốn kiểm tra nhanh (chỉ 1 seed):

```bash
python train.py
```

Hoặc chạy thử trực tiếp (train 50 episode, in moving-average cuối cùng ra
console, không ghi JSON):

```bash
python algorithms/actor_critic.py
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
2. Biểu đồ minh họa quá trình huấn luyện — có thể tua lại theo episode
   (Play/Pause, scrubber, tốc độ phát), dải mờ ±1 std giữa 5 seed, bảng chỉ số
3. Kết luận rút ra từ thực nghiệm

## Cấu trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py                    # seeding + RewardTracker (moving average, solved_at)
│   ├── trading_env.py                # TradingEnv — bootstrap từ giá SPY thật (data/spy.csv)
│   ├── fetch_data.py                 # tải/làm mới giá SPY từ Yahoo Finance -> data/spy.csv
│   ├── data/spy.csv                  # giá đóng cửa SPY 2015-2024 (cache, không cần mạng để train)
│   ├── algorithms/
│   │   └── actor_critic.py          # Actor-Critic 1-step (build_networks + custom_loss)
│   ├── train.py                      # 1 seed — kiểm tra nhanh
│   └── train_all.py                  # 5 seed — nguồn dữ liệu chính thức
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── data/results.json             # sinh ra bởi backend/train_all.py
└── docs/
    └── baseline/
        ├── 01_system_overview.md
        ├── 02_evaluation_metrics.md
        └── 03_results.md
```

## Tài liệu

- [`docs/baseline/01_system_overview.md`](./docs/baseline/01_system_overview.md) — tổng quan hệ thống, kiến trúc, luồng dữ liệu
- [`docs/baseline/02_evaluation_metrics.md`](./docs/baseline/02_evaluation_metrics.md) — giải thích các chỉ số đánh giá
- [`docs/baseline/03_results.md`](./docs/baseline/03_results.md) — kết quả và phân tích
