# Tổng quan hệ thống

## 1. Mục tiêu

Demo này triển khai **Actor-Critic (A2C)** cho bài toán **tối ưu chiến lược
giao dịch** (`TradingEnv`, tự cài đặt — xem mục 4):

- **Actor** học chính sách π(a|s) — chọn vị thế Sell/Hold/Buy.
- **Critic** học hàm giá trị trạng thái V(s) để cung cấp tín hiệu **advantage**
  theo từng bước, thay vì phải đợi hết episode như policy gradient thuần.

`backend/algorithms/actor_critic.py` dùng advantage 1-bước, cập nhật online
sau mỗi bước; tách rõ 2 hàm theo yêu cầu đề bài: `build_networks()` dựng
Actor và Critic (2 mạng riêng biệt), `custom_loss()` kết hợp log-likelihood
của Actor (nhân với Advantage) và MSE của Critic.

## 2. Kiến trúc thư mục

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
│   ├── train.py                      # chạy nhanh 1 seed — kiểm tra code
│   └── train_all.py                  # chạy đủ: 5 seed, ghi frontend/data/results.json
├── frontend/                          # dashboard đọc frontend/data/results.json
└── docs/
    └── baseline/       # tài liệu (file này nằm ở đây)
```

## 3. Luồng dữ liệu

Bộ dữ liệu chính thức đến từ `backend/train_all.py`, chạy **5 seed**:

```
backend/train_all.py
   for mỗi seed trong [42, 43, 44, 45, 46]:
       train_actor_critic(seed=seed) → {rewards, moving_avg, solved_at}
       │
       ▼ gộp qua 5 seed
   moving_avg_mean / moving_avg_std theo từng episode
   metrics: solved_rate, solved_at_episode (mean/std), final_avg_reward_last20 (mean/std), ...
        │
        ▼
   frontend/data/results.json   (1 run: variant "baseline")
        │
        ▼
   frontend/app.js (fetch) → playback quá trình huấn luyện
```

`backend/train.py` (1 seed) vẫn được giữ lại như một cách chạy nhanh để kiểm
tra code còn hoạt động không, nhưng **không phải nguồn dữ liệu chính thức** —
mọi số liệu trong docs đều lấy từ `train_all.py`.

Frontend là site tĩnh (HTML/CSS/JS thuần, không phụ thuộc framework hay CDN
ngoài), nên chỉ cần một local static server là chạy được, không cần build step.

## 4. Bài toán & môi trường

- **Môi trường**: `TradingEnv` (`backend/trading_env.py`, tự cài đặt, không
  phụ thuộc gym) — mô phỏng giao dịch một tài sản trên **giá đóng cửa thật
  của SPY (ETF theo dõi S&amp;P 500), 2015–2024** (`backend/data/spy.csv`,
  tải qua `backend/fetch_data.py` — dùng `yfinance`, chỉ cần mạng lúc tải,
  không cần khi train).
- **Sinh episode**: mỗi episode bootstrap 1 đoạn **200 phiên liên tục** ngẫu
  nhiên trong lịch sử log-return thật (~2500 phiên, 2015–2024) — vị trí bắt
  đầu chọn theo `seed + episode`, nên cùng seed luôn tái tạo lại đúng chuỗi
  episode. Đây là dữ liệu giá thật 100%, không sinh tổng hợp.
- **State**: cửa sổ 10 log-return gần nhất + vị thế hiện tại → vector 11 chiều.
- **Action**: rời rạc, 3 lựa chọn — `Sell` (vị thế mục tiêu −1), `Hold` (0),
  `Buy` (+1).
- **Reward**: `(vị thế trước đó) × (log-return bước này, thật) − phí giao
  dịch khi đổi vị thế`, nhân 100 để hiển thị theo đơn vị **% lãi/lỗ**. Phí
  giao dịch thấp (0.02% mỗi lần đổi vị thế) để không phạt quá nặng một chính
  sách còn đang khám phá (dò dẫm chuyển vị thế liên tục).
- Mỗi episode dài cố định **200 bước**; mỗi seed × episode bootstrap một
  đoạn lịch sử giá khác nhau (`env.reset(seed=seed + episode)`).

## 5. Mô hình mạng nơ-ron

`backend/algorithms/actor_critic.py`:

```python
def build_networks(obs_dim, n_actions, hidden=128):
    actor = Actor(obs_dim, n_actions, hidden)   # MLP 11 -> 128 -> 3, softmax
    critic = Critic(obs_dim, hidden)             # MLP 11 -> 128 -> 1
    return actor, critic
```

| Mạng          | Vai trò                              | Tín hiệu học                              |
|---------------|----------------------------------------|-------------------------------------------|
| Actor: π(a\|s) | Xác suất đặt vị thế Sell/Hold/Buy | log-likelihood × advantage         |
| Critic: V(s)   | Giá trị trạng thái thị trường     | MSE với TD-target r + γV(s')       |

Advantage 1 bước: `advantage = r + γV(s')·(1-done) − V(s)`, cập nhật **online
sau mỗi bước** (không đợi hết episode). `custom_loss()` gộp cả 2 tín hiệu:

```python
def custom_loss(log_prob, advantage, value, target, value_coef=0.5):
    actor_loss  = -log_prob * advantage.detach()
    critic_loss = (target - value).pow(2)
    return actor_loss + value_coef * critic_loss
```

## 6. Cách chạy

```bash
py -3.11 -m pip install -r requirements.txt
cd backend
py -3.11 fetch_data.py            # (tuỳ chọn) làm mới data/spy.csv
py -3.11 train_all.py             # 5 seed x 400 episode
# hoặc để kiểm tra nhanh (1 seed):
py -3.11 train.py

cd ../frontend
py -3.11 -m http.server 8000      # bắt buộc: fetch() cần http://, không mở trực tiếp file://
# mở http://localhost:8000 trên trình duyệt
```
