# Tổng quan hệ thống

## 1. Mục tiêu

Demo này triển khai **Actor-Critic (A2C)** cho bài toán **tối ưu chiến lược
giao dịch** (`TradingEnv`, tự cài đặt — xem mục 4):

- **Actor** học chính sách π(a|s) — chọn vị thế Sell/Hold/Buy.
- **Critic** học hàm giá trị trạng thái V(s) để cung cấp tín hiệu **advantage**
  theo từng bước, thay vì phải đợi hết episode như policy gradient thuần.

Baseline (`backend/algorithms/actor_critic.py`) dùng advantage 1-bước, cập
nhật online sau mỗi bước. Đây là nửa "trước" của dự án — nửa "sau"
(`docs/improved/`) áp dụng GAE (Generalized Advantage Estimation) lên đúng
thuật toán này và so sánh lại, cùng một môi trường, cùng ngân sách huấn
luyện, cùng tập seed.

## 2. Kiến trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py                    # seeding + RewardTracker (moving average, solved_at)
│   ├── trading_env.py                # TradingEnv — môi trường giao dịch dùng chung cho cả 2 cấu hình
│   ├── algorithms/
│   │   ├── actor_critic.py          # Actor-Critic 1-step (baseline)
│   │   └── actor_critic_gae.py      # Actor-Critic + GAE — cải tiến, xem docs/improved/
│   ├── train.py                      # chạy nhanh 1 seed — kiểm tra code
│   └── train_all.py                  # chạy đủ: baseline + improved x 5 seed, ghi frontend/data/results.json
├── frontend/                          # dashboard đọc frontend/data/results.json
└── docs/
    ├── baseline/       # tài liệu về Actor-Critic baseline (file này nằm ở đây)
    └── improved/       # tài liệu về cải tiến GAE + so sánh trước/sau
```

## 3. Luồng dữ liệu

Bộ dữ liệu chính thức đến từ `backend/train_all.py`, chạy **5 seed** cho mỗi
trong 2 cấu hình (baseline / improved):

```
backend/train_all.py
   for mỗi biến thể (baseline, improved) trong 2 cấu hình:
       for mỗi seed trong [42, 43, 44, 45, 46]:
           train_fn(seed=seed) → {rewards, moving_avg, solved_at}
       │
       ▼ gộp qua 5 seed
   moving_avg_mean / moving_avg_std theo từng episode
   metrics: solved_rate, solved_at_episode (mean/std), final_avg_reward_last20 (mean/std), ...
        │
        ▼
   frontend/data/results.json   (2 runs: variant "baseline" | "improved")
        │
        ▼
   frontend/app.js (fetch) → 1 view: playback so sánh baseline vs improved
```

`backend/train.py` (1 seed) vẫn được giữ lại như một cách chạy nhanh để kiểm
tra code còn hoạt động không, nhưng **không phải nguồn dữ liệu chính thức** —
mọi số liệu trong docs đều lấy từ `train_all.py`.

Frontend là site tĩnh (HTML/CSS/JS thuần, không phụ thuộc framework hay CDN
ngoài), nên chỉ cần một local static server là chạy được, không cần build step.

## 4. Bài toán & môi trường

- **Môi trường**: `TradingEnv` (`backend/trading_env.py`, tự cài đặt, không
  phụ thuộc gym) — mô phỏng giao dịch một tài sản trên chuỗi giá tổng hợp.
- **Sinh giá**: log-return đi theo quá trình tự tương quan bậc 1 (AR(1)):
  `r_t = φ·r_{t-1} + ε_t`, `ε_t ~ N(0, σ)`, `φ = 0.4`, `σ = 0.01`. Vì φ > 0,
  return có xu hướng "momentum" — tiếp diễn dấu của return ngay trước đó. Đây
  chính là "alpha" mà agent có thể học được từ state; nếu dùng random walk
  thuần (không tự tương quan) thì không có tín hiệu nào trong state dự báo
  được return kế tiếp, và chính sách tối ưu sẽ suy biến về "không làm gì".
- **State**: cửa sổ 10 log-return gần nhất + vị thế hiện tại → vector 11 chiều.
- **Action**: rời rạc, 3 lựa chọn — `Sell` (vị thế mục tiêu −1), `Hold` (0),
  `Buy` (+1).
- **Reward**: `(vị thế trước đó) × (log-return bước này) − phí giao dịch khi
  đổi vị thế`, nhân 100 để hiển thị theo đơn vị **% lãi/lỗ**. Phí giao dịch
  thấp (0.02% mỗi lần đổi vị thế) để không phạt quá nặng một chính sách còn
  đang khám phá (dò dẫm chuyển vị thế liên tục).
- Mỗi episode dài cố định **200 bước**; mỗi seed × episode sinh một chuỗi giá
  khác nhau (`env.reset(seed=seed + episode)`).

## 5. Mô hình mạng nơ-ron (baseline)

`backend/algorithms/actor_critic.py` dùng một MLP dùng chung thân mạng (128
unit, ReLU) trên input 11 chiều, với 2 đầu ra (2 head):

| Đầu ra          | Vai trò                              | Tín hiệu học                              |
|-----------------|----------------------------------------|-------------------------------------------|
| π(a\|s) (actor head)  | Xác suất đặt vị thế Sell/Hold/Buy | log-likelihood × advantage         |
| V(s) (critic head)    | Giá trị trạng thái thị trường     | MSE với TD-target r + γV(s')       |

Advantage 1 bước: `advantage = r + γV(s')·(1-done) − V(s)`, cập nhật **online
sau mỗi bước** (không đợi hết episode).

## 6. Cách chạy

```bash
py -3.11 -m pip install -r requirements.txt
cd backend
py -3.11 train_all.py             # 2 cấu hình x 5 seed x 400 episode
# hoặc để kiểm tra nhanh (1 seed):
py -3.11 train.py

cd ../frontend
py -3.11 -m http.server 8000      # bắt buộc: fetch() cần http://, không mở trực tiếp file://
# mở http://localhost:8000 trên trình duyệt
```
