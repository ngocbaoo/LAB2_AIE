# Tổng quan hệ thống

## 1. Mục tiêu

Demo này so sánh ba cách tiếp cận Reinforcement Learning trên cùng một bài
toán (`CartPole-v1`, thư viện `gym`), để làm rõ:

1. **Deep Q-Network (DQN)** — học giá trị hành động (value-based).
2. **REINFORCE** — Policy Gradient thuần (policy-based).
3. **Actor-Critic** — kết hợp cả hai: Actor học chính sách, Critic học giá trị
   trạng thái để cung cấp tín hiệu advantage theo từng bước.

Cả ba được huấn luyện với **cùng môi trường, cùng số episode, cùng tập seed**,
để kết quả so sánh được là công bằng nhất có thể ở quy mô baseline. Đây là
nửa "trước" của dự án — nửa "sau" (`docs/improved/`) áp dụng các cải tiến cụ
thể lên từng thuật toán và so sánh lại.

## 2. Kiến trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py                    # seeding + RewardTracker (moving average, solved_at)
│   ├── algorithms/
│   │   ├── dqn.py                   # Deep Q-Network (baseline)
│   │   ├── reinforce.py             # Policy Gradient / REINFORCE (baseline)
│   │   ├── actor_critic.py          # Actor-Critic 1-step (baseline)
│   │   ├── dqn_double.py            # cải tiến — xem docs/improved/
│   │   ├── reinforce_improved.py    # cải tiến — xem docs/improved/
│   │   └── actor_critic_gae.py      # cải tiến — xem docs/improved/
│   ├── train.py                      # chạy nhanh 1 seed, chỉ 3 thuật toán baseline
│   └── train_all.py                  # chạy đủ: baseline + improved x 5 seed, ghi frontend/data/results.json
├── frontend/                          # dashboard đọc frontend/data/results.json
└── docs/
    ├── baseline/       # tài liệu về 3 thuật toán gốc (file này nằm ở đây)
    └── improved/       # tài liệu về các cải tiến + so sánh trước/sau
```

## 3. Luồng dữ liệu

Bộ dữ liệu chính thức dùng cho mọi so sánh (baseline lẫn improved) đến từ
`backend/train_all.py`, chạy **5 seed** cho mỗi trong 6 cấu hình (3 thuật
toán × baseline/improved):

```
backend/train_all.py
   for mỗi (thuật toán, biến thể) trong 6 cấu hình:
       for mỗi seed trong [42, 43, 44, 45, 46]:
           train_fn(seed=seed) → {rewards, moving_avg, solved_at}
       │
       ▼ gộp qua 5 seed
   moving_avg_mean / moving_avg_std theo từng episode
   metrics: solved_rate, solved_at_episode (mean/std), final_avg_reward_last20 (mean/std), ...
        │
        ▼
   frontend/data/results.json   (6 runs, mỗi run có variant: "baseline" | "improved")
        │
        ▼
   frontend/app.js (fetch) → 3 view: so sánh baseline, so sánh improved, so sánh từng cặp trước/sau
```

`backend/train.py` (1 seed, chỉ baseline) vẫn được giữ lại như một cách chạy
nhanh để kiểm tra code còn hoạt động không, nhưng **không phải nguồn dữ liệu
chính thức** — mọi số liệu trong docs đều lấy từ `train_all.py`.

Frontend là site tĩnh (HTML/CSS/JS thuần, không phụ thuộc framework hay CDN
ngoài), nên chỉ cần một local static server là chạy được, không cần build step.

## 4. Bài toán & môi trường

- **Môi trường**: `CartPole-v1` (gym 0.26.2) — trạng thái liên tục 4 chiều
  (vị trí, vận tốc xe; góc, vận tốc góc của cột), hành động rời rạc (trái/phải).
- Mỗi bước còn giữ cột thẳng được cộng reward = 1; episode kết thúc khi cột đổ
  hoặc đạt 500 bước.
- Vì cả DQN, REINFORCE và Actor-Critic ở đây đều xử lý hành động rời rạc, đây
  là bài toán trung lập, không thiên vị thuật toán nào về mặt action space.

## 5. Mô hình mạng nơ-ron (baseline)

Cả ba dùng MLP 2 lớp ẩn (128 unit, ReLU) trên cùng input 4 chiều — khác biệt
nằm ở **đầu ra và hàm mất mát**, không phải độ phức tạp mạng:

| Thuật toán   | Đầu ra mạng                          | Tín hiệu học                                   |
|--------------|----------------------------------------|-------------------------------------------------|
| DQN          | Q(s, a) cho từng hành động             | TD-target từ Target Network + Replay Buffer     |
| REINFORCE    | π(a\|s) — phân phối xác suất hành động | Return chiết khấu G_t của cả episode (chuẩn hoá) |
| Actor-Critic | π(a\|s) **và** V(s) (2 đầu ra chung thân mạng) | Advantage 1 bước: r + γV(s') − V(s)      |

## 6. Cách chạy

```bash
py -3.11 -m pip install -r requirements.txt
cd backend
py -3.11 train_all.py             # 6 cấu hình x 5 seed x 400 episode — mất khoảng 40-60 phút
# hoặc để kiểm tra nhanh (1 seed, chỉ baseline, ~5-6 phút):
py -3.11 train.py

cd ../frontend
py -3.11 -m http.server 8000      # bắt buộc: fetch() cần http://, không mở trực tiếp file://
# mở http://localhost:8000 trên trình duyệt
```
