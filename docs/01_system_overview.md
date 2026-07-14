# Tổng quan hệ thống

## 1. Mục tiêu

Demo này so sánh ba cách tiếp cận Reinforcement Learning trên cùng một bài
toán (`CartPole-v1`, thư viện `gym`), để làm rõ:

1. **Deep Q-Network (DQN)** — học giá trị hành động (value-based).
2. **REINFORCE** — Policy Gradient thuần (policy-based).
3. **Actor-Critic** — kết hợp cả hai: Actor học chính sách, Critic học giá trị
   trạng thái để cung cấp tín hiệu advantage theo từng bước.

Cả ba được huấn luyện với **cùng môi trường, cùng số episode, cùng seed**, để
kết quả so sánh được là công bằng nhất có thể ở quy mô baseline.

## 2. Kiến trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py              # seeding + RewardTracker (moving average, solved_at)
│   ├── algorithms/
│   │   ├── dqn.py             # Deep Q-Network
│   │   ├── reinforce.py       # Policy Gradient (REINFORCE)
│   │   └── actor_critic.py    # Actor-Critic (1-step A2C)
│   └── train.py                # huấn luyện cả 3, xuất frontend/data/results.json
├── frontend/
│   ├── index.html              # dashboard: so sánh lý thuyết + biểu đồ + bảng chỉ số
│   ├── style.css
│   ├── app.js                  # vẽ biểu đồ bằng Canvas API thuần (không cần CDN)
│   └── data/results.json        # sinh ra bởi backend/train.py
└── docs/
    ├── 01_system_overview.md    # file này
    ├── 02_evaluation_metrics.md
    └── 03_results.md
```

## 3. Luồng dữ liệu

```
backend/train.py
   ├─ train_dqn()            ─┐
   ├─ train_reinforce()       ├─ mỗi hàm trả về {rewards, moving_avg, solved_at}
   └─ train_actor_critic()   ─┘
        │
        ▼
   summarize() tính thêm: env_steps_to_solve, final_avg/std, best reward, training_time
        │
        ▼
   frontend/data/results.json
        │
        ▼
   frontend/app.js  (fetch)  →  vẽ biểu đồ Canvas + bảng chỉ số trong index.html
```

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
py -3.11 train.py                 # huấn luyện cả 3 thuật toán, ghi frontend/data/results.json

cd ../frontend
py -3.11 -m http.server 8000      # bắt buộc: fetch() cần http://, không mở trực tiếp file://
# mở http://localhost:8000 trên trình duyệt
```
