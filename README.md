# Deep Q-Learning vs Policy Gradient vs Actor-Critic — Demo

So sánh trực quan ba thuật toán Reinforcement Learning trên `CartPole-v1`:

- **Deep Q-Network (DQN)** — học giá trị hành động Q(s, a) (value-based)
- **REINFORCE (Policy Gradient)** — học trực tiếp phân phối xác suất hành động π(a|s) (policy-based)
- **Actor-Critic** — Actor chọn hành động, Critic học V(s) để đánh giá hành động đó (advantage), kết hợp cả hai cách trên

Chi tiết kiến trúc, chỉ số đánh giá và phân tích kết quả nằm trong [`docs/`](./docs).

## Yêu cầu

- Python 3.10+ (đã test với 3.11)
- Trình duyệt bất kỳ cho phần frontend

## Cài đặt

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
gym==0.26.2
torch>=2.0
numpy>=1.24
```

> Trên Windows nếu `python`/`pip` không nhận, dùng launcher `py`:
> `py -3.11 -m pip install -r requirements.txt`

## Chạy huấn luyện (backend)

```bash
cd backend
python train.py
```

Script huấn luyện cả 3 thuật toán (mặc định 400 episode/thuật toán, seed 42)
trên `CartPole-v1` và ghi kết quả ra `frontend/data/results.json` — file này
là dữ liệu mà frontend đọc để vẽ biểu đồ và bảng chỉ số. Có thể chạy riêng lẻ
từng thuật toán để thử nhanh, ví dụ:

```bash
python algorithms/dqn.py
python algorithms/reinforce.py
python algorithms/actor_critic.py
```

(mỗi file khi chạy trực tiếp sẽ tự train 50 episode và in ra moving-average
cuối cùng, dùng để kiểm tra nhanh code còn chạy được không, không ghi JSON.)

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

1. So sánh lý thuyết DQN vs Policy Gradient
2. Sơ đồ giải thích Actor-Critic (Actor / Critic / advantage)
3. **Minh họa quá trình huấn luyện** — biểu đồ reward có thể tua lại theo
   từng episode: nút Play/Pause, thanh trượt tua, chọn tốc độ phát, và bảng
   chỉ số trực tiếp (reward hiện tại + đã hội tụ hay chưa) cho từng thuật toán
4. Bảng chỉ số đánh giá cuối cùng (episode hội tụ, reward trung bình, độ lệch
   chuẩn, thời gian huấn luyện...)

## Cấu trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py               # seeding + RewardTracker (moving average, solved_at)
│   ├── algorithms/
│   │   ├── dqn.py
│   │   ├── reinforce.py
│   │   └── actor_critic.py
│   └── train.py                 # huấn luyện cả 3, xuất frontend/data/results.json
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── data/results.json         # sinh ra bởi backend/train.py, đọc bởi frontend
└── docs/
    ├── 01_system_overview.md
    ├── 02_evaluation_metrics.md
    └── 03_results.md
```

## Tài liệu

- [`docs/01_system_overview.md`](./docs/01_system_overview.md) — tổng quan hệ thống, kiến trúc, luồng dữ liệu
- [`docs/02_evaluation_metrics.md`](./docs/02_evaluation_metrics.md) — giải thích các chỉ số đánh giá
- [`docs/03_results.md`](./docs/03_results.md) — kết quả thực nghiệm và phân tích
