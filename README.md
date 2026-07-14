# Deep Q-Learning vs Policy Gradient vs Actor-Critic — Baseline &amp; Cải tiến

So sánh trực quan ba thuật toán Reinforcement Learning trên `CartPole-v1`,
theo hai giai đoạn:

1. **Baseline** — cài đặt gốc của ba thuật toán
   - **Deep Q-Network (DQN)** — học giá trị hành động Q(s, a) (value-based)
   - **REINFORCE (Policy Gradient)** — học trực tiếp phân phối xác suất hành động π(a|s) (policy-based)
   - **Actor-Critic** — Actor chọn hành động, Critic học V(s) để đánh giá hành động đó (advantage 1-bước)
2. **Improved** — mỗi thuật toán được cải tiến nhắm đúng điểm yếu quan sát được ở baseline:
   - **Double DQN** — giảm overestimation bias
   - **REINFORCE + learned baseline + entropy bonus** — giảm variance, tránh hội tụ sớm rồi thoái lui
   - **Actor-Critic + GAE(λ=0.95)** — cân bằng bias/variance tốt hơn advantage 1-bước

Mọi cấu hình (cả baseline lẫn improved) được huấn luyện trên **5 seed** để so
sánh đáng tin cậy hơn một lần chạy đơn lẻ.

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
gym==0.26.2
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

Script này huấn luyện **cả 6 cấu hình** (DQN/REINFORCE/Actor-Critic ×
baseline/improved), mỗi cấu hình trên **5 seed** × 400 episode, rồi ghi kết
quả đã gộp (mean/std qua seed) ra `frontend/data/results.json` — file này là
dữ liệu duy nhất mà frontend đọc. Chạy đủ mất khoảng 40-60 phút tùy phần
cứng.

Muốn kiểm tra nhanh (chỉ 1 seed, chỉ baseline, ~5-6 phút):

```bash
python train.py
```

Hoặc chạy thử từng thuật toán riêng lẻ (train 50 episode, in moving-average
cuối cùng ra console, không ghi JSON):

```bash
python algorithms/dqn.py
python algorithms/dqn_double.py
python algorithms/reinforce.py
python algorithms/reinforce_improved.py
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

1. So sánh lý thuyết DQN vs Policy Gradient
2. Sơ đồ giải thích Actor-Critic (Actor / Critic / advantage)
3. Tóm tắt các cải tiến áp dụng lên từng thuật toán
4. Ba tab minh họa &amp; so sánh:
   - **So sánh Baseline** — biểu đồ có thể tua lại theo episode (Play/Pause,
     scrubber, tốc độ phát), dải mờ ±1 std giữa 5 seed, bảng chỉ số
   - **So sánh Improved** — cùng giao diện, cho 3 phiên bản đã cải tiến
   - **Trước / Sau theo thuật toán** — mỗi thuật toán một biểu đồ ghép
     baseline (nét đứt) và improved (nét liền) để thấy trực tiếp cải tiến có
     tác dụng hay không

## Cấu trúc thư mục

```
Lab2_aie/
├── requirements.txt
├── backend/
│   ├── common.py                    # seeding + RewardTracker (moving average, solved_at)
│   ├── algorithms/
│   │   ├── dqn.py                   # Deep Q-Network (baseline)
│   │   ├── reinforce.py             # Policy Gradient / REINFORCE (baseline)
│   │   ├── actor_critic.py          # Actor-Critic 1-step (baseline)
│   │   ├── dqn_double.py            # Double DQN (improved)
│   │   ├── reinforce_improved.py    # REINFORCE + baseline + entropy (improved)
│   │   └── actor_critic_gae.py      # Actor-Critic + GAE (improved)
│   ├── train.py                      # 1 seed, chỉ baseline — kiểm tra nhanh
│   └── train_all.py                  # 6 cấu hình x 5 seed — nguồn dữ liệu chính thức
├── frontend/
│   ├── index.html                    # 3 tab: baseline / improved / trước-sau
│   ├── style.css
│   ├── app.js
│   └── data/results.json             # sinh ra bởi backend/train_all.py
└── docs/
    ├── baseline/
    │   ├── 01_system_overview.md
    │   ├── 02_evaluation_metrics.md
    │   └── 03_results.md
    └── improved/
        ├── 01_system_overview.md     # lý thuyết Double DQN / REINFORCE+baseline+entropy / AC+GAE
        ├── 02_evaluation_metrics.md
        └── 03_results.md             # 3 phép so sánh: baseline-3, improved-3, từng cặp trước/sau
```

## Tài liệu

- [`docs/baseline/01_system_overview.md`](./docs/baseline/01_system_overview.md) — tổng quan hệ thống, kiến trúc, luồng dữ liệu
- [`docs/baseline/02_evaluation_metrics.md`](./docs/baseline/02_evaluation_metrics.md) — giải thích các chỉ số đánh giá
- [`docs/baseline/03_results.md`](./docs/baseline/03_results.md) — kết quả baseline và phân tích
- [`docs/improved/01_system_overview.md`](./docs/improved/01_system_overview.md) — lý thuyết từng cải tiến
- [`docs/improved/02_evaluation_metrics.md`](./docs/improved/02_evaluation_metrics.md) — 3 cách so sánh (baseline-3, improved-3, trước/sau)
- [`docs/improved/03_results.md`](./docs/improved/03_results.md) — kết quả improved và so sánh trước/sau
