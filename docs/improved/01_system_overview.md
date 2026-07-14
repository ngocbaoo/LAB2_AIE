# Tổng quan cải tiến thuật toán

Tài liệu này mô tả cải tiến áp dụng lên Actor-Critic baseline (xem
[`docs/baseline/`](../baseline/01_system_overview.md)), *nhắm thẳng* vào điểm
yếu lý thuyết của advantage 1-bước: bias cao vì mọi thứ sau 1 bước đều "tin
tưởng" vào Critic — vốn cũng đang học.

Nguyên tắc thiết kế: **thay đổi tối thiểu, có chủ đích** — chỉ sửa đúng phần
lý thuyết gây ra điểm yếu đã quan sát, giữ nguyên kiến trúc mạng, learning
rate, và ngân sách huấn luyện (400 episode, 5 seed) để phần chênh lệch kết
quả có thể quy trực tiếp về cải tiến, không lẫn với việc "cho học nhiều hơn"
hay "mạng lớn hơn".

## Actor-Critic + GAE (`backend/algorithms/actor_critic_gae.py`)

Actor-Critic baseline dùng advantage thô nhất — bootstrap 1 bước:

```
advantage = r_t + γV(s_{t+1}) − V(s_t)
```

Phương sai thấp nhưng thiên vị (bias) cao. REINFORCE-with-baseline nằm ở cực
còn lại: return Monte-Carlo đầy đủ, phương sai cao nhưng không thiên vị.
GAE(λ) nội suy giữa hai cực bằng một tham số duy nhất:

```
δ_t     = r_t + γV(s_{t+1}) − V(s_t)                         (TD error 1 bước)
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

λ=0 chính là advantage 1-bước của baseline; λ=1 chính là advantage
Monte-Carlo đầy đủ. File này dùng **λ=0.95** — gần với Monte-Carlo nhưng vẫn
bootstrap đủ để giảm variance.

Vì GAE cần toàn bộ chuỗi reward để tính lùi (backward), bản improved thu thập
**trọn 1 episode rồi mới cập nhật 1 lần** — khác với baseline cập nhật online
sau từng bước. Entropy bonus và gradient clipping (norm 0.5) được thêm vào để
giữ khám phá và tránh update quá đà, cùng lý do các cải tiến RL khác thường
dùng khi chuyển từ online-1-bước sang cập nhật theo cả episode.

## Đa seed cho toàn bộ so sánh

Cả baseline lẫn improved đều chạy trên **5 seed cố định** (42–46) — xem
[`02_evaluation_metrics.md`](./02_evaluation_metrics.md) để biết cách các
chỉ số được tổng hợp qua nhiều seed, và
[`03_results.md`](./03_results.md) cho kết quả cụ thể baseline vs GAE.
