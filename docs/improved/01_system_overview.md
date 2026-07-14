# Tổng quan cải tiến thuật toán

Tài liệu này mô tả các cải tiến áp dụng lên từng thuật toán baseline (xem
[`docs/baseline/`](../baseline/01_system_overview.md)), *nhắm thẳng* vào điểm
yếu mà kết quả baseline đã chỉ ra
([`docs/baseline/03_results.md`](../baseline/03_results.md)):

| Thuật toán   | Điểm yếu quan sát được ở baseline                          | Cải tiến áp dụng |
|--------------|-------------------------------------------------------------|-------------------|
| DQN          | Chưa hội tụ trong 400 episode                               | **Double DQN** — giảm overestimation bias |
| REINFORCE    | Hội tụ sớm nhất rồi thoái lui (variance cao)                | **Learned baseline V(s) + entropy bonus** |
| Actor-Critic | Ổn định nhất nhưng hội tụ chậm nhất (advantage 1-bước bias cao) | **GAE(λ=0.95)** + entropy + gradient clipping |

Nguyên tắc chung khi thiết kế các cải tiến: **thay đổi tối thiểu, có chủ
đích** — chỉ sửa đúng phần lý thuyết gây ra điểm yếu đã quan sát, giữ nguyên
kiến trúc mạng, learning rate, và ngân sách huấn luyện (400 episode, 5 seed)
để phần chênh lệch kết quả có thể quy trực tiếp về cải tiến, không lẫn với
việc "cho học nhiều hơn" hay "mạng lớn hơn".

## 1. Double DQN (`backend/algorithms/dqn_double.py`)

DQN gốc dùng cùng một mạng (target network) để **vừa chọn** vừa **vừa đánh
giá** hành động tốt nhất ở trạng thái kế tiếp:

```
target = r + γ · max_a' Q_target(s', a')
```

Vì `max` tác động lên các ước lượng còn nhiễu, DQN có xu hướng **hệ thống
đánh giá quá cao** Q-value — nó cứ chọn hành động nào đang bị định giá lố,
rồi bootstrap tiếp trên chính sai số đó, gây học chậm/không ổn định. Double
DQN tách hai vai trò: mạng **online** (đang cập nhật) chọn hành động, mạng
**target** chỉ chấm điểm hành động đó:

```
a*     = argmax_a' Q_online(s', a')
target = r + γ · Q_target(s', a*)
```

Đây là toàn bộ khác biệt so với `dqn.py` — cùng kiến trúc mạng, cùng replay
buffer, cùng lịch trình ε, nên chênh lệch kết quả quy trực tiếp về việc sửa
Bellman target này.

## 2. REINFORCE + Baseline + Entropy (`backend/algorithms/reinforce_improved.py`)

Hai cơ chế độc lập, cùng nhắm vào "học nhanh nhưng thoái lui":

- **Learned baseline V(s)**: một mạng giá trị nhỏ được huấn luyện (MSE với
  chính return Monte-Carlo G_t) để trừ khỏi return trước khi nhân với
  log-probability: `advantage = G_t − V(s_t)`. Đây **không phải Actor-Critic
  thật** — vẫn là một update Monte-Carlo cho cả episode, không bootstrap sau
  từng bước — nhưng baseline làm giảm mạnh phương sai của gradient (kỳ vọng
  của V(s) không làm lệch gradient, chỉ giảm variance), nên vài episode return
  bất thường không còn kéo chính sách đi quá xa.
- **Entropy bonus**: cộng thêm `− entropy_coef · H(π(·|s))` vào loss để giữ
  chính sách còn ngẫu nhiên một chút lâu hơn — đánh thẳng vào nguyên nhân
  "hội tụ sớm rồi sụp": không có gì ngăn chính sách đẩy xác suất hành động về
  gần 0/1 ngay khi vừa tìm được một quỹ đạo tốt, và đó chính xác là điều xảy
  ra ở baseline.

## 3. Actor-Critic + GAE (`backend/algorithms/actor_critic_gae.py`)

Actor-Critic baseline dùng advantage thô nhất — bootstrap 1 bước:

```
advantage = r_t + γV(s_{t+1}) − V(s_t)
```

Phương sai thấp nhưng thiên vị (bias) cao: mọi thứ sau 1 bước đều "tin
tưởng" vào Critic — vốn cũng đang học. REINFORCE-with-baseline nằm ở cực còn
lại: return Monte-Carlo đầy đủ, phương sai cao nhưng không thiên vị. GAE(λ)
nội suy giữa hai cực bằng một tham số duy nhất:

```
δ_t     = r_t + γV(s_{t+1}) − V(s_t)                         (TD error 1 bước)
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

λ=0 chính là advantage 1-bước của baseline; λ=1 chính là advantage
Monte-Carlo đầy đủ (REINFORCE-with-baseline). File này dùng **λ=0.95** —
gần với Monte-Carlo nhưng vẫn bootstrap đủ để giảm variance.

Vì GAE cần toàn bộ chuỗi reward để tính lùi (backward), bản improved thu thập
**trọn 1 episode rồi mới cập nhật 1 lần** — khác với baseline cập nhật online
sau từng bước. Entropy bonus và gradient clipping (norm 0.5) được thêm vào vì
cùng lý do như ở REINFORCE improved: giữ khám phá và tránh update quá đà.

## 4. Đa seed cho toàn bộ so sánh

Cả baseline lẫn improved giờ đều chạy trên **5 seed cố định** (42–46) thay vì
1 seed như lần chạy đầu tiên — xem
[`02_evaluation_metrics.md`](./02_evaluation_metrics.md) để biết cách các
chỉ số được tổng hợp qua nhiều seed, và
[`03_results.md`](./03_results.md) cho 3 phép so sánh: 3 thuật toán baseline,
3 thuật toán improved, và từng cặp baseline-vs-improved theo thuật toán.
