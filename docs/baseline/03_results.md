# Kết quả thực nghiệm (Baseline, 5 seed)

> Số liệu dưới đây lấy từ `frontend/data/results.json`, sinh bởi
> `backend/train_all.py`: mỗi thuật toán chạy trên **5 seed** [42, 43, 44, 45,
> 46] × 400 episode trên `CartPole-v1`, các chỉ số là **trung bình ± độ lệch
> chuẩn giữa 5 seed** (xem cách đọc ở
> [`02_evaluation_metrics.md`](./02_evaluation_metrics.md)).

## 1. Bảng chỉ số

| Thuật toán   | Tỷ lệ hội tụ | Episode hội tụ (mean) | Bước env để hội tụ (mean) | Reward TB 20 ep cuối (mean±std) | Reward tốt nhất (mean) | Thời gian train (s, mean) |
|--------------|:---:|:---:|:---:|:---:|:---:|:---:|
| DQN          | 4/5 | 358.8 | 30,441 | 206.7 ± 96.6  | 408.4 | 156.2 |
| REINFORCE    | 4/5 | 205.8 | 15,817 | 137.1 ± 39.6  | 464.8 | 172.3 |
| Actor-Critic | 2/5 | 369.0 | 18,410 | 154.7 ± 34.2  | 418.6 | 71.3  |

("Tỷ lệ hội tụ" = số seed trong 5 seed đạt moving-average 20 episode ≥ 195
trong 400 episode; "Episode hội tụ" chỉ tính trung bình trên các seed *đã*
hội tụ.)

## 2. Điều thú vị: kết quả đa-seed khác hẳn lần chạy 1-seed trước đó

Lần chạy baseline đầu tiên (1 seed, xem lịch sử commit) cho thấy DQN **chưa
từng hội tụ** trong 400 episode. Chạy lại đúng thuật toán, đúng seed 42, đúng
400 episode trong lần đa-seed này lại cho seed 42 **hội tụ ở episode 337**
với reward trung bình cuối 384.2. Cùng seed, cùng code, kết quả khác nhau
đáng kể — đây là một phát hiện kỹ thuật đáng ghi lại, không phải sai số ngẫu
nhiên nên bỏ qua:

`train_dqn` dùng `env.action_space.sample()` cho bước khám phá ε-greedy.
`set_seed()` (trong `backend/common.py`) chỉ seed `random`, `numpy` và
`torch` — **không** seed action space RNG riêng của Gym, vốn được khởi tạo từ
entropy hệ thống nếu không gọi `action_space.seed()` tường minh. REINFORCE và
Actor-Critic không dùng `action_space.sample()` (luôn lấy mẫu từ phân phối
`torch.distributions.Categorical`, được seed đầy đủ bởi `torch.manual_seed`),
nên chúng tái lập được tốt hơn giữa các lần chạy. Đây là một giới hạn tái lập
(reproducibility) thật của cài đặt DQN baseline, được ghi nhận trung thực thay
vì che giấu — và là một phần lý do khiến DQN có `final_avg_reward_last20` với
độ lệch chuẩn **giữa các seed** lớn nhất (96.6) trong ba thuật toán.

## 3. Đọc kết quả theo từng thuật toán

### DQN: hội tụ được (4/5 seed) nhưng dao động mạnh giữa các lần chạy

Không như ấn tượng "chưa bao giờ hội tụ" từ lần chạy 1-seed, trên 5 seed DQN
thực ra hội tụ ở 4/5 lần, chỉ seed 45 không hội tụ (final avg 109.3). Nhưng
`final_avg_reward_last20` dao động từ 109 đến 384 tùy seed — độ lệch chuẩn
96.6 là cao nhất trong ba thuật toán, phản ánh cả tính bất ổn định vốn có của
Q-learning bootstrapping lẫn vấn đề reproducibility ở mục 2.

### REINFORCE: hội tụ nhanh nhất, nhưng vẫn giữ đặc tính "học rồi có thể quên"

REINFORCE hội tụ sớm nhất trong ba thuật toán (trung bình episode 205.8,
tương đương ít hơn ~15,800 bước môi trường) — đúng như quan sát ở lần chạy
1-seed đầu tiên. `final_std_reward_last20` trung bình 36.1 không phải cao
nhất, nhưng đây là trung bình *trong từng seed* — chưa nói lên việc một số
seed có thể đã đạt đỉnh cao hơn rồi tụt lại trước khi kết thúc 400 episode
(hiện tượng đã thấy rõ ở seed 42 của lần chạy đơn: hội tụ ở ep 236 rồi giảm
còn 81.6). Đây chính xác là vấn đề mà bản improved
([`docs/improved/`](../improved/01_system_overview.md)) nhắm tới.

### Actor-Critic: tỷ lệ hội tụ thấp nhất (2/5) nhưng rẻ nhất về thời gian

Actor-Critic baseline chỉ hội tụ ở 2/5 seed trong ngân sách 400 episode —
thấp nhất trong ba thuật toán ở lần chạy đa-seed này — dù từng hội tụ ở seed
42 trong lần chạy đơn. Advantage 1-bước (bias cao, variance thấp) khiến tốc
độ học phụ thuộc nhiều vào việc Critic học đúng V(s) đủ sớm; với một số seed,
Critic học chậm hơn khiến Actor không có tín hiệu tốt kịp thời. Điểm cộng rõ
rệt: **thời gian huấn luyện thấp nhất hẳn** (71.3s so với ~156-172s của DQN/
REINFORCE) vì cập nhật online mỗi bước, không cần Replay Buffer hay đợi hết
episode mới update.

## 4. Kết luận baseline

Không thuật toán baseline nào thắng tuyệt đối:

| Tiêu chí                        | Thắng          |
|-----------------------------------|-----------------|
| Tỷ lệ hội tụ cao nhất              | DQN & REINFORCE (4/5, hòa) |
| Hội tụ nhanh nhất (episode)        | REINFORCE (205.8) |
| Rẻ nhất về thời gian tính toán     | Actor-Critic (71.3s) |
| Ổn định nhất giữa các seed (std thấp nhất) | Actor-Critic (final_avg std 34.2) |

Đây là điểm xuất phát cho các cải tiến ở
[`docs/improved/03_results.md`](../improved/03_results.md) — mỗi cải tiến
nhắm đúng điểm yếu quan sát được ở đây, và kết quả sau cải tiến không phải
lúc nào cũng tốt hơn theo mọi tiêu chí (xem phân tích Double DQN ở đó).

## 5. Giới hạn

- 5 seed đã tốt hơn nhiều so với 1, nhưng vẫn là cỡ mẫu nhỏ theo chuẩn nghiên
  cứu RL (thường dùng ≥10-30 seed cho kết luận thống kê chặt).
- DQN baseline có vấn đề reproducibility đã nêu ở mục 2 — nên hiểu độ lệch
  chuẩn giữa seed của DQN gồm cả biến thiên thuật toán lẫn biến thiên do
  action-space RNG chưa được seed.
- Hyperparameter dùng chung giữa ba thuật toán để so sánh công bằng, chưa
  tinh chỉnh riêng cho từng thuật toán.
