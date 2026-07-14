# Kết quả thực nghiệm

> Số liệu dưới đây lấy trực tiếp từ `frontend/data/results.json`, sinh bởi
> `backend/train.py` với `CartPole-v1`, 400 episode/thuật toán, seed 42, chạy
> một lần (không average nhiều seed) — đúng tinh thần "baseline", chưa phải
> kết quả đã tinh chỉnh kỹ (tuned).

## 1. Bảng chỉ số

| Thuật toán   | Episode hội tụ | Bước env để hội tụ | Reward TB (20 ep cuối) | Std (20 ep cuối) | Reward tốt nhất | Thời gian train |
|--------------|:--:|:--:|:--:|:--:|:--:|:--:|
| DQN          | chưa hội tụ | — | 57.3 | 36.9 | 186 | 94.6s |
| REINFORCE    | 236 | 15,095 | 81.6 | 23.9 | 500 | 111.0s |
| Actor-Critic | 342 | 17,071 | **198.7** | **20.9** | 500 | 129.5s |

("Hội tụ" = moving-average 20 episode ≥ 195, xem [`02_evaluation_metrics.md`](./02_evaluation_metrics.md).)

## 2. Đọc kết quả — đừng chỉ nhìn dòng cuối

Đây là phần thú vị nhất của bài baseline: **con số cuối cùng không kể hết câu
chuyện**.

### DQN: chưa từng hội tụ trong 400 episode

DQN không vượt ngưỡng 195 lần nào, dù `best_episode_reward = 186` cho thấy nó
*đã từng gần chạm ngưỡng*. Đây khớp với đặc tính đã biết của DQN "vani"
(không Double DQN, không Dueling, không Prioritized Replay):

- Cần **nhiều tương tác môi trường hơn** để Replay Buffer đủ đa dạng.
- ε-greedy suy giảm tuyến tính trong 250/400 episode đầu — với ngân sách 400
  episode, quá trình khai thác (exploitation) thực sự chỉ có ~150 episode để
  hội tụ.
- Q-learning bootstrapping (học từ chính ước lượng của mình qua Target
  Network) vốn nổi tiếng kém ổn định hơn Policy Gradient/Actor-Critic ở giai
  đoạn đầu huấn luyện.

→ Đây là hạn chế thật của thuật toán ở cấu hình baseline, không phải lỗi cài
đặt — cải thiện hợp lý: tăng `n_episodes`, Double DQN để giảm overestimation,
hoặc kéo dài `eps_decay_episodes` chậm hơn.

### REINFORCE: hội tụ sớm rồi... thoái lui

REINFORCE là thuật toán *duy nhất* chạm ngưỡng 195 sớm nhất về số episode
(236), nhưng `final_avg_reward_last20` cuối cùng chỉ còn 81.6 — nghĩa là sau
khi đạt đỉnh, chính sách **suy giảm trở lại** trong các episode còn lại.

Đây là minh hoạ sách giáo khoa cho nhược điểm cốt lõi của Policy Gradient
thuần: gradient ước lượng bằng return cả episode có **phương sai rất cao**;
một vài episode return bất thường có thể đẩy tham số chính sách đi quá xa,
làm mất chính sách tốt vừa học được ("catastrophic policy update"), không có
cơ chế nào (buffer, trust region, clipping) để hãm lại.

### Actor-Critic: hội tụ muộn hơn nhưng ổn định nhất

Actor-Critic hội tụ muộn nhất về episode (342) — chậm hơn REINFORCE — nhưng:

- Là thuật toán duy nhất có `final_avg_reward_last20` (198.7) **vẫn ở trên
  ngưỡng hội tụ tại thời điểm kết thúc training**, tức là chính sách cuối
  cùng thực sự ổn định, không "học rồi quên" như REINFORCE.
- Có `final_std_reward_last20` thấp nhất (20.9) trong ba thuật toán — đúng
  như kỳ vọng lý thuyết: Critic cung cấp baseline V(s) làm giảm phương sai
  của gradient, nên cập nhật chính sách "êm" hơn REINFORCE.

→ Đánh đổi kinh điển: Actor-Critic tốn thêm một mạng Critic và nhiều phép
tính hơn mỗi bước (129.5s so với 111.0s của REINFORCE), nhưng đổi lại chính
sách hội tụ **và ở lại** vùng hội tụ.

## 3. Kết luận so sánh

| Tiêu chí                              | Thắng                |
|----------------------------------------|-----------------------|
| Hội tụ nhanh nhất (episode)            | REINFORCE (236)       |
| Chính sách cuối ổn định nhất            | Actor-Critic          |
| Ít tốn thời gian tính toán nhất         | DQN (94.6s, dù chưa hội tụ) |
| Không thuật toán nào tối ưu mọi mặt     | —                      |

Kết quả củng cố đúng trực giác lý thuyết trình bày ở
[`01_system_overview.md`](./01_system_overview.md) và
[`02_evaluation_metrics.md`](./02_evaluation_metrics.md): value-based (DQN)
mạnh về off-policy/data-reuse nhưng cần thời gian dài hơn để ổn định; pure
policy gradient (REINFORCE) học nhanh nhưng phương sai cao khiến chính sách
dễ dao động; Actor-Critic dùng Critic để giảm phương sai của Actor, đánh đổi
lấy chính sách hội tụ ổn định hơn với chi phí tính toán cao hơn một chút.

## 4. Giới hạn của baseline này

- Chạy 1 seed duy nhất — chưa đo được phương sai *giữa các lần chạy* (chỉ mới
  đo phương sai *trong* một lần chạy qua std 20 episode cuối). Để kết luận
  chắc chắn hơn nên chạy ≥5 seed và lấy trung bình/khoảng tin cậy.
- Hyperparameter (learning rate, kiến trúc mạng, lịch trình ε...) chưa được
  tinh chỉnh riêng cho từng thuật toán — dùng chung để so sánh công bằng,
  đổi lại DQN có thể chưa phát huy hết tiềm năng.
- Ngưỡng "hội tụ" 195 là quy ước riêng của baseline này (thấp hơn ngưỡng
  chính thức 475 của CartPole-v1) để phù hợp ngân sách 400 episode.
