# Kết quả thực nghiệm (Improved, 5 seed)

> Cùng nguồn `frontend/data/results.json`, cùng 5 seed [42, 43, 44, 45, 46] ×
> 400 episode, cùng ngân sách với baseline — chỉ khác thuật toán. Xem cách
> đọc chỉ số ở [`02_evaluation_metrics.md`](./02_evaluation_metrics.md).

## 1. So sánh 3 thuật toán improved

| Thuật toán                     | Tỷ lệ hội tụ | Episode hội tụ (mean) | Bước env để hội tụ (mean) | Reward TB 20 ep cuối (mean±std) | Reward tốt nhất (mean) | Thời gian train (s, mean) |
|---------------------------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Double DQN                      | 2/5 | 296.0 | 23,665 | 116.2 ± 26.4  | 343.0 | 157.6 |
| REINFORCE + Baseline + Entropy  | 5/5 | 185.0 | 13,136 | 238.4 ± 134.2 | 500.0 | 144.0 |
| Actor-Critic + GAE              | 3/5 | 304.0 | 23,888 | 164.8 ± 38.1  | 444.2 | 74.9  |

Sau cải tiến, **REINFORCE + Baseline + Entropy thắng rõ rệt trên mọi tiêu
chí chính**: hội tụ ở cả 5/5 seed, nhanh nhất (episode 185), và là thuật
toán duy nhất đạt `best_episode_reward = 500.0` với **std = 0** — nghĩa là
cả 5 seed đều chạm trần reward tối đa của CartPole-v1 ít nhất một lần.

## 2. So sánh từng cặp Baseline ↔ Improved

### DQN → Double DQN: cải tiến làm *giảm* tỷ lệ hội tụ — một kết quả phản trực giác đáng phân tích

| Chỉ số | DQN (Baseline) | Double DQN |
|---|---:|---:|
| Tỷ lệ hội tụ | 4/5 | **2/5** |
| Episode hội tụ (mean) | 358.8 | 296.0 (nhanh hơn *khi* hội tụ) |
| Reward TB 20 ep cuối | 206.7 ± 96.6 | 116.2 ± 26.4 |
| Std giữa seed (final_avg) | 96.6 | **26.4 (thấp hơn hẳn)** |

Double DQN **không** cải thiện tỷ lệ hội tụ so với baseline trong thiết lập
này — thậm chí giảm từ 4/5 xuống 2/5. Đây là kết quả thật, không phải lỗi
cài đặt (xem `backend/algorithms/dqn_double.py`, đúng công thức Double
DQN chuẩn). Cách đọc hợp lý nhất:

- Double DQN sửa đúng thứ nó nhắm tới — **giảm phương sai giữa các seed**
  (96.6 → 26.4, thấp hơn hẳn) và giảm phương sai *trong* mỗi seed
  (`final_std_reward_last20` 59.4 → 40.9, xem bảng chi tiết trong JSON) —
  chính sách học được nhất quán hơn nhiều.
- Nhưng độ lạc quan (overestimation bias) mà Double DQN loại bỏ, trong môi
  trường đơn giản như CartPole với ngân sách khám phá ngắn (ε suy giảm trong
  250/400 episode), vô tình đóng vai trò một dạng "bonus khám phá ngầm" —
  Q-value bị thổi phồng khiến DQN gốc dám khai thác sớm các hành động có vẻ
  tốt. Loại bỏ độ lạc quan đó khiến Double DQN thận trọng hơn, cần nhiều
  episode/exploration hơn để tự tin vào một chính sách — điều ngân sách 400
  episode ở đây không đủ cấp cho một số seed.
- Đây là hiện tượng **đã được ghi nhận trong tài liệu RL**: Double DQN không
  phải lúc nào cũng thắng DQN gốc trên các bài toán đơn giản/không gian
  trạng thái nhỏ — lợi ích của nó rõ nhất trên các bài toán phức tạp hơn nơi
  overestimation bias thực sự gây sai lệch chính sách nghiêm trọng (Atari,
  trong paper gốc của van Hasselt et al.).

**Kết luận trung thực**: Double DQN ở đây là một đánh đổi — đổi tỷ lệ hội tụ
lấy độ ổn định — không phải một chiến thắng tuyệt đối. Đây chính là giá trị
của việc đo nhiều seed: một lần chạy đơn (vốn dùng seed 42 — nơi Double DQN
*không* hội tụ) sẽ tạo ấn tượng sai rằng cải tiến này "thất bại hoàn toàn",
trong khi thực ra nó thành công một phần, đúng như thiết kế lý thuyết dự
đoán, chỉ không phải phần tỷ lệ hội tụ.

### REINFORCE → REINFORCE + Baseline + Entropy: cải tiến thành công rõ rệt

| Chỉ số | REINFORCE (Baseline) | + Baseline + Entropy |
|---|---:|---:|
| Tỷ lệ hội tụ | 4/5 | **5/5** |
| Episode hội tụ (mean) | 205.8 | **185.0 (nhanh hơn)** |
| Reward TB 20 ep cuối | 137.1 ± 39.6 | **238.4 ± 134.2 (cao hơn hẳn)** |
| Reward tốt nhất (mean) | 464.8 | **500.0 ± 0 (mọi seed đều đạt trần)** |

Đây là cải tiến **thành công nhất** trong ba cải tiến. Cả tỷ lệ hội tụ, tốc
độ hội tụ, và chất lượng chính sách cuối cùng đều tăng. `final_avg` có std
giữa seed khá cao (134.2) — một số seed vẫn dao động mạnh trong 20 episode
cuối dù đã dùng baseline+entropy — nhưng điều quan trọng là **không seed nào
sụp đổ hoàn toàn về gần 0** như kiểu "học rồi quên" thấy ở baseline (seed 42
gốc: hội tụ ep 236 rồi tụt còn 81.6). Việc `best_episode_reward` đạt đúng
500.0 với std=0 ở mọi seed xác nhận: baseline V(s) + entropy bonus giải quyết
đúng vấn đề đã chẩn đoán — chính sách không còn "quên" khả năng đạt điểm tối
đa, dù vẫn còn dao động quanh đó.

### Actor-Critic → Actor-Critic + GAE: cải tiến đúng hướng, vừa phải

| Chỉ số | Actor-Critic (Baseline) | + GAE |
|---|---:|---:|
| Tỷ lệ hội tụ | 2/5 | **3/5 (tăng)** |
| Episode hội tụ (mean) | 369.0 | **304.0 (nhanh hơn)** |
| Reward TB 20 ep cuối | 154.7 ± 34.2 | 164.8 ± 38.1 |
| Thời gian train (s) | 71.3 | 74.9 |

GAE cải thiện đúng như lý thuyết dự đoán: tỷ lệ hội tụ tăng (2/5 → 3/5),
hội tụ nhanh hơn khi đã hội tụ (episode 369 → 304), với chi phí thời gian
gần như không đổi (71.3s → 74.9s, do vẫn chỉ 1 update/episode, chỉ thêm phép
tính GAE backward — rẻ). Đây là cải tiến **có tác dụng nhưng vừa phải** —
không đột phá như REINFORCE, hợp lý vì baseline Actor-Critic vốn đã có Critic
giảm variance sẵn; GAE chỉ tinh chỉnh thêm cán cân bias/variance của advantage
đã có, không sửa một lỗ hổng lớn như hai trường hợp kia.

## 3. Tổng kết ba phép so sánh

1. **Baseline 3 thuật toán** ([`docs/baseline/03_results.md`](../baseline/03_results.md)):
   không thuật toán nào thắng tuyệt đối; DQN & REINFORCE đồng hạng tỷ lệ hội
   tụ, Actor-Critic rẻ nhất nhưng hội tụ ít nhất.
2. **Improved 3 thuật toán** (mục 1 ở trên): REINFORCE+Baseline+Entropy vượt
   trội rõ rệt trên mọi tiêu chí.
3. **Từng cặp trước/sau** (mục 2): cải tiến không đồng đều — REINFORCE cải
   thiện mạnh nhất, Actor-Critic cải thiện vừa phải, còn Double DQN là một
   đánh đổi (ổn định hơn nhưng tỷ lệ hội tụ thấp hơn) chứ không phải cải
   thiện đơn thuần.

**Bài học chính**: "cải tiến lý thuyết đúng" không tự động đồng nghĩa với
"thắng trên mọi chỉ số" trong một ngân sách huấn luyện cố định — đặc biệt
khi cải tiến đó (như Double DQN) hoạt động bằng cách loại bỏ một thiên lệch
mà, tình cờ, đang có lợi cho tốc độ học trong bài toán cụ thể này. Đo trên
nhiều seed là điều giúp phát hiện ra sắc thái này thay vì kết luận vội từ một
lần chạy may/rủi.
