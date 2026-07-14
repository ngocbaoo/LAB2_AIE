# Kết quả thực nghiệm (Improved: Actor-Critic + GAE, 5 seed)

> Cùng nguồn `frontend/data/results.json`, cùng 5 seed [42, 43, 44, 45, 46] ×
> 400 episode, cùng `TradingEnv`, cùng ngân sách với baseline — chỉ khác
> advantage estimator (1-bước → GAE(λ=0.95)) và có thêm entropy bonus +
> gradient clipping. Xem cách đọc chỉ số ở
> [`02_evaluation_metrics.md`](./02_evaluation_metrics.md).

## 1. Bảng chỉ số

| Chỉ số | Actor-Critic (Baseline) | Actor-Critic + GAE |
|---|---:|---:|
| Tỷ lệ hội tụ | **5/5** | 4/5 |
| Episode hội tụ (mean ± std, seed đã hội tụ) | **25.4 ± 7.6** | 105.0 ± 93.1 |
| Bước env để hội tụ (mean) | **5,280** | 21,200 |
| Reward TB 20 ep cuối (mean ± std qua seed, %) | 0.93 ± 1.89 | 0.21 ± 3.29 |
| Độ lệch chuẩn 20 ep cuối (mean, *trong* seed) | 1.78 | **14.71** |
| Reward tốt nhất (mean, %) | **61.0** | 39.6 |
| Thời gian train (s, mean) | 342.9 | **230.0** |

## 2. Đọc theo từng seed — GAE không "chốt" vào Hold như baseline

| Seed | Baseline: reward TB / std cuối | GAE: reward TB / std cuối |
|---|---:|---:|
| 42 | 4.72% / 8.17 | −3.93% / 12.26 |
| 43 | −0.06% / 0.71 | **4.63% / 14.77** |
| 44 | 0.00% / **0.00** | **3.04% / 15.29** |
| 45 | 0.00% / **0.00** | −2.85% / 16.00 |
| 46 | 0.00% / **0.00** | 0.17% / 15.24 |

Khác biệt rõ nhất không nằm ở `final_avg_reward_last20` trung bình (hai bên
gần bằng nhau, 0.93% vs 0.21%) mà ở **`final_std_reward_last20`**: baseline
có 3/5 seed đóng băng ở std = 0 (đã suy biến hoàn toàn về Hold — xem
[`docs/baseline/03_results.md`](../baseline/03_results.md) mục 2), trong khi
**không seed GAE nào suy biến** — std cuối luôn ở mức 12–16, tức chính sách
vẫn đang chủ động giao dịch tới tận episode 400. Đây đúng là điều entropy
bonus được thiết kế để làm: giữ phân phối hành động đủ ngẫu nhiên để agent
không bao giờ "chốt cứng" vào một hành động duy nhất, kể cả khi hành động đó
(Hold) có vẻ an toàn trong ngắn hạn.

## 3. Cái giá phải trả: hội tụ chậm hơn và `solved_rate` thấp hơn

GAE không thắng tuyệt đối. So với baseline:

- **`solved_rate` giảm** (5/5 → 4/5): seed 45 không bao giờ vượt ngưỡng 2.0%
  trong suốt 400 episode — chính sách còn ngẫu nhiên (nhờ entropy) tiếp tục
  chịu nhiễu/phí giao dịch nhiều hơn một chính sách đã "chốt cứng", nên khó
  giữ moving-average ổn định trên ngưỡng.
- **`solved_at_episode` tăng mạnh** (25.4 → 105.0, std cũng tăng theo, 7.6 →
  93.1): GAE cần thu thập trọn episode rồi mới cập nhật 1 lần (thay vì mỗi
  bước như baseline), nên tốc độ học theo số episode chậm hơn hẳn — dù chi
  phí wall-clock/episode lại **thấp hơn** (230.0s so với 342.9s, vì ít lần
  gọi optimizer hơn).
- **`best_episode_reward` thấp hơn** (61.0% → 39.6%): baseline, dù bất ổn
  định, vẫn có những đợt "ăn may" đạt đỉnh cao hơn trước khi suy biến; GAE ổn
  định hơn nhưng trần năng lực quan sát được trong 400 episode thấp hơn.

## 4. Kết luận: GAE + entropy đổi "tốc độ hội tụ & đỉnh cao" lấy "không suy biến"

Đây là một đánh đổi thực, không phải một chiến thắng đơn giản — đúng tinh
thần của việc đo đa-seed thay vì tin vào một lần chạy may/rủi:

- Nếu mục tiêu là **không bao giờ để chính sách "chết" về trạng thái trung
  lập tuyệt đối** (một lỗi nghiêm trọng hơn trong ứng dụng thực tế — nghĩa là
  hệ thống ngừng giao dịch hoàn toàn dù thị trường vẫn có cơ hội), GAE +
  entropy là lựa chọn rõ ràng tốt hơn: 0/5 seed suy biến so với 3/5 ở
  baseline.
- Nếu mục tiêu là **hội tụ nhanh nhất có thể trong ngân sách hạn chế**,
  baseline online-1-bước vẫn thắng rõ rệt (25 episode so với 105 episode).

Không có phiên bản nào thắng tuyệt đối trên mọi tiêu chí — đúng bài học mà
việc đo trên nhiều seed luôn hướng tới: cải tiến lý thuyết đúng hướng (GAE
cân bằng bias/variance tốt hơn, entropy giữ khám phá) không tự động đồng
nghĩa với thắng trên mọi chỉ số trong một ngân sách huấn luyện cố định — ở
đây nó đổi tốc độ hội tụ và trần năng lực lấy sự bền vững của chính sách.
