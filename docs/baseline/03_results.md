# Kết quả thực nghiệm (Baseline, 5 seed)

> Số liệu dưới đây lấy từ `frontend/data/results.json`, sinh bởi
> `backend/train_all.py`: Actor-Critic baseline chạy trên **5 seed** [42, 43,
> 44, 45, 46] × 400 episode trên `TradingEnv`, các chỉ số là **trung bình ±
> độ lệch chuẩn giữa 5 seed** (xem cách đọc ở
> [`02_evaluation_metrics.md`](./02_evaluation_metrics.md)).

## 1. Bảng chỉ số

| Chỉ số | Giá trị |
|---|---:|
| Tỷ lệ hội tụ | **5/5** |
| Episode hội tụ (mean ± std, seed đã hội tụ) | 25.4 ± 7.6 |
| Bước env để hội tụ (mean) | 5,280 |
| Reward TB 20 ep cuối (mean ± std qua seed, %) | 0.93 ± 1.89 |
| Reward tốt nhất (mean, %) | 61.0 |
| Thời gian train (s, mean) | 342.9 |

("Tỷ lệ hội tụ" = số seed trong 5 seed đạt moving-average 20 episode ≥ 2.0%
trong 400 episode.)

## 2. Điều thú vị nhất: hội tụ cực nhanh, nhưng rồi phần lớn "quên" luôn edge đã học được

Đây là phát hiện đáng chú ý nhất của baseline: **cả 5/5 seed đều vượt ngưỡng
hội tụ rất sớm** (episode 19–39, trung bình 25.4) — Actor-Critic tìm ra tín
hiệu momentum trong `TradingEnv` gần như ngay lập tức. Nhưng nhìn vào từng
seed ở cuối quá trình huấn luyện (episode 400):

| Seed | Episode hội tụ | Reward TB 20 ep cuối | Std 20 ep cuối |
|---|---:|---:|---:|
| 42 | 39 | **4.72%** | 8.17 |
| 43 | 22 | −0.06% | 0.71 |
| 44 | 19 | **0.00%** | **0.00** |
| 45 | 19 | **0.00%** | **0.00** |
| 46 | 28 | **0.00%** | **0.00** |

4/5 seed kết thúc với reward trung bình đúng **0.00%** và **độ lệch chuẩn
đúng 0** — nghĩa là chính sách đã suy biến hoàn toàn về **Hold liên tục**
(không giao dịch gì nữa). Chỉ seed 42 giữ được chính sách còn hoạt động
(std 8.17) và kết thúc có lãi (4.72%).

Đây không phải một chỉ số tệ theo nghĩa "chưa học được" — ngược lại, baseline
học *rất* nhanh (hội tụ ở episode ~25, sớm hơn nhiều so với các thí nghiệm
CartPole trước đó). Vấn đề là baseline **không có entropy bonus**: một khi
chính sách ngẫu nhiên (softmax) dần trở nên gần như xác định (peaked), và nếu
nó "chốt" vào đúng góc Hold (phần thưởng luôn bằng 0, không có phí giao dịch,
không rủi ro) thì gradient tại đó gần như triệt tiêu — không có tín hiệu nào
kéo chính sách ra khỏi điểm an toàn tuyệt đối này nữa. Đây chính xác là vấn
đề mà GAE + entropy bonus ở bản improved nhắm tới (xem
[`docs/improved/03_results.md`](../improved/03_results.md)).

## 3. Vì sao thời gian train của baseline lâu hơn GAE dù cùng ngân sách episode?

Baseline cập nhật **online sau mỗi bước** (200 lần gọi optimizer/episode),
trong khi GAE thu thập trọn episode rồi mới cập nhật **1 lần**/episode. Nhiều
lần gọi optimizer hơn khiến baseline chậm hơn về wall-clock (342.9s so với
230.0s của GAE) dù cùng 400 episode — một chi phí tính toán thực tế của cách
cập nhật 1-bước, độc lập với chất lượng chính sách học được.

## 4. Kết luận baseline

Actor-Critic baseline (advantage 1-bước, không entropy) học nhanh nhưng
**không ổn định lâu dài**: dễ tìm ra edge ban đầu, nhưng dễ suy biến về chính
sách trung lập tuyệt đối (Hold) sau đó vì thiếu áp lực giữ khám phá. Đây là
điểm xuất phát cho cải tiến GAE + entropy ở
[`docs/improved/03_results.md`](../improved/03_results.md).

## 5. Giới hạn

- 5 seed đã tốt hơn nhiều so với 1, nhưng vẫn là cỡ mẫu nhỏ theo chuẩn nghiên
  cứu RL (thường dùng ≥10-30 seed cho kết luận thống kê chặt).
- `TradingEnv` là môi trường tổng hợp (AR(1) log-return), không phải dữ liệu
  thị trường thật — dùng để minh họa sự khác biệt thuật toán, không phải để
  đánh giá một chiến lược giao dịch thực tế.
- Hyperparameter dùng chung giữa baseline và improved để so sánh công bằng,
  chưa tinh chỉnh riêng cho từng phiên bản.
