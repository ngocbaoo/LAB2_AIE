# Chỉ số đánh giá (bản improved)

Dùng đúng bộ chỉ số như [`docs/baseline/02_evaluation_metrics.md`](../baseline/02_evaluation_metrics.md)
để hai bên so sánh được trực tiếp — không có chỉ số nào mới, chỉ có thêm
**3 cách nhìn** trên cùng một bộ chỉ số:

## 1. So sánh 3 thuật toán baseline (đã có)

Xem [`docs/baseline/03_results.md`](../baseline/03_results.md).

## 2. So sánh 3 thuật toán improved

Cùng bảng chỉ số (episode hội tụ, số bước để hội tụ, reward TB/std 20 episode
cuối, reward tốt nhất, thời gian train), nhưng cả ba đều đã áp dụng cải tiến
riêng của mình (Double DQN, REINFORCE+baseline+entropy, Actor-Critic+GAE).
Trả lời câu hỏi: "sau khi mỗi thuật toán đã được cải thiện đúng điểm yếu của
nó, thứ tự xếp hạng giữa chúng có đổi không?"

## 3. So sánh từng cặp baseline ↔ improved

Với **mỗi thuật toán riêng lẻ**, đặt baseline và improved cạnh nhau trên cùng
một biểu đồ + bảng. Đây là phép so sánh quan trọng nhất của tài liệu improved
— trả lời trực tiếp: "cải tiến có thực sự sửa được điểm yếu đã quan sát ở
baseline hay không?", ví dụ:

- DQN (baseline) so với Double DQN: `solved_rate` có tăng không?
- REINFORCE (baseline) so với REINFORCE+baseline+entropy: `final_avg_reward_last20`
  cuối cùng có còn *thấp hơn* mức đã đạt ở giữa quá trình train không (dấu
  hiệu "học rồi quên") — nếu cải tiến hiệu quả, khoảng cách này phải thu hẹp.
- Actor-Critic (baseline) so với Actor-Critic+GAE: `solved_at_episode` có
  giảm (hội tụ nhanh hơn) mà `final_std_reward_last20` không tăng (không đánh
  đổi mất ổn định để lấy tốc độ) không?

## Về việc tổng hợp qua nhiều seed

Giống hệt cách làm ở baseline
([`docs/baseline/02_evaluation_metrics.md#0-vì-sao-chạy-5-seed-thay-vì-1`](../baseline/02_evaluation_metrics.md)):
mỗi cấu hình (bao gồm cả 3 improved) chạy trên **cùng 5 seed** [42, 43, 44,
45, 46] như baseline, để so sánh trước/sau không bị nhiễu bởi việc chọn seed
may mắn ở một bên. Biểu đồ hiển thị đường trung bình ± dải std giữa 5 seed;
bảng hiển thị `solved_rate` (bao nhiêu trong 5 seed thực sự hội tụ) bên cạnh
trung bình các chỉ số còn lại.

Số liệu cụ thể và phân tích từng phép so sánh nằm ở
[`03_results.md`](./03_results.md).
