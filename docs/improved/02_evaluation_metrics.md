# Chỉ số đánh giá (bản improved)

Dùng đúng bộ chỉ số như [`docs/baseline/02_evaluation_metrics.md`](../baseline/02_evaluation_metrics.md)
để hai bên so sánh được trực tiếp — không có chỉ số nào mới. Chỉ có một câu
hỏi trọng tâm: **GAE có thực sự sửa được điểm yếu advantage 1-bước hay
không?**, ví dụ:

- `solved_rate` có tăng không (tỷ lệ seed hội tụ trong ngân sách 400 episode)?
- `solved_at_episode` có giảm (hội tụ nhanh hơn) không?
- `final_std_reward_last20` có giảm (chính sách cuối cùng ổn định hơn) mà
  `final_avg_reward_last20` không giảm theo (không đánh đổi mất chất lượng để
  lấy ổn định) không?
- `training_time_sec` tăng bao nhiêu để đổi lấy các cải thiện trên (nếu có) —
  GAE cần thu thập trọn episode rồi mới cập nhật, khác baseline cập nhật
  online mỗi bước.

## Về việc tổng hợp qua nhiều seed

Giống hệt cách làm ở baseline
([`docs/baseline/02_evaluation_metrics.md#0-vì-sao-chạy-5-seed-thay-vì-1`](../baseline/02_evaluation_metrics.md)):
cả baseline và improved chạy trên **cùng 5 seed** [42, 43, 44, 45, 46], để so
sánh trước/sau không bị nhiễu bởi việc chọn seed may mắn ở một bên. Biểu đồ
hiển thị đường trung bình ± dải std giữa 5 seed; bảng hiển thị `solved_rate`
(bao nhiêu trong 5 seed thực sự hội tụ) bên cạnh trung bình các chỉ số còn lại.

Số liệu cụ thể và phân tích nằm ở [`03_results.md`](./03_results.md).
