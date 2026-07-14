# Chỉ số đánh giá

So sánh DQN, Policy Gradient và Actor-Critic công bằng đòi hỏi nhiều hơn một
con số "reward cuối cùng" — ba họ thuật toán này có đặc tính hội tụ và độ ổn
định khác nhau, nên hệ thống theo dõi các chỉ số sau (tính trong
`backend/train.py::summarize`, hiển thị ở bảng trong `frontend/index.html`).

## 1. Episode reward (`rewards`)

Tổng reward thô của từng episode = số bước cột đứng vững trước khi đổ (tối đa
500). Đây là dữ liệu gốc, nhiễu cao — dùng để vẽ đường mờ phía sau trên biểu đồ
(bật bằng checkbox "Hiện reward thô").

## 2. Moving average (`moving_avg`, cửa sổ 20 episode)

Trung bình trượt 20 episode gần nhất. Đây là đường chính trên biểu đồ vì phản
ánh xu hướng học thực sự thay vì nhiễu từng episode — đặc biệt quan trọng với
REINFORCE, vốn có phương sai episode-to-episode rất lớn.

## 3. Episode hội tụ (`solved_at_episode`)

Episode đầu tiên mà moving average vượt ngưỡng **195** (chọn thấp hơn ngưỡng
chính thức 475 của CartPole-v1 vì ngân sách huấn luyện baseline chỉ 400
episode — mục tiêu là so sánh *tốc độ học tương đối*, không phải đạt SOTA).

→ Đo **sample efficiency theo số episode**: thuật toán nào cần ít lần thử hơn
để học được chính sách tốt.

## 4. Số bước môi trường để hội tụ (`env_steps_to_solve`)

Tổng reward (= tổng số bước) cộng dồn từ episode 0 đến `solved_at_episode`.

→ Đo **sample efficiency theo số tương tác thực với môi trường** — chỉ số này
quan trọng hơn số episode trong thực tế, vì mỗi bước môi trường có thể tốn
kém (robot thật, giả lập vật lý nặng...). Một thuật toán hội tụ ở episode
muộn hơn nhưng với episode ngắn (thất bại nhanh) có thể vẫn tốn ít bước hơn.

## 5. Reward trung bình 20 episode cuối (`final_avg_reward_last20`)

Đánh giá **chất lượng chính sách cuối cùng** sau khi huấn luyện xong, độc lập
với tốc độ hội tụ — một thuật toán có thể hội tụ chậm nhưng đạt policy cuối
tốt hơn.

## 6. Độ lệch chuẩn 20 episode cuối (`final_std_reward_last20`)

Đo **độ ổn định (stability)** của chính sách đã hội tụ. Std thấp = chính sách
nhất quán; std cao = chính sách còn dao động dù trung bình đã cao — thường
gặp ở REINFORCE do phương sai gradient lớn, và là điểm Actor-Critic được kỳ
vọng cải thiện nhờ baseline V(s) làm giảm phương sai.

## 7. Reward tốt nhất (`best_episode_reward`)

Giá trị cao nhất từng đạt được trong toàn bộ quá trình huấn luyện — cho biết
"trần năng lực" chính sách có thể chạm tới, kể cả khi chưa ổn định.

## 8. Thời gian huấn luyện (`training_time_sec`)

Wall-clock time cho toàn bộ `n_episodes`, cùng phần cứng (đo bằng
`time.perf_counter()` trong `train.py`). Phản ánh **chi phí tính toán thực
tế**: DQN tốn thêm chi phí replay buffer + sample batch mỗi bước, trong khi
REINFORCE/Actor-Critic cập nhật trực tiếp trên từng episode/bước nên đơn giản
hơn nhưng có thể cần nhiều episode hơn để bù lại.

## Vì sao dùng từng ấy chỉ số?

| Câu hỏi muốn trả lời                          | Chỉ số dùng                          |
|-----------------------------------------------|----------------------------------------|
| Học nhanh hay chậm (theo số lần thử)?         | `solved_at_episode`                    |
| Học nhanh hay chậm (theo tương tác thực)?     | `env_steps_to_solve`                   |
| Chính sách cuối cùng tốt đến đâu?              | `final_avg_reward_last20`              |
| Chính sách cuối cùng ổn định đến đâu?          | `final_std_reward_last20`              |
| Tiềm năng tối đa của chính sách?               | `best_episode_reward`                  |
| Chi phí tính toán để đạt được điều đó?         | `training_time_sec`                    |

Không có thuật toán nào thắng tuyệt đối trên mọi chỉ số — đó chính là lý do
cần nhìn cả bảng thay vì một con số duy nhất. Kết quả cụ thể và phân tích ở
[`03_results.md`](./03_results.md).
