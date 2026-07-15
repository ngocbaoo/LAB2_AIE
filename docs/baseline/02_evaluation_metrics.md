# Chỉ số đánh giá

Đánh giá Actor-Critic đòi hỏi nhiều hơn một con số "reward cuối cùng" — cập
nhật online mỗi bước có thể học nhanh nhưng dao động, nên hệ thống theo dõi
các chỉ số sau (tính trong `backend/train_all.py`, hiển thị ở bảng trong
`frontend/index.html`).

## 0. Vì sao chạy 5 seed thay vì 1?

Một lần chạy duy nhất không phân biệt được "học được thật" với "lần này may
mắn hơn" — nhất là với môi trường nhiễu cao như `TradingEnv`. `train_all.py`
chạy Actor-Critic trên **5 seed cố định** (42–46), mỗi seed bootstrap một tập
đoạn giá SPY thật khác nhau, và báo cáo **trung bình ± độ lệch chuẩn giữa các
seed** cho mọi chỉ số bên dưới, thay vì một con số đơn lẻ. Đường trung bình
trên biểu đồ đi kèm dải mờ ±1 std để thấy được mức độ dao động giữa các lần
chạy, không chỉ dao động trong một lần chạy.

## Chỉ số bổ sung: Tỷ lệ hội tụ (`solved_rate`)

Tỷ lệ trong 5 seed mà thuật toán đạt ngưỡng hội tụ (định nghĩa ở mục 3). Ví
dụ `solved_rate = 0.6` nghĩa là 3/5 seed hội tụ trong 400 episode, 2/5 thì
không — bản thân độ ổn định của việc *có hội tụ được hay không* cũng là một
khía cạnh cần so sánh, tách biệt với "hội tụ nhanh cỡ nào" (chỉ tính trên các
seed đã hội tụ).

## 1. Episode reward (`rewards`)

Tổng reward thô của từng episode = tổng **% lãi/lỗ** tích luỹ qua 200 bước
giao dịch trong episode đó (có thể âm). Đây là dữ liệu gốc, nhiễu cao — dùng
để vẽ đường mờ phía sau trên biểu đồ (bật bằng checkbox "Hiện reward thô").

## 2. Moving average (`moving_avg`, cửa sổ 20 episode)

Trung bình trượt 20 episode gần nhất. Đây là đường chính trên biểu đồ vì phản
ánh xu hướng học thực sự thay vì nhiễu từng episode — đặc biệt quan trọng ở
môi trường này vì reward episode-to-episode dao động rất lớn (mỗi episode là
một chuỗi giá tổng hợp khác nhau).

## 3. Episode hội tụ (`solved_at_episode`, trung bình qua các seed đã hội tụ)

Episode đầu tiên mà moving average vượt ngưỡng **2.0% lãi/episode**. Ngưỡng
này được chọn dựa trên một chính sách tham chiếu đơn giản — "theo dấu return
gần nhất" (long nếu return trước dương, short nếu âm) — đạt trung bình ~7.0%
lãi/episode (nhưng std rất cao, ~15.8, vì tín hiệu momentum yếu và nhiễu lớn),
trong khi chính sách "không làm gì" (Hold liên tục) luôn cho đúng 0%. Ngưỡng
2.0% nằm giữa hai mốc đó: đủ cao để loại trừ chính sách trung lập/ăn may
ngẫu nhiên, nhưng không đòi hỏi agent bắt được toàn bộ edge lý thuyết.

→ Đo **sample efficiency theo số episode**: thuật toán nào cần ít lần thử hơn
để học được chính sách tốt.

## 4. Số bước môi trường để hội tụ (`env_steps_to_solve`)

`(solved_at_episode + 1) × 200` — vì mỗi episode trong `TradingEnv` luôn dài
đúng 200 bước cố định (khác CartPole, nơi reward trùng với số bước sống sót
nên có thể cộng dồn trực tiếp).

→ Đo **sample efficiency theo số tương tác thực với môi trường** — chỉ số này
quan trọng hơn số episode trong thực tế, vì mỗi bước môi trường có thể tốn
kém (dữ liệu thị trường thật, chi phí giao dịch thật...).

## 5. Reward trung bình 20 episode cuối (`final_avg_reward_last20`, mean ± std qua 5 seed)

Đánh giá **chất lượng chính sách cuối cùng** sau khi huấn luyện xong, độc lập
với tốc độ hội tụ — một thuật toán có thể hội tụ chậm nhưng đạt policy cuối
tốt hơn.

## 6. Độ lệch chuẩn 20 episode cuối (`final_std_reward_last20`)

Đo **độ ổn định (stability)** của chính sách đã hội tụ. Std thấp = chính sách
nhất quán qua các episode; std cao (hoặc đúng bằng 0) = chính sách còn dao
động nhiều, hoặc đã "chốt cứng" vào một hành động duy nhất (suy biến).

## 7. Reward tốt nhất (`best_episode_reward`)

Giá trị cao nhất từng đạt được trong toàn bộ quá trình huấn luyện — cho biết
"trần năng lực" chính sách có thể chạm tới, kể cả khi chưa ổn định.

## 8. Thời gian huấn luyện (`training_time_sec`)

Wall-clock time cho toàn bộ `n_episodes`, cùng phần cứng (đo bằng
`time.perf_counter()` trong `train_all.py`). Phản ánh **chi phí tính toán
thực tế**: Actor-Critic cập nhật online sau mỗi bước (200 lần gọi
optimizer/episode), nên chi phí tỷ lệ trực tiếp với số bước huấn luyện.

## Vì sao dùng từng ấy chỉ số?

| Câu hỏi muốn trả lời                          | Chỉ số dùng                          |
|-----------------------------------------------|----------------------------------------|
| Học nhanh hay chậm (theo số lần thử)?         | `solved_at_episode`                    |
| Học nhanh hay chậm (theo tương tác thực)?     | `env_steps_to_solve`                   |
| Chính sách cuối cùng tốt đến đâu?              | `final_avg_reward_last20`              |
| Chính sách cuối cùng ổn định đến đâu?          | `final_std_reward_last20`              |
| Tiềm năng tối đa của chính sách?               | `best_episode_reward`                  |
| Chi phí tính toán để đạt được điều đó?         | `training_time_sec`                    |

Không có chỉ số đơn lẻ nào kể hết câu chuyện — đó chính là lý do cần nhìn cả
bảng thay vì một con số duy nhất. Kết quả cụ thể và phân tích ở
[`03_results.md`](./03_results.md).
