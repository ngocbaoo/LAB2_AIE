# Kết quả thực nghiệm (Baseline, 5 seed, dữ liệu giá thật)

> Số liệu dưới đây lấy từ `frontend/data/results.json`, sinh bởi
> `backend/train_all.py`: Actor-Critic baseline chạy trên **5 seed** [42, 43,
> 44, 45, 46] × 400 episode trên `TradingEnv`, môi trường lấy dữ liệu từ
> **giá đóng cửa thật của SPY (ETF theo dõi S&P 500), 2015–2024** (xem
> `backend/trading_env.py` + `backend/fetch_data.py`) — mỗi episode là 1 đoạn
> 200 phiên liên tục được chọn ngẫu nhiên trong lịch sử giá thật đó. Các chỉ
> số là **trung bình ± độ lệch chuẩn giữa 5 seed** (xem cách đọc ở
> [`02_evaluation_metrics.md`](./02_evaluation_metrics.md)).

## 1. Bảng chỉ số

| Chỉ số | Giá trị |
|---|---:|
| Tỷ lệ hội tụ | **5/5** |
| Episode hội tụ (mean ± std, seed đã hội tụ) | 32.6 ± 14.4 |
| Bước env để hội tụ (mean) | 6,720 |
| Reward TB 20 ep cuối (mean ± std qua seed, %) | 11.37 ± 0.16 |
| Độ lệch chuẩn 20 ep cuối (mean, *trong* seed) | 11.18 |
| Reward tốt nhất (mean, %) | 43.8 |
| Thời gian train (s, mean) | 511.2 |

("Tỷ lệ hội tụ" = số seed trong 5 seed đạt moving-average 20 episode ≥ 2.0%
trong 400 episode.)

## 2. Khác biệt lớn nhất so với môi trường tổng hợp: không seed nào suy biến về Hold

Ở phiên bản môi trường **tổng hợp** (AR(1) log-return giả lập) trước đây,
baseline hội tụ rất nhanh nhưng 4/5 seed sau đó "chốt cứng" vào Hold (reward
và std cuối đúng bằng 0). Trên **dữ liệu SPY thật**, hiện tượng đó **không
còn xảy ra**:

| Seed | Episode hội tụ | Reward TB 20 ep cuối | Std 20 ep cuối |
|---|---:|---:|---:|
| 42 | 24 | 11.66% | 11.16 |
| 43 | 52 | 11.28% | 11.17 |
| 44 | 19 | 11.42% | 11.17 |
| 45 | 48 | 11.27% | 11.19 |
| 46 | 20 | 11.23% | 11.18 |

Cả 5/5 seed đều kết thúc episode 400 với reward trung bình ~11.2–11.7% và độ
lệch chuẩn nội-seed ~11.2 — nghĩa là chính sách **vẫn đang chủ động giao
dịch** đến tận cuối, không rơi vào trạng thái Hold-mãi-mãi như trên dữ liệu
tổng hợp. Lý do nhiều khả năng nằm ở chính bản chất dữ liệu: giá SPY thật có
volatility-clustering và các đoạn xu hướng/đảo chiều đa dạng hơn nhiều so với
1 quá trình AR(1) cố định tham số — môi trường liên tục "đẩy" chính sách ra
khỏi vùng an toàn tuyệt đối (Hold luôn có reward 0), nên gradient tại đó
không triệt tiêu hoàn toàn như trong môi trường tổng hợp đơn giản.

Đây là một minh chứng rõ về việc **kết luận rút ra từ môi trường mô phỏng có
thể không giữ nguyên khi chuyển sang dữ liệu thật** — vấn đề "suy biến về
Hold" của baseline là thật (có thể tái hiện được), nhưng mức độ nghiêm trọng
của nó phụ thuộc nhiều vào cấu trúc thống kê cụ thể của dữ liệu huấn luyện.

## 3. Chi phí tính toán

Actor-Critic cập nhật **online sau mỗi bước** (200 lần gọi optimizer/episode),
nên thời gian train (511.2s cho 400 episode) tỷ lệ trực tiếp với tổng số
bước huấn luyện — một chi phí tính toán thực tế của cách cập nhật 1-bước.

## 4. Kết luận

Trên dữ liệu SPY thật, Actor-Critic (advantage 1-bước) hội tụ nhanh (~episode
33), ổn định giữa các seed (std chỉ 0.16 điểm % trên reward trung bình cuối),
và **không suy biến về Hold** — kết quả tốt hơn kỳ vọng ban đầu, vốn dựa trên
thử nghiệm với môi trường mô phỏng tổng hợp (nơi baseline hay "chốt cứng"
vào Hold do thiếu cơ chế giữ khám phá). Trên dữ liệu thị trường thật, chính
bản chất nhiễu và biến động phức tạp của giá đã tự nhiên giữ cho chính sách
không rơi vào trạng thái trung lập tuyệt đối.

## 5. Giới hạn

- 5 seed đã tốt hơn nhiều so với 1, nhưng vẫn là cỡ mẫu nhỏ theo chuẩn nghiên
  cứu RL (thường dùng ≥10-30 seed cho kết luận thống kê chặt).
- Mỗi episode chỉ lấy 200 phiên giao dịch liên tục, bootstrap ngẫu nhiên
  trong ~2500 phiên lịch sử SPY (2015–2024) — nhiều episode giữa các seed có
  thể trùng lặp một phần giai đoạn thị trường, không hoàn toàn độc lập về
  mặt thống kê.
- Đây vẫn là một bài toán minh họa thuật toán (transaction cost đơn giản, chỉ
  giao dịch 1 tài sản, không có slippage/thanh khoản thực tế) — không phải
  một chiến lược sẵn sàng triển khai thật.
