# Bước 06.5 — So sánh tổng hợp IF (mô hình chính), AE và LOF



## Phạm vi và kiểm tra dữ liệu

- Đọc điểm IF/AE/LOF trên đúng **1,430** phiên Validation chọn cấu hình từ Bước 06.1–06.3.

- Mỗi mô hình sử dụng cùng 4 đặc trưng của phương án A, W=60 và mức sàn gốc.

- Bước 06.4 đã xác minh bản `RUN_AE_REFITS=True`: 15 tổ hợp, 21.450 điểm.

- Sai khác tối đa điểm baseline giữa Bước 06.4 và bảng điểm gốc: {'IF': 0.0, 'AE': 0.0, 'LOF': 0.0}.

- **Không dùng Validation hiệu chỉnh, Test hay proxy để chọn mô hình**. Không huấn luyện lại trong Bước 06.5.



## Đồng thuận và khác biệt

| top_n | model_a | model_b | spearman_all_val | intersection_count | overlap_pct | jaccard_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | IF | AE | 0.81 | 5 | 33.33 | 20.00 |
| 15 | IF | LOF | 0.63 | 7 | 46.67 | 30.43 |
| 15 | AE | LOF | 0.55 | 9 | 60.00 | 42.86 |
| 29 | IF | AE | 0.81 | 14 | 48.28 | 31.82 |
| 29 | IF | LOF | 0.63 | 12 | 41.38 | 26.09 |
| 29 | AE | LOF | 0.55 | 17 | 58.62 | 41.46 |
| 72 | IF | AE | 0.81 | 49 | 68.06 | 51.58 |
| 72 | IF | LOF | 0.63 | 46 | 63.89 | 46.94 |
| 72 | AE | LOF | 0.55 | 39 | 54.17 | 37.14 |



Đồng thuận ba mô hình và phiên chỉ do một mô hình ưu tiên:

| top_rate | top_n | all_three_count | any_model_count | if_only_count | ae_only_count | lof_only_count | at_least_two_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 15 | 4 | 28 | 7 | 5 | 3 | 13 |
| 0.02 | 29 | 9 | 53 | 12 | 7 | 9 | 25 |
| 0.05 | 72 | 30 | 112 | 7 | 14 | 17 | 74 |



- Có **4/15** phiên được cả ba cùng xếp Top 1%; **30/72** phiên được cả ba cùng xếp Top 5%.

- Đồng thuận không phải ground truth; bất đồng không chứng minh một mô hình sai.



### Các trường hợp IF ưu tiên riêng trong Top 1% (để rà, không tự gán nhãn)

| ticker | date | rank_if | rank_ae | rank_lof | z_gap | z_intraday | z_range | z_volume |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAB | 2025-04-09 | 4 | 28 | 26 | -9.10 | 7.31 | 10.26 | 2.07 |
| DHG | 2025-04-04 | 6 | 305 | 29 | -28.66 | 5.11 | 6.24 | 2.40 |
| ANV | 2025-05-06 | 9 | 83 | 49 | -13.16 | 5.35 | 7.42 | 1.97 |
| VNM | 2025-04-09 | 10 | 81 | 31 | -25.36 | 1.82 | 9.95 | 2.85 |
| MWG | 2025-04-04 | 12 | 313 | 74 | -19.15 | 3.79 | 5.16 | 3.32 |
| DHG | 2025-04-08 | 14 | 25 | 32 | 5.12 | -3.79 | 9.14 | 2.19 |
| IMP | 2025-04-09 | 15 | 49 | 35 | -13.96 | 3.94 | 9.21 | 0.49 |



## Sự tập trung theo thời gian và theo mã

- Tháng 04/2025 chiếm 247/1430 phiên Validation, xấp xỉ 17.3%.

- Trong Top 5%, số phiên tháng 04/2025: IF 57/72; AE 55/72; LOF 58/72.

- Các mô hình có thể cùng phản ứng với giai đoạn thị trường biến động, nhưng chưa xác nhận nguyên nhân.

- Tỷ lệ theo mã khác nhau không tự chứng minh thiên lệch. Cần diễn giải cùng nền lịch sử riêng mã.



## Phản ứng trước sáu tình huống OHLCV đã thử

Bảng: tỷ lệ tăng điểm trong các trường hợp thực sự tăng độ lớn của thành phần mục tiêu (không phải Recall).

| scenario | AE | IF | LOF |
| --- | --- | --- | --- |
| combination | 98.72 | 100.00 | 100.00 |
| gap_amplify | 91.03 | 100.00 | 96.15 |
| intraday_close_shift | 92.19 | 98.44 | 68.75 |
| range_wider | 100.00 | 100.00 | 100.00 |
| volume_quarter | 98.52 | 100.00 | 94.07 |
| volume_x4 | 88.89 | 99.31 | 88.89 |

- Các phép biến đổi chỉ được kiểm tra logic OHLC, chưa kiểm tra đầy đủ quy định biên độ/bước giá HOSE.

- Mẫu thử 156 phiên, mỗi mã–tháng có 2 phiên gốc. Cường độ một số biến đổi khá lớn.



## Độ nhạy W và mức sàn (Top 5% trùng với baseline của cùng thuật toán)

| case | AE | IF | LOF |
| --- | --- | --- | --- |
| W120_floorx1 | 77.78 | 80.56 | 75.00 |
| W20_floorx1 | 63.89 | 69.44 | 56.94 |
| W60_floorx0.5 | 80.56 | 91.67 | 84.72 |
| W60_floorx1 | 100.00 | 100.00 | 100.00 |
| W60_floorx2 | 70.83 | 76.39 | 68.06 |

- Đây là mức **khác biệt so với W60/floor×1**, không phải phép đo mô hình nào đúng nhất.

- W=20/60/120 dùng số phiên Train khác nhau, nên vừa thay đầu vào vừa thay lượng mẫu huấn luyện.

- Bộ dữ liệu đầu ra không đủ để khẳng định W=60 tối ưu; cần đối chiếu có kiểm soát nếu dùng W để chọn cấu hình.



## Bằng chứng từ từng thuật toán

- IF: Spearman trung vị giữa 5 seed ở 200 cây/256 mẫu = **0.9899**; Top 5% trùng trung vị = **91.7%**.

- AE: Spearman trung vị 5 seed = **0.9999**; Top 5% trùng trung vị = **98.6%**. 1.500 epoch vẫn chạm giới hạn; chưa chứng minh hội tụ.

- LOF: k=20 vs k=100 trùng Top 1% **40.0%**; LOF có độ nhạy cao với số hàng xóm.

- Ba thuật toán có thang điểm khác nhau: so thứ hạng, Top-k, hành vi, không so điểm số tuyệt đối.



## Quyết định phương pháp sau Bước 06.5

**Đề xuất có điều kiện: giữ phương án A làm baseline để trình bày và hiệu chỉnh ngưỡng:**

- IF **mô hình chính**: 200 cây, 256 mẫu/cây, seed42.

- AE **đối chiếu**: 4→2→4, Adam lr=0.001, MSE, seed42, checkpoint epoch1495.

- LOF **đối chiếu**: k=20, Euclid, novelty=True.

- Cửa sổ W=60, mức sàn Train ban đầu là cấu hình baseline, **không gọi tối ưu**.

- Không tự chọn IF tốt nhất theo tỷ lệ tăng điểm giả lập hoặc độ đồng thuận.



### Hai cảnh báo trước Bước 07/08

1. Nếu mục tiêu là **lựa chọn W và mức sàn tối ưu**, cần phép kiểm tra độc lập với chỉ số đồng thuận; các phép thử W hiện mới mô tả độ nhạy. Có thể giữ W=60 như phương án định trước và báo cáo độ nhạy.

2. **Test dự kiến từng được chạy ở notebook cũ**: phải rà mức ảnh hưởng tới thiết kế trước khi tuyên bố phép đánh giá Test độc lập.



Bước 07 chỉ hiệu chỉnh ba ngưỡng trên Validation hiệu chỉnh **sau khi người nghiên cứu chốt các lựa chọn**; không sử dụng tập này để sửa cấu hình.



## Dấu vết đầu vào

- `dev`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\03_feature_engineering\03_model_ready_development.csv` (sha256: `0f035c54869b7389548ce6ce0b574335ed149e8378b09351eaa8245b840f472a`)

- `if`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_1_if_validation_A\06_1_baseline_val_scores.csv` (sha256: `2f7d5b1f6fd0d09b8f7a22d6ffdb07b130937764802c34fa40db351218598b96`)

- `ae`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_2_ae_validation_A\06_2_ae_val_select_scores_seed42.csv` (sha256: `7491522a7e5e6b91ed345c234bae729b585a5d466786587f05b89a28e96f46b4`)

- `lof`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_3_lof_validation_A\06_3_lof_baseline_val_scores.csv` (sha256: `69109a8451202fc96e123fe2b34deb1fcd91bbeba4bb84e07d48b3e58895d91f`)

- `full_scores`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_4_joint_behavior_A_full\06_4_w_floor_scores.csv` (sha256: `d779cb91653d40a80952207a56a9f7428dbf8cb88bb27dfd1e5d48ba23fdb711`)

- `full_sensitivity`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_4_joint_behavior_A_full\06_4_w_floor_sensitivity.csv` (sha256: `a7ee04b62feab71aa7b925abbf020f05ef5d5bbc04f59deb7df08625d0e313f8`)

- `synthetic_pairs`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_4_joint_behavior_A_full\06_4_synthetic_pairs.csv` (sha256: `ea09dd56a530a0bf1d55b82f752e76f2ee3392fda25191ee189b3f3a308cc16c`)

- `synthetic_summary`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_4_joint_behavior_A_full\06_4_synthetic_summary.csv` (sha256: `ee4378f2ac16ae09080556e89c7452382ded55dcba25a34ab3e8687f9290c2fb`)

- `full_config`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_4_joint_behavior_A_full\06_4_run_config.json` (sha256: `39d3fa0ab4227a57ccd2778903bd77f4f6e3cc54c221e557dc96a1361c8819e3`)

- `if_stability`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_1_if_validation_A\06_1_stability_summary.csv` (sha256: `72f3f07628c43bee64ce710084f95585c5cd504de28e6e987419f1d83cc0cd83`)

- `ae_stability`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_2_ae_validation_A\06_2_ae_seed_pairwise.csv` (sha256: `9563c2e51cf4691195f7b2934277c27632acab66d69fea415dc24e2aa44c32f2`)

- `lof_sensitivity`: `d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_3_lof_validation_A\06_3_lof_config_comparison.csv` (sha256: `786b87a62cdfa4facf051713f2602df866b5dee229cc795864b3433634924d73`)
