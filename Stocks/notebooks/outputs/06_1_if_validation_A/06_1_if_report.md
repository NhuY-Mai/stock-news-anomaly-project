# Thí nghiệm 1 — Bước 06.1: Isolation Forest baseline A

## Phạm vi
- Train: **10,387**; Validation chọn cấu hình: **1,430**; **loại Val calibrate trước khi phân tích và không sử dụng Test**.
- Bốn đặc trưng `z_gap, z_intraday, z_range, z_volume`, W=60, giữ nguyên dữ liệu và chuẩn hóa Bước 03.
- IF baseline **200 cây × 256 mẫu/cây, seed 42**, điểm cao = khác thường hơn (`-score_samples`).
- Không gắn nhãn thật và không đặt ngưỡng cảnh báo; top 1%, 2%, 5% chỉ là tập cố định để so thứ hạng.

## Kiểm tra tái lập Bước 05
{"old_scores_found": true, "max_abs_difference": 0.0, "matched_rows": 1430}

## Ổn định khi đổi seed ở cấu hình baseline (5 seed, 10 cặp)
- Spearman: trung vị **0.9899**, thấp nhất **0.9874**.
- Top 1%: trung vị trùng **86.7%**, thấp nhất **80.0%**.
- Top 5%: trung vị trùng **91.7%**, thấp nhất **87.5%**.
- Không có tiêu chuẩn thống kê chung để tuyên bố bao nhiêu % trùng là 'đạt'. Xem cả bảng chi tiết và theo từng ngày/mã.

## Các quan sát cần rà
- Phiên |z_gap|>30 trên Train: **14**, Val-select: **9**. Đều **được giữ nguyên**, không dùng để lọc Train.
- Bảng 06_1_feature_ablation.csv cho biết *mức thay đổi thứ hạng* khi bỏ từng đặc trưng, không phải độ chính xác hay tầm quan trọng nhân quả.
- Các ngày có nhiều mã nằm trong top 5% theo IF baseline (chỉ là mô tả):
|   top_rate | date       |   n_tickers |
|-----------:|:-----------|------------:|
|       0.05 | 2025-04-04 |          11 |
|       0.05 | 2025-04-09 |           8 |
|       0.05 | 2025-04-11 |           7 |
|       0.05 | 2025-04-03 |           6 |
|       0.05 | 2025-04-08 |           6 |
- Tháng 04/2025 chỉ có **247 / 1430** phiên Val-select (**17.3%**), nhưng có **57 / 72** phiên top 5% (**79.2%**). Đây là **tập trung theo thời gian** cần phân tích, không chứng minh lỗi hoặc một nguyên nhân thị trường cụ thể.

## Cần làm gì tiếp?
1. Đọc các bảng về seed, tham số và ablation để hiểu phương án A có ổn định hay không.
2. Kiểm tra những phiên/nhóm ngày có điểm cao; không tự suy ra sự kiện, thao túng hoặc chất lượng dự báo.
3. Kiểm tra phản ứng với biến đổi OHLCV hợp lệ ở Bước 06.4; so IF với AE/LOF ở Bước 06.5.
4. Chỉ sử dụng Val calibrate để lấy ngưỡng ở Bước 07 sau khi chốt cấu hình, không dùng Test để sửa.
