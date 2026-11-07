# Bước 08 — Test Evaluation, phương án A

## Phạm vi
- Test: 2405 phiên, 13 mã, 185 ngày (2025-12-24 đến 2026-09-25).
- Đầu vào mô hình: z_gap, z_intraday, z_range, z_volume; W60 và floor Train gốc.
- IF chính; AE và LOF đối chiếu. Checkpoint đã đối chiếu trên Val-select.
- Không huấn luyện lại, không dùng Proxy P0 hoặc Test để tính Q99.

## Ngưỡng Q99 đã khóa trên Val-calibrate
- IF: 0.696300845396
- AE: 6.43437588215
- LOF: 1.51219420791

## Số phiên được mô hình cảnh báo trên Test (không phải số phiên bất thường thật)
- IF: 13/2405 (0.54%).
- AE: 21/2405 (0.87%).
- LOF: 31/2405 (1.29%).
- Có ít nhất một cảnh báo: 48/2405.
- Cả ba mô hình cảnh báo: 4/2405.

## Kết quả kiểm tra
- Đã đối chiếu SHA-256 đầu vào, checkpoint và ngưỡng trước khi mở Test.
- Đã tính lại bốn đặc trưng Test độc lập từ 60 phiên trước, không dùng thông tin tương lai.
- Không có giá trị thiếu ở đặc trưng, điểm hoặc nhãn; không trùng ticker/date.
- Các ngưỡng không bị thay đổi để ép tỷ lệ cảnh báo Test về 1%.
- Chưa đối chiếu Proxy P0/ROC-AUC/AP: đó là Bước 09.

## Đầu ra
- 08_test_predictions_A.csv: điểm và dự đoán của từng mã-ngày Test.
- 08_test_IF_alerts_A.csv: các phiên được IF cảnh báo.
- 08_locked_thresholds_used.csv: ngưỡng cố định đã sử dụng.
- 08_test_QA_manifest.json: dấu vết cấu hình/checksum và kiểm tra kỹ thuật.
