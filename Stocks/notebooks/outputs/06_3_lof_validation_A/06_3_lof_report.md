# Bước 06.3 — Kiểm chứng Local Outlier Factor (phương án A)

**IF là mô hình chính. LOF là mô hình đối chiếu.** Không sử dụng Test, Val-calibrate hay nhãn proxy.
- Đầu vào: 10,387 Train; 1,430 Val-select; 4 đặc trưng chuẩn hóa W=60.
- Baseline LOF: k=20, Euclid, `novelty=True`; điểm càng lớn càng khác thường.
- Tái lập Bước 05: True ; max diff: 5.931091306976555e-08.
- Refit cùng cấu hình cho điểm trùng nhau: max diff = 0.0.

## So sánh cấu hình với baseline (chỉ độ nhất quán)

| k | Khoảng cách | Spearman | Trùng Top 1% | Trùng Top 5% |
|---:|---|---:|---:|---:|
| 10 | euclidean | 0.8452 | 80.0% | 86.1% |
| 20 | euclidean | 1.0000 | 100.0% | 100.0% |
| 35 | euclidean | 0.9320 | 53.3% | 86.1% |
| 50 | euclidean | 0.8910 | 40.0% | 83.3% |
| 100 | euclidean | 0.8197 | 40.0% | 80.6% |
| 20 | manhattan | 0.9564 | 73.3% | 93.1% |

## Các dấu hiệu cần xem xét
- Trong Val-select: 9 phiên có |z_gap| > 30, trong đó 9 phiên thuộc LOF Top 5%.
- Top 5% có 58/72 phiên ở tháng 04/2025 (chưa có diễn giải nguyên nhân).
- Trung bình tỷ lệ hàng xóm cùng mã: 8.21% (LOF dùng một mô hình gộp 13 mã).
- Các mốc Top 1/2/5% và |z_gap| > 30 chỉ phục vụ kiểm tra, KHÔNG phải ngưỡng bất thường chính thức.

## Quyết định ở giai đoạn này
- Giữ LOF k=20, Euclid làm baseline để so sánh, CHƯA chốt cấu hình cuối.
- Không chọn k/metric chỉ dựa vào Spearman hoặc độ trùng Top-k.
- Tiếp tục Bước 06.4 (thử thay đổi OHLCV hợp lệ, W, mức sàn) sau khi tổng hợp LOF với IF/AE.
- Không huấn luyện lại bằng Val-calibrate hoặc Test; chưa tính ngưỡng cảnh báo.

## Các cảnh báo phương pháp
- Hàng xóm của LOF nằm trong Train của 13 mã gộp, không nhất thiết thuộc cùng mã.
- Điểm LOF cho Val-select là novelty scoring; điểm trên Train nếu dùng negative_outlier_factor_ không so trực tiếp.
- LOF nhạy với cách đo khoảng cách và cách chuẩn hóa; điểm lớn không tự chứng minh bất thường thật.
