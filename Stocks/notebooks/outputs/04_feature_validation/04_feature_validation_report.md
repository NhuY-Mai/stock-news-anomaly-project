# Báo cáo Bước 04 — Kiểm tra đặc trưng (Thí nghiệm 1)

**Kết luận:** PASS về toán học và toàn vẹn dữ liệu; CẦN RÀ về phương pháp trước khi chốt mô hình.

- Cửa sổ: W=60 giá trị quá khứ cùng mã; MAD × 1.4826; mức sàn học trên Train (phân vị 10% của thang MAD dương).
- Đã kiểm tra toàn bộ 15,977 dòng về số học. Không dùng Validation calibrate hoặc Test để điều chỉnh thiết kế.
- Số phiên đủ điều kiện Train: 10,387; Validation select: 1,430.
- 14 phiên Train có |z_gap|>30; chiếm 52.18% tổng bình phương bốn trục của Train.

## Kết quả kiểm tra kỹ thuật
- PASS: 13 mã, khóa duy nhất, OHLCV hợp lệ và mốc thời gian không chồng lấn
- PASS: Bốn công thức thô, Close trước, đồng nhất thức lợi suất và phiên High=Low
- PASS: Rolling median/MAD W giá trị quá khứ của từng mã
- PASS: Fit 4 mức sàn chỉ trên Train, chuẩn hóa z, cờ MAD=0 / floor và tính đủ điều kiện
- PASS: File Development đúng tập con Train/Validation; không chứa Test
- PASS: Bảng Coverage và Feature Status khớp dữ liệu từng dòng
- PASS: Đối chiếu SHA256, OHLCV và nhãn của file Bước 02
- PASS: Tiền tố Train không bị ảnh hưởng bởi dữ liệu xảy ra sau Train

## Những điểm cần rà soát (không phải lỗi đã được chứng minh)
- Có 14 phiên Train với |z_gap| > 30; rà giá gốc và thang chuẩn hóa, không tự loại.
- Có 7 tổ hợp mã–đặc trưng sử dụng sàn >25% (Train + Val select).
- Tên trạng thái 'invalid_raw_feature' cho phiên đầu mỗi mã nên được diễn giải thành 'missing_previous_close' — OHLCV không sai.
- Không có kiểm chứng độc lập nguồn giá cho các phiên gap cực trị trong Bước 04.
- Tính độc lập của giai đoạn Test trước ảnh hưởng notebook cũ vẫn cần được xác minh trước đánh giá cuối.

## Thứ tự xử lý tiếp
1. Rà lại nguồn hoặc logic dữ liệu của các phiên gap cực trị nhưng chưa sửa hay xóa OHLCV.
2. Kiểm tra độ nhạy của mức sàn và ảnh hưởng theo mã; không chọn cấu hình bằng số cảnh báo hoặc F1 proxy.
3. Khi nghiên cứu mô hình chung IF, AE, LOF, đối chiếu trên Train/Validation select; chỉ sau khi khóa cấu hình mới tính ngưỡng trên Validation calibrate.
4. Không dùng kết quả Test để chọn thiết kế. Chỉ báo cáo kết quả Test theo phương án đã khóa và ghi giới hạn về tính độc lập nếu có.

**Lưu ý:** Các mốc rà soát |z_gap|>30 và 25% dùng sàn không phải ngưỡng bất thường hoặc tiêu chuẩn tối ưu.
