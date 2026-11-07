# Bước 05 — Model Training, phương án A

**Trạng thái:** đã huấn luyện IF, AE, LOF bằng Train; chấm điểm Validation select. **Chưa hiệu chỉnh ngưỡng và chưa đánh giá Test.**

## Dữ liệu
- Train: **10,387** phiên thuộc 13 mã.
- Validation select: **1,430** phiên.
- Train kết thúc: 2025-04-01; Val-select bắt đầu: 2025-04-02.
- Đầu vào: z_gap, z_intraday, z_range, z_volume; không thêm chuẩn hóa lần hai, không loại gap cực trị.

## Mô hình
- IF: 200 cây, 256 mẫu/cây, `contamination='auto'` nhưng chấm bằng `-score_samples`.
- LOF: 20 hàng xóm, Euclid, `novelty=True`; chấm Val-select bằng `-score_samples`.
- AE: 4→2→4, tanh, MSE, Adam lr=0.001; chạy 1000 epoch; checkpoint tốt nhất epoch **997** với Val-select MSE **1.226357**.
- **Cảnh báo:** AE chạm giới hạn số epoch; dừng sớm chưa xảy ra. Chưa được coi là đã hội tụ hoặc cấu hình tối ưu. Kiểm tra ngân sách huấn luyện tiếp trong Bước 06.

## Chẩn đoán (chưa phải đánh giá độ chính xác)
- Train có **14** phiên `|z_gap| > 30`, chiếm **52.18%** tổng bình phương của cả bốn đặc trưng.
- Trong Validation select, thành phần gap chiếm **29.76%** tổng bình phương lỗi tái tạo AE.
- Cần xem chi tiết lỗi AE theo từng thành phần trong `05_ae_error_by_feature.csv` và bảng điểm theo mã.
- Các kết quả trên **không chứng minh phiên nào là bất thường thật hoặc mô hình nào tốt nhất**.

## Kiểm tra đã thực hiện
- Train và Validation select tách đúng thứ tự thời gian.
- Cùng bốn đặc trưng cho ba mô hình; không fit `calibrate` hoặc Test.
- Các điểm IF/AE/LOF hữu hạn, lưu/tải lại khớp.
- Không tạo nhãn, không tạo ngưỡng, không tính Accuracy/Precision/Recall/F1 từ proxy.
- **Lưu ý:** LOF novelty chỉ dùng điểm trên dữ liệu mới; không so trực tiếp với `negative_outlier_factor_` trên Train.

## Bước tiếp theo
Bước 06: Thử sự ổn định của IF/AE, số hàng xóm LOF, ảnh hưởng các phiên gap cực trị và dạng biến động OHLCV trên Train + Validation select. Chỉ cân nhắc phương án B sau khi đánh giá đầy đủ phương án A.
