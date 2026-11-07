# Bước 07 — Threshold Calibration (phương án A)

## Quy tắc cố định
- Đầu vào model: `z_gap`, `z_intraday`, `z_range`, `z_volume`.
- Chỉ chấm Val-calibrate: 962 phiên, 13 mã × 74 ngày, 2025-09-11 đến 2025-12-23.
- IF (mô hình chính), AE và LOF (đối chiếu). Không fit/refit.
- Mỗi mô hình tính ngưỡng chung cho tất cả 13 mã: phân vị 99% của 962 điểm.
- `score > threshold` => `prediction=1`; ngược lại 0. Không dùng `predict()` mặc định.
- Chưa dùng Test, proxy hoặc news; chưa lựa chọn lại mô hình/ngưỡng.

## Đối chiếu checkpoint trên Val-select
```
model  n_val_select  max_abs_score_diff  tolerance  passed
   IF          1430                 0.0    0.00001    True
   AE          1430                 0.0    0.00050    True
  LOF          1430                 0.0    0.00001    True
```

## Ngưỡng và số cảnh báo trên Val-calibrate
```
model       role  quantile   threshold              rule  n_calibration  n_alert_calibration  alert_rate_calibration_pct     score_min  score_median   score_max
   IF       main      0.99 0.696300845 score > threshold            962                   10                  1.03950104   0.357107424   0.406227023 0.759867928
   AE comparison      0.99  6.43437588 score > threshold            962                   10                  1.03950104 0.00102253305   0.263619095  14.9430628
  LOF comparison      0.99  1.51219421 score > threshold            962                   10                  1.03950104   0.957469106    1.04426646  2.14214754
```

## Đồng thuận cảnh báo trong Val-calibrate
```
  pattern  n_sessions
     none         946
  IF_only           3
  AE_only           2
 LOF_only           2
    IF_AE           1
   IF_LOF           1
   AE_LOF           2
all_three           5
```

## Cảnh báo chuyên môn
- Tỷ lệ 1% là mức ngân sách cảnh báo thiết kế, không phải tỷ lệ bất thường thật hoặc Accuracy/Recall.
- Ngưỡng phân vị tính trên Validation hiệu chỉnh; không được quay lại tinh chỉnh IF/AE/LOF sau khi xem tập này.
- Các ngày và mã có cảnh báo chỉ được diễn giải mô tả OHLCV; không gán sự kiện/news ở Thí nghiệm 1.
- Ngưỡng theo toàn bộ mẫu (pooled). Một số mã có thể có nhiều cảnh báo hơn; đây không phải lỗi tự động.
- Các mô hình sử dụng các thang điểm riêng; không so sánh trực tiếp trị số threshold giữa IF, AE và LOF.
- File joblib khác phiên bản scikit-learn cần đối chiếu điểm Val-select; bước này đã đối chiếu thành công.

## Tài liệu được giữ cho Bước 08
- `07_thresholds.json`: ngưỡng đóng băng cho cả ba mô hình.
- `07_calibration_scores.csv`: điểm và dự đoán trên Val-calibrate (không phải Test).
- `07_run_config.json`: dấu vết đầu vào, checkpoint, thời gian và kiểm tra.
