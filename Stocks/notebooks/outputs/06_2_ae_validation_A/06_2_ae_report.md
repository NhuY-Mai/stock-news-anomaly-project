# Bước 06.2 — Kiểm chứng Autoencoder, phương án A

**Vai trò:** AE là mô hình đối chiếu; Isolation Forest vẫn là mô hình chính.
- Train: **10,387**; Validation chọn cấu hình: **1,430**; không dùng Validation hiệu chỉnh hay Test.
- Đầu vào 4 đặc trưng W=60, không bỏ phiên gap cực trị, không thêm scaler hay chuyển B.
- Kiểm tra lại checkpoint Bước 05: {'previous_checkpoint_found': True, 'previous_scores_found': True, 'recomputed_val_mse': 1.2263565243163204, 'max_abs_score_diff': 1.943359379197318e-06}.

## Bảng chạy
```
 seed     lr  max_epochs  epochs_executed  best_epoch  best_val_mse  restored_val_mse  restored_train_mse stop_reason   seconds  best_epoch_close_to_limit
    7 0.0010        1500             1500        1500       1.03623           1.03623             1.78004  max_epochs 302.66917                       True
   19 0.0010        1500             1500        1500       1.04008           1.04008             1.79007  max_epochs 181.44835                       True
   42 0.0010        1500             1500        1495       1.03729           1.03729             1.77714  max_epochs 180.80387                       True
   73 0.0010        1500             1500        1491       1.03909           1.03909             1.78226  max_epochs 177.90303                       True
  101 0.0010        1500             1500        1496       1.03835           1.03835             1.78000  max_epochs 180.77219                       True
   42 0.0005        1500             1500        1492       1.44830           1.44830             2.25610  max_epochs 323.10817                       True
   42 0.0020        1500             1500        1499       0.93143           0.93143             1.31013  max_epochs 314.35886                       True
```

## Độ ổn định giữa seed (LR=0.001, checkpoint có Val-MSE thấp nhất)
```
        spearman  top1_overlap_pct  top5_overlap_pct
median    0.9999             100.0           98.6111
min       0.9997             100.0           98.6111
max       1.0000             100.0          100.0000
```

- Số phiên Train |z_gap|>30: **14**; chiếm **64.33%** tổng lỗi tái tạo Train tại checkpoint seed42.
## Tỷ trọng lỗi theo đặc trưng
```
    subset    feature   mse  share_squared_error_pct
     train      z_gap 4.582                   64.464
     train z_intraday 1.592                   22.397
     train    z_range 0.345                    4.855
     train   z_volume 0.589                    8.284
val_select      z_gap 0.812                   19.573
val_select z_intraday 2.318                   55.862
val_select    z_range 0.439                   10.574
val_select   z_volume 0.581                   13.991
```
## Độ nhạy learning rate cùng seed 42
```
 learning_rate  val_mse  spearman_vs_lr_main  top5_overlap_vs_lr_main_pct
        0.0010   1.0373               1.0000                     100.0000
        0.0005   1.4483               0.9963                      81.9444
        0.0020   0.9314               0.9973                      90.2778
```
## So checkpoint cùng một lần học (seed 42)
```
 epoch  val_mse  spearman_vs_best  top5_overlap_vs_best_pct
   300   3.0026            0.9810                   62.5000
  1000   1.2361            0.9972                   90.2778
  1500   1.0388            0.9997                   98.6111
```

## Các giới hạn trước khi kết luận
- MSE nhỏ hơn không chứng minh khả năng phát hiện bất thường tốt hơn; Validation select không có nhãn thật.
- Best checkpoint lựa chọn trên chính Validation select, nên các kết quả thứ hạng tại đây là phân tích phát triển, không phải đánh giá độc lập.
- Nếu nhiều seed còn chạm giới hạn epoch, phải ghi nhận chưa có bằng chứng hội tụ, không tự tuyên bố đã tối ưu.
- Xem độ ổn định Top-k và biến động giữa checkpoint; Top-k không phải ngưỡng cảnh báo.
- Chưa dùng proxy, Validation calibrate hay Test và không chọn phương án B.
- Sau bước này cần kiểm tra LOF 06.3, rồi chung OHLCV 06.4, cuối cùng mới so sánh ba mô hình ở 06.5.
## Độ dao động 100 epoch cuối (không phải độ chính xác)
```
 seed     lr  last100_val_mean  last100_val_std  last100_val_min  last100_val_max  last_val_mse
    7 0.0010           1.05016          0.00645          1.03623          1.06834       1.03623
   19 0.0010           1.05559          0.00765          1.04008          1.07591       1.04008
   42 0.0005           1.48105          0.02071          1.44830          1.52923       1.44990
   42 0.0010           1.04848          0.00649          1.03729          1.06711       1.03881
   42 0.0020           0.96231          0.02941          0.93143          1.08797       1.03792
   73 0.0010           1.05066          0.00609          1.03909          1.06326       1.04173
  101 0.0010           1.05134          0.00675          1.03835          1.06201       1.04025
```
- Chú ý: learning rate cao có thể có MSE checkpoint thấp nhất nhưng dao động lớn hơn ở những epoch cuối; không chọn LR chỉ vì best MSE.
