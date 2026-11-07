# Bước 06.4 — Kiểm tra hành vi chung (phương án A)

IF là mô hình chính; AE và LOF là mô hình so sánh.

Train: 10,387 phiên.
Validation chọn cấu hình: 1,430 phiên.

## Đối chiếu checkpoint
model  max_abs_score_diff  tolerance                                                                                                                   model_file
   IF                 0.0    0.00001                             d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\05_model_training_A\05_if.joblib
   AE                 0.0    0.00050  d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_2_ae_validation_A\06_2_ae_seed42_lr001_epoch1495_best.pt
  LOF                 0.0    0.00001 d:\LUẬN VĂN\stock-news-anomaly-project\Stocks\notebooks\outputs\06_3_lof_validation_A\06_3_lof_baseline_k20_euclidean.joblib

## Biến đổi OHLCV có kiểm soát
Số phiên gốc: 156.
Số phiên biến đổi: 936.

            scenario model  n_cases  n_effective  n_with_larger_target_abs_z  score_increased_pct  score_increased_among_amplified_pct  median_delta_score  median_delta_percentile_pts
         gap_amplify    IF      156          156                         156              100.000                              100.000               0.136                       35.385
         gap_amplify    AE      156          156                         156               91.026                               91.026               1.007                       27.448
         gap_amplify   LOF      156          156                         156               96.154                               96.154               0.239                       41.958
intraday_close_shift    IF      156          156                         128               82.692                               98.438               0.011                        5.175
intraday_close_shift    AE      156          156                         128               75.641                               92.188               0.144                        8.112
intraday_close_shift   LOF      156          156                         128               64.103                               68.750               0.009                        4.091
         range_wider    IF      156          156                         156              100.000                              100.000               0.201                       42.063
         range_wider    AE      156          156                         156              100.000                              100.000              10.507                       45.559
         range_wider   LOF      156          156                         156              100.000                              100.000               0.813                       50.315
           volume_x4    IF      156          156                         144               92.949                               99.306               0.080                       25.140
           volume_x4    AE      156          156                         144               87.179                               88.889               0.993                       31.923
           volume_x4   LOF      156          156                         144               85.256                               88.889               0.172                       33.287
      volume_quarter    IF      156          156                         135               94.231                              100.000               0.091                       28.706
      volume_quarter    AE      156          156                         135               96.795                               98.519               1.442                       31.014
      volume_quarter   LOF      156          156                         135               94.231                               94.074               0.252                       35.839
         combination    IF      156          156                         156              100.000                              100.000               0.299                       46.503
         combination    AE      156          156                         156               98.718                               98.718              10.787                       43.392
         combination   LOF      156          156                         156              100.000                              100.000               1.331                       51.049

## Độ nhạy W / mức sàn MAD

Đã tính lại đặc trưng và huấn luyện lại IF, AE, LOF cho cả 5 cấu hình.

Mức sàn được học từ Train.
Không sử dụng Test hoặc Validation hiệu chỉnh.

         case   W  floor_multiplier model  n_train  n_val_select  spearman  top1_overlap_pct  top5_overlap_pct
  W20_floorx1  20               1.0    IF    10907          1430     0.793            66.667            69.444
  W20_floorx1  20               1.0   LOF    10907          1430     0.533            46.667            56.944
  W20_floorx1  20               1.0    AE    10907          1430     0.820            60.000            63.889
W60_floorx0.5  60               0.5    IF    10387          1430     0.979            80.000            91.667
W60_floorx0.5  60               0.5   LOF    10387          1430     0.910            73.333            84.722
W60_floorx0.5  60               0.5    AE    10387          1430     0.984            66.667            80.556
  W60_floorx1  60               1.0    IF    10387          1430     1.000           100.000           100.000
  W60_floorx1  60               1.0   LOF    10387          1430     1.000           100.000           100.000
  W60_floorx1  60               1.0    AE    10387          1430     1.000           100.000           100.000
  W60_floorx2  60               2.0    IF    10387          1430     0.926            53.333            76.389
  W60_floorx2  60               2.0   LOF    10387          1430     0.717            46.667            68.056
  W60_floorx2  60               2.0    AE    10387          1430     0.924            53.333            70.833
 W120_floorx1 120               1.0    IF     9607          1430     0.904            80.000            80.556
 W120_floorx1 120               1.0   LOF     9607          1430     0.670            66.667            75.000
 W120_floorx1 120               1.0    AE     9607          1430     0.925            80.000            77.778

## Giới hạn nghiên cứu
- Không có nhãn bất thường thật.
- Không suy ra Accuracy hoặc Recall từ tỷ lệ tăng điểm.
- Top 1% và Top 5% chỉ phục vụ phân tích.
- Số phiên Train khác nhau khi thay W.
- Phép biến đổi chưa kiểm tra quy tắc trần/sàn HOSE.
- Chưa lựa chọn cấu hình tối ưu.
- Chưa sử dụng Test hoặc Validation hiệu chỉnh.

Bước tiếp theo: 06.5 — So sánh tổng hợp IF, AE và LOF.