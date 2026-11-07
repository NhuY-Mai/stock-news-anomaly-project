# Thí nghiệm 1 — Proxy P0 và kiểm tra trước Test (chỉ dữ liệu phát triển)

## Quy tắc P0 đã thống nhất
- `price=1` khi `abs(log(C_t/C_prev)) > 2*Std(previous 20 log returns, ddof=1)`.
- `volume=1` khi `V_t > 3*Mean(previous 20 volumes)`.
- `P0=price OR volume`; lịch sử từng mã, shift(1), không dùng tương lai.
- P0 không phải ground truth; không bao quát Range-only, intraday reversal và volume collapse.

## Số liệu Proxy
```
split       val_role     n  proxy_positives  price_positives  volume_positives  price_and_volume  proxy_prevalence_pct
train not_applicable 10387              845              753               223               131              8.135169
  val         select  1430              149              137                28                16             10.419580
```

## ROC-AUC / AP / PR-AUC (chỉ Val-select)
```
model                    proxy  n_val_select  proxy_positives  proxy_prevalence  roc_auc_proxy  average_precision_proxy  pr_auc_trapezoid_proxy  val_select_q99_preview_only  n_val_select_top1  n_proxy_in_top1  max_baseline_score_diff  roc_auc_ci_lower  roc_auc_ci_upper  ap_ci_lower  ap_ci_upper
   IF P0_final_std20_mean20_OR          1430              149          0.104196       0.922895                 0.556061                0.553233                     0.721175                 15               10                      0.0          0.892513          0.946022     0.386248     0.693270
   AE P0_final_std20_mean20_OR          1430              149          0.104196       0.898218                 0.534810                0.532950                    13.489787                 15               13                      0.0          0.859017          0.926599     0.361333     0.680808
  LOF P0_final_std20_mean20_OR          1430              149          0.104196       0.794199                 0.490199                0.488863                     2.182332                 15               14                      0.0          0.713811          0.852326     0.266547     0.645144
```

## QA trước Test
- SHA-256 checkpoint IF/AE/LOF và Feature-config khớp Step07; chấm lại Val-select sai khác tối đa: {'IF': 0.0, 'AE': 0.0, 'LOF': 0.0}.
- Q99 Val-cal giữ nguyên; số nhãn 1 mỗi mô hình: {'IF': {'threshold': 0.6963008453957038, 'n_calibrate_alerts': 10}, 'AE': {'threshold': 6.434375882148741, 'n_calibrate_alerts': 10}, 'LOF': {'threshold': 1.5121942079067223, 'n_calibrate_alerts': 10}}.
- Validation-select đã được dùng trong phát triển mô hình; AUC/AP là kết quả phát triển, không phải Test độc lập.
- Chưa chấm Test, chưa huấn luyện lại, chưa điều chỉnh mô hình hay ngưỡng.
- Quy tắc P0 đã được đồng ý triển khai; khóa manifest trước khi mở Test.
- Không dùng AUC/AP so với P0 để tuyên bố độ chính xác thật hoặc thay Q99.