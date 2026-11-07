# Bước 09 — Trực quan hóa kiểm chứng phương án A (dữ liệu Test)

## Phạm vi và tính toàn vẹn
- Test: **2,405 mã–ngày**, **13 mã**, 185 ngày, 2025-12-24 đến 2026-09-25.
- Ngưỡng IF Q99 cố định = **0.696300845**, lấy trên Val-calibrate; không fit lại.
- IF: **13** phiên; AE: **21**; LOF: **31**.
- Đối chiếu SHA-256, đặc trưng và nhãn 0/1 với file Bước 08: **đạt**.
- P0 chỉ dùng đánh giá; không ảnh hưởng dự đoán/score của IF, AE, LOF.

## Năm nhóm hình có mục đích riêng
1. **13 timeline IF:** mã nào, phiên nào vượt Q99; kể cả những mã không có cảnh báo. Có thêm một bản tổng quan dạng mục lục.
2. **Feature Signature:** bốn giá trị z của chính 13 phiên IF=1; màu không biểu thị feature importance.
3. **Consensus & Overlap Matrix:** số phiên đồng thuận IF/AE/LOF, Jaccard chỉ trên tập có cảnh báo.
4. **ROC/PR:** điểm liên tục so với P0, KHÔNG sử dụng Q99; dùng lại hình hiện có nếu kiểm chứng hợp lệ.
5. **Score Distribution:** đối chiếu Val-calibrate và Test theo từng mô hình, không đặt đường ngưỡng; mỗi mô hình có thang riêng.
Nhóm biểu đồ nến/khối lượng chỉ dành cho 8 trường hợp đã được chọn theo quy tắc cố định, không vẽ đại trà.

## Những con số cần hiểu đúng
- Các cặp đồng thuận: IF–AE: 8 phiên, Jaccard 30.77% | IF–LOF: 7 phiên, Jaccard 18.92% | AE–LOF: 6 phiên, Jaccard 13.04%.
- Cả ba cùng cảnh báo: **4** phiên.
- Có ít nhất một mô hình cảnh báo: **48** phiên.
- Trong 13 cảnh báo IF, **13/13 phiên có z_range > 5**; đây là mốc mô tả mức rộng biên độ, KHÔNG phải quy tắc gán nhãn hay feature importance.
- Proxy P0=1: **210 / 2405** phiên; P0 đo return close-to-close và khối lượng tăng, không bao quát mọi kiểu bất thường OHLCV.
- IF: ROC-AUC với P0 **0.9063**, AP **0.5319** (KHÔNG phải độ chính xác với nhãn thật).
- AE: ROC-AUC với P0 **0.9239**, AP **0.5811** (KHÔNG phải độ chính xác với nhãn thật).
- LOF: ROC-AUC với P0 **0.7512**, AP **0.3745** (KHÔNG phải độ chính xác với nhãn thật).

## Điều không được suy luận
- Điểm IF cao hoặc vượt Q99 không xác nhận một phiên chắc chắn bất thường thật.
- IF/AE/LOF đồng thuận không phải bằng chứng ba mô hình đều đúng.
- Cùng P0 không chứng minh chính xác; P0 có phạm vi hẹp hơn bốn đặc trưng của mô hình.
- Sự dịch chuyển phân bố điểm giữa hai giai đoạn không tự chứng minh mô hình suy giảm hay lỗi dữ liệu.
- Không được thay ngưỡng/mô hình dựa trên Test vừa quan sát.

## Mẫu phiên OHLCV được vẽ
- 2 phiên IF=1, P0=0 (toàn bộ nhóm bất đồng này).
- 4 phiên IF=AE=LOF=1 (toàn bộ nhóm ba mô hình đồng thuận).
- 2 phiên P0=1, IF=0: ưu tiên một price-only và một volume-only có điểm IF cao nhất trong nhóm, để đối chiếu sát quyết định.
- Mỗi hình chỉ dùng 60 phiên trước và phiên đang xét, không thêm tin tức hay dữ liệu tương lai.

## Bảng đầu ra
- `tables/09_proxy_on_test.csv`: P0 được tính lại từ OHLCV, ghép với điểm Test.
- `tables/09_model_roc_pr_metrics.csv`: ROC-AUC / AP từ điểm liên tục.
- `tables/09_pairwise_consensus.csv`: giao nhau và Jaccard giữa ba mô hình.
- `tables/09_company_alert_counts.csv`: số cảnh báo theo 13 mã.
- `tables/09_feature_signature_IF13.csv`: đặc trưng 13 phiên IF cảnh báo.
- `tables/09_selected_OHLCV_cases.csv`: quy tắc lựa chọn và thông tin 8 phiên rà soát.
- `tables/09_score_distribution_summary.csv`: trung vị và phân vị điểm theo từng tập.
- `09_visualization_QA.json`: các kiểm tra chống sai nguồn và thay đổi Test.

## Quyết định phương pháp
**Giữ nguyên A, không thay IF/AE/LOF hay Q99.** Đây là trực quan hóa và đối chiếu kết quả đã khóa; không là phép huấn luyện hay cải thiện mô hình.