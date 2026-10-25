# Stock News Anomaly Project

Dự án khóa luận xử lý tin tức tài chính, phân loại loại tin bằng PhoBERT + CNN và liên kết với bất thường cổ phiếu.

Mã nguồn, dữ liệu, notebook có output và báo cáo nằm trong [news_processing](news_processing/). Xem [hướng dẫn chạy và cấu trúc dự án](news_processing/README.md).

- Chuẩn bị dữ liệu: notebook 01–02.
- Phân loại loại tin: notebook 03–06; đã có kết quả huấn luyện, đánh giá và dự đoán.
- Sentiment và liên kết TN1: thư mục riêng, chưa triển khai.

Repository lưu cấu hình và tokenizer của model đã chốt. Tải riêng `best_model.pt` từ [thư mục model trên Drive](https://drive.google.com/drive/folders/1F6IBmR_TTgstNFS-uFfEpXJcyMTz4N55) vào `news_processing/models/news_type_phobert_cnn/runs/20261024_153932_380173/` nếu cần chạy lại suy luận tại máy. Môi trường Python, Java, cache và trọng số model không nằm trong Git.
