# News Processing

## Quy trình

- Chuẩn bị chung: EDA raw → clean dữ liệu.
- Phase 1: gán nhãn → huấn luyện/đánh giá `news_type` → dự đoán phần tin chưa có nhãn.
- Phase 2: làm sentiment riêng theo bài–ticker.
- Phase 3: ghép kết quả news với TN1.

## Cấu trúc

```text
news_processing/
├── notebooks/
│   ├── 01_EDA_data.ipynb
│   ├── 02_clean_news.ipynb
│   ├── phase_1_news_type/
│   │   ├── 03_prepare_annotation_sample.ipynb
│   │   ├── 04_prepare_model_data.ipynb
│   │   ├── 05_train_phobert_cnn.ipynb
│   │   └── 06_predict_news_type.ipynb
│   ├── phase_2_sentiment/
│   └── phase_3_tn1/
├── data/
│   ├── raw/news_raw.csv
│   ├── processed/news_clean.csv
│   ├── interim/                     # Log EDA và cleaning
│   ├── splits/news_type/             # train.csv, validation.csv, test.csv
│   └── annotation/news_type/
│       ├── labeling_guidelines.md
│       └── news_manually_labeled.csv
├── models/news_type_phobert_cnn/label_mapping.json
├── reports/news_type/               # Lý do chọn mẫu, quota nguồn, độ bao phủ
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Đang thực hiện

`data/annotation/news_type/news_manually_labeled.csv` là bản nhãn chính gồm 1.200 bài đã nhập từ file người gán. Đồng bộ sau clean chỉ cập nhật `content`; không sửa nhãn, trạng thái, ghi chú hay ticker. Ngoại lệ được người dùng phê duyệt là đợt chỉnh nhãn tin giá vàng `gold_price_industry_v1`, có đối soát theo ID trong `data/annotation/news_type/label_corrections.csv`. Trạng thái, ghi chú, trường tin và thứ tự dòng của đợt này vẫn giữ nguyên. File gốc ở Downloads không thay đổi.

Notebook 02 tách boilerplate theo nguồn và lưu bài không đạt vào `data/interim/news_excluded_records.csv`. Trong file annotation, content của các bài này để trống để tránh dùng nhầm danh sách tin hoặc nội dung sai làm thân bài. Notebook 03 đối chiếu ID, log hợp nhất và trạng thái gán nhãn để xác định những bài đủ điều kiện; chưa chia tập hoặc huấn luyện.

Đoạn cuối API CafeF không có ranh giới rõ được đánh dấu `UNRESOLVED_TRAILING_FRAGMENT` để xem lại, không tự cắt. Đây là nghi vấn chất lượng, không phải kết luận toàn bộ bài sai.

Các báo cáo trong `reports/news_type` ghi lại đợt chọn 800 bài trước khi sửa boilerplate, không phải phân bố nhãn hiện tại. Không chạy lại lấy mẫu để thay thế những bài đã gán.

Notebook 04 tạo ba tập train, validation và test khi chưa có; nếu đã có đủ ba file, giữ nguyên ID và thứ tự từng tập rồi cập nhật nhãn từ annotation. Thiếu một file hoặc thay đổi tập bài đủ điều kiện thì dừng, không tự chia lại. Trọng số lớp chỉ được tính từ train. Bộ nhãn huấn luyện vẫn có 7 lớp; `macro` được gộp vào `industry` khi ánh xạ.

### Nhất quán nhãn tin giá vàng

Theo `gold_price_industry_v1`, nội dung chính là cập nhật giá/thị trường vàng được gán `industry`, kể cả bản tin chỉ có giá mua–bán. Không đổi bài về lợi nhuận, hoạt động riêng của PNJ hay giá cổ phiếu chỉ vì có từ “giá vàng”. Việc sửa chỉ áp dụng danh sách ID đã đọc xác nhận, không tự gán bằng regex.

Đợt 24/10/2026 sửa 42 nhãn `mention_only` thành `industry`: 24 train, 9 validation, 9 test. Có 76 ứng viên được rà; 21 bài thị trường vàng đã đúng nhãn và 13 bài ngoài phạm vi/chưa rõ được giữ nguyên. Membership, nội dung, trạng thái và ghi chú không đổi; trọng số lớp và bảng phân bố được cập nhật. Nhãn held-out được đồng bộ theo cùng quy tắc, không dựa vào dự đoán hay điểm test.

Checkpoint và báo cáo cũ không bị ghi đè. Accuracy/F1 đã công bố thuộc phiên bản nhãn cũ, không phải kết quả mới. Đổi nhãn không đồng nghĩa mô hình đã tốt hơn; cần huấn luyện/đánh giá riêng và ghi đúng phiên bản. Gói ZIP Colab đã cập nhật dữ liệu mới; phiên Colab đang mở không tự nhận bản sửa. Không dùng `Run all` trên bản cũ để thay cho cập nhật dữ liệu.

## Huấn luyện loại tin

### Rà soát validation ngày 25/10/2026

Phiên bản nhãn hiện tại là `company_awards_listing_v2`. Đợt này chỉ sửa 6 nhãn đã được người gán xác nhận: STT 4, 8, 9, 10 trong báo cáo lỗi gốc thành `business_operations`; STT 11, 12 thành `governance_capital`. Các quyết định được lưu theo `news_id` trong `label_corrections.csv`, không suy nhãn từ dự đoán hoặc tự áp dụng bằng từ khóa cho các bài khác.

Ba bài STT 11, 13, 15 được cắt đúng đoạn mã hóa/liên hệ hoặc footer đã đối chiếu, giữ nguyên phần tin. Hàm sửa nội dung chỉ chấp nhận hash gốc hoặc hash sau sửa đã duyệt và có thể chạy lại an toàn. Notebook 02 tái tạo được các sửa này; annotation và văn bản model được đồng bộ từ bộ clean.

Hồ sơ `data/annotation/news_type/validation_revision.json` ghi 8 bài thay đổi (STT 11 có cả nhãn và nội dung), hash trước/sau và nguồn checkpoint. Người gán đã xác nhận STT 16, 18 cùng giữ `financial_results`, với lý do ở `resolved_reviews`; `pending_reviews` hiện rỗng. Xác nhận này không sửa CSV, nhãn, trạng thái hoặc membership. Train/test giữ nguyên byte; ID và thứ tự ba tập vẫn là 740/159/159; trọng số lớp từ train không đổi.

Checkpoint vẫn thuộc lần train `20261024_153932_380173` và chính sách nhãn cũ. Khi review, code dựng ngược các sửa đã duyệt để kiểm tra hash toàn bộ validation cũ, đồng thời kiểm tra train/test, mapping, kiến trúc, tài nguyên tách từ và cấu hình model. Đây là quyền đánh giá lại validation có đối soát, không phải bỏ kiểm tra hash hoặc tiếp tục huấn luyện.

Báo cáo validation sửa nhãn lưu riêng tại `reports/news_type/evaluations/validation_review_20261025_v2/` cùng `evaluation_config.json`. Không ghi đè báo cáo huấn luyện gốc hoặc dùng dự đoán cũ để giả lập suy luận mới. Việc xác nhận giữ nhãn STT 16, 18 không thay đổi đầu vào hay điểm validation đã tính; metadata của lần đánh giá trước vẫn được giữ theo thời điểm chạy. Không so trực tiếp điểm trước/sau sửa nhãn như mô hình tự cải thiện. Notebook vẫn chặn test nếu có bài chờ. Output test và hồ sơ bàn giao thuộc lần chạy ngày 25/10/2026 đã bật `EVALUATE_TEST = True`; sau đó cờ được đặt lại `False` để không tự đánh giá test khi chạy lại. Test đã được xem ở các phiên bản nhãn trước, nên không coi đây là một tập kiểm định cuối hoàn toàn chưa từng sử dụng.

Chạy `05_train_phobert_cnn.ipynb` sau notebook 04. Cài các gói trong `requirements.txt` và ưu tiên môi trường GPU với bản PyTorch hỗ trợ CUDA. Nếu không nhận GPU, notebook vẫn chạy CPU và thông báo thời gian huấn luyện có thể dài.

VnCoreNLP cần Java 8+. Có thể dùng Java sẵn có qua `JAVA_HOME`/PATH, hoặc Java portable trong `models/java/<thư mục runtime>`. Notebook chỉ cấu hình `JAVA_HOME` và PATH của kernel, không thay đổi biến môi trường Windows. Java portable được tải từ [Eclipse Adoptium](https://adoptium.net/installation/archives/) và kiểm tra SHA-256 trước khi giải nén. Sau khi cập nhật môi trường, Restart Kernel rồi chạy lại từ đầu.

Mô hình dùng PhoBERT-base-v2, bốn nhánh TextCNN và nhánh masked mean, ghép thành 640 đặc trưng trước khi phân loại 7 nhãn. VnCoreNLP tách từ trước BPE; Head–Tail giữ tối đa 254 token nội dung rồi thêm hai special token. Tỷ lệ giữ đầu bài là cấu hình khởi đầu, chưa phải kết quả tối ưu hóa.

Checkpoint được chọn theo Macro-F1 validation; test chỉ được đánh giá sau khi chốt cấu hình. Bản hiện tại dùng `review` và `EVALUATE_TEST = False`, giữ output test đã lưu. Chỉ bật cờ khi chủ động tái hiện đánh giá test với checkpoint đã chốt; khi huấn luyện hoặc khảo sát mới, giữ cờ `False`. Seed hỗ trợ tái lập trong cùng môi trường, không bảo đảm kết quả giống hệt giữa mọi phần cứng.

Notebook hiện mặc định `RUN_MODE = 'review'`, đọc lại đúng lần chạy `20261024_153932_380173` đã lưu trên Drive. Chế độ này không tạo optimizer, không huấn luyện hoặc khảo sát thêm, không ghi lại checkpoint/cấu hình/lịch sử/chỉ số gốc. Bất đồng đầu vào chỉ được chấp nhận qua hồ sơ validation đã duyệt ở trên; các kiểm tra khác vẫn giữ nguyên. Thiếu hoặc không khớp thì dừng, không tự chuyển sang train hay tự chọn model mới nhất.

Muốn huấn luyện mới, đổi `RUN_MODE = 'train'` và đặt `EVALUATE_TEST = False`. Giữ `DROPOUT = 0.5` và `RUN_PARAMETER_STUDY = False` nếu chưa khảo sát cấu hình khác. Mục 12 vẽ Accuracy/Macro-F1, ma trận nhầm lẫn và F1 từng nhãn trong cả hai chế độ; mục 13 xuất bảng rà lỗi validation. Khi khảo sát tắt, có thể bỏ mục 11: mục 12 tự xóa trạng thái khảo sát cũ trước khi tiếp tục. Khi bật khảo sát, mục 12 chỉ chấp nhận kết quả của khảo sát hiện tại đã hoàn tất và đúng dữ liệu/code.

Mỗi lần huấn luyện tạo một thư mục `runs/<RUN_NAME>` trong `models/news_type_phobert_cnn` và `reports/news_type`. Thư mục model lưu checkpoint tốt nhất, tokenizer, cấu hình backbone và cấu hình chạy; thư mục báo cáo lưu lịch sử train/validation và, khi bật đánh giá cuối, kết quả test. Git lưu cấu hình/tokenizer và báo cáo của lần chạy đã chốt `20261024_153932_380173`, kết quả đánh giá `validation_review_20261025_v2` cùng hai file dự đoán của notebook 06. Trọng số model, môi trường và cache được bỏ qua; checkpoint tải riêng từ Drive theo hướng dẫn ở README gốc. Dữ liệu gốc, dữ liệu clean và nhãn không bị ghi đè.

Notebook 05 có khảo sát sáu cấu hình learning rate, dropout và weight decay khi `RUN_PARAMETER_STUDY = True`. Mỗi cấu hình bắt đầu từ PhoBERT gốc, cùng split và seed; chọn theo Macro-F1 validation, rồi Accuracy và loss nếu bằng nhau. Khảo sát cần GPU và chỉ giữ checkpoint thắng trong `models/news_type_phobert_cnn/studies/<STUDY_NAME>`. Bảng so sánh, lịch sử từng cấu hình và biểu đồ lưu ở thư mục `studies` tương ứng trong báo cáo.

Biểu đồ gồm Loss/Accuracy/Macro-F1 theo epoch, so sánh tham số, Accuracy/Macro-F1 của checkpoint chọn và F1 từng nhãn. Chỉ bật `EVALUATE_TEST` sau khi chốt cấu hình; biểu đồ tổng kết validation/test không tham gia lựa chọn tham số.

Sau khi chọn checkpoint, mục **Rà lỗi validation** đối chiếu F1, số mẫu train và các cặp nhầm lẫn, đồng thời lưu `validation_review_summary.csv` và `validation_errors.csv` vào thư mục báo cáo của lần chạy. File lỗi giữ nội dung bài, nhãn đã gán, dự đoán, xác suất Softmax chưa hiệu chỉnh và thống kê cắt token để đọc lại; không tự sửa nhãn, đổi split hoặc huấn luyện lại. Dùng phần này để xác định nhóm cần bổ sung mẫu, không tạo quota từ dự đoán và không dùng lỗi test để chọn nhóm.

### Gán loại tin bằng notebook 06

`06_predict_news_type.ipynb` dùng checkpoint đã chọn để dự đoán phần clean chưa có nhãn, không train hoặc tự đánh giá test. Notebook đối chiếu annotation gốc với log gộp trùng/loại bài, giữ nhãn thủ công theo ID chuẩn và không dùng split CSV thay cho toàn bộ hồ sơ annotation. Với dữ liệu hiện tại: 3.356 bài clean gồm 1.058 nhãn thủ công đã chốt, 2.260 bài chưa từng gán nhãn và 38 bài `review_required` từ đợt annotation trước. Hai bài validation STT 16, 18 đã được chốt không thuộc nhóm 38 bài này.

Trước khi nạp model để dự đoán, notebook cần `test_evaluation_config.json` cùng ba báo cáo test thật từ mục 14 của notebook 05. Cell bàn giao mới ở notebook 05 chỉ chạy khi người dùng chủ động bật `EVALUATE_TEST`; việc thêm cell không chạy test. Hồ sơ đối chiếu checksum checkpoint, cấu hình, tokenizer/backbone, dữ liệu, tách từ và báo cáo; nhãn/ID test và các chỉ số phân loại được kiểm tra lại. Không có ngưỡng test để chọn lại model. Thiếu hoặc không khớp thì notebook 06 dừng, không tự chuyển model, train hay tạo kết quả giả.

Đầu ra là `data/processed/news_type_predictions.csv` với đúng bốn cột: `news_id`, `news_type`, `label_source`, `prediction_score`. Mỗi bài clean xuất hiện một lần và giữ nguyên thứ tự. Nhãn thủ công có `label_source=manual`; dự đoán mới có `model`; bài chờ giữ `review_required` và để trống nhãn/điểm, không bị tự gán thay. Điểm model là Softmax chưa hiệu chỉnh; không phải độ tin cậy đã được kiểm chứng. `news_clean.csv` vẫn giữ nguyên tám trường.

File đi kèm `news_type_predictions_metadata.json` ghi nguồn model, hồ sơ test, hash đầu vào và hash CSV để các bước sau kiểm tra. Notebook không ghi đè đầu ra đã có; nếu lưu lỗi giữa chừng, chỉ các file vừa tạo trong lần đó được thu hồi. Khi chạy Colab, hai file nằm trong `MyDrive/stock_news_anomaly_project/data/processed`; khi chạy local nằm trong `news_processing/data/processed`. Chỉ có đầu ra thật sau khi chạy suy luận, không tạo sẵn file nhãn. Các bước sentiment sau đó ghép với clean bằng `news_id` và phân biệt nhãn thủ công/model/bài chờ.

### Chạy bằng GPU Colab trong VS Code

Bản Colab đang dùng: [notebook 05](https://colab.research.google.com/drive/15PU9KCDNkAmzxWN0Yb7AxQLK8TQmtbe7) và [notebook 06](https://colab.research.google.com/drive/1rE1DT_gflQsKkqyJUEWXA_pL2qVqLG2o). Đây là hai bước khác nhau; không cần tạo thêm bản sao notebook để chạy lại.

Hai file nằm chung tại [notebooks/phase_1_news_type trên Drive](https://drive.google.com/drive/folders/1QT5J4RQW9ALAYjERN1W1FdCQ4H__eP4-) trong `MyDrive/stock_news_anomaly_project`, giữ nguyên link mở ở trên. Thư mục tương ứng trên máy và trong ZIP là `news_processing/notebooks/phase_1_news_type/`. Các bản không tự đồng bộ: sửa code trên máy cần cập nhật ZIP và đúng file notebook trên Drive; sau khi chạy Colab, lưu notebook rồi tải bản có output cùng dữ liệu và báo cáo đã xuất về thư mục tương ứng trên máy để nộp khóa luận.

Extension chính thức là `google.colab`. Mở notebook 05, chọn `Select Kernel > Colab > New Colab Server`, đăng nhập Google và chọn T4 nếu Colab có cấp. Đăng nhập/cấp quyền phải do chủ tài khoản thực hiện. Không chọn máy tính phí khi chưa chủ động đồng ý chi phí.

Notebook ưu tiên `MyDrive/stock_news_anomaly_project/news_processing_colab.zip` sau khi kết nối Drive. Gói hiện gồm notebook 05/06, `src/*.py`, clean, CSV nhãn gốc và audit, log loại/gộp bài, ba tập train/validation/test, mapping và hồ sơ cập nhật validation. Các đầu vào này giúp notebook 06 giữ đúng nhãn thủ công và trạng thái chờ; không chứa raw, Java Windows, checkpoint hoặc kết quả dự đoán giả. Nếu Drive chưa có ZIP, dùng Explorer `Upload to Colab` hoặc bảng Files trong trình duyệt để đưa gói vào `/content/news_processing_colab.zip`. Chỉ giữ một bản ZIP đúng tên trong thư mục Drive dự án và cập nhật bản đó khi code thay đổi.

Chạy từ đầu notebook: phần Colab kiểm tra CUDA, cài các thư viện cần thiết nhưng không thay PyTorch/CUDA, kết nối Drive và giải nén gói dữ liệu. Nếu chưa có GPU, notebook dừng với thông báo chọn GPU, không âm thầm huấn luyện CPU. PyTorch cần từ 2.6 để nạp checkpoint `.bin` của PhoBERT qua Transformers đang dùng.

Để nhận bản sửa, mở lại notebook 05 đã cập nhật. Nếu đã import code cũ, khởi động lại phiên Python trước khi chạy từ đầu. Cell giải nén ưu tiên ZIP trên Drive, rồi mới dùng ZIP trong `/content`, và cập nhật cả thư mục dự án đã tồn tại; không cần xóa thư mục. Chỉ dùng `Run all` trên bản không có cell tạm nạp checkpoint cũ. Việc tải/giải nén ZIP không tự thay nội dung notebook đang mở trong trình duyệt. Giữ `RUN_MODE = 'review'` để tiếp tục từ model mới đã lưu mà không train lại; sau khi mở phiên mới vẫn cần chạy phần thiết lập, đọc dữ liệu và mã hóa validation.

Checkpoint và báo cáo huấn luyện của Colab lưu trong `MyDrive/stock_news_anomaly_project/models/news_type_phobert_cnn/runs/<RUN_NAME>` và `MyDrive/stock_news_anomaly_project/reports/news_type/runs/<RUN_NAME>`. Đánh giá validation sửa nhãn lưu riêng trong `reports/news_type/evaluations/<revision_id>`. Cache và dữ liệu đọc giữ tại `/content/news_processing`. Không tải Java portable Windows lên Colab. Checkpoint tốt nhất hiện không bao gồm optimizer/scheduler, nên không coi là cơ chế tiếp tục huấn luyện đầy đủ khi phiên Colab kết thúc.

Kết nối thành công phải được xác nhận bằng output CUDA và tên GPU thực tế. Các output cũ từ máy cục bộ không chứng minh Colab đã kết nối. Tham khảo [hướng dẫn extension](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) và [giới hạn tài nguyên Colab](https://research.google.com/colaboratory/faq.html#resource-limits).

Phase 2 và 3 hiện chỉ có thư mục; chỉ tạo notebook và đầu ra khi thực hiện. Không dùng notebook review riêng. Các nhóm ứng viên trong báo cáo chọn mẫu không phải nhãn xác nhận.

