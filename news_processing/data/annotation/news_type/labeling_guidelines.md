# Hướng dẫn gán nhãn loại tin

## 1. Mục tiêu

Mỗi bài viết được gán đúng một `news_type` theo chủ đề chính. Nhãn được xác định từ tiêu đề, phần mở đầu và nội dung bài; không dựa vào biến động giá xảy ra sau khi bài được đăng.

Hai thuộc tính `impact_level` và `impact_direction` chưa được gán ở giai đoạn này. Tác động thực tế sẽ được đánh giá riêng khi liên kết tin với kết quả TN1.

## 2. Bộ nhãn

### 2.1. `governance_capital` — Quản trị, sở hữu và hành động vốn

Chọn khi trọng tâm là:

- thay đổi nhân sự cấp cao, HĐQT, ban kiểm soát hoặc cơ cấu quản trị;
- cổ đông lớn, sở hữu và giao dịch của người nội bộ;
- cổ tức, ESOP, phát hành hoặc mua lại cổ phiếu;
- đại hội cổ đông và nghị quyết quản trị.

Quy tắc `company_awards_listing_v2` (25/10/2026): IPO, niêm yết lần đầu và chuyển sàn thuộc `governance_capital` khi chính hành động huy động vốn/niêm yết là sự kiện trung tâm. Một bài phân tích giá, định giá hoặc cơ hội đầu tư vẫn thuộc `stock_market`, dù có nhắc kế hoạch niêm yết. Không gán nhãn chỉ theo từ khóa IPO, HoSE hoặc chuyển sàn.

Không chọn cho M&A, dự án đầu tư hoặc mở rộng kinh doanh; các trường hợp này thuộc `business_operations`.

### 2.2. `financial_results` — Kết quả kinh doanh và tài chính

Chọn khi trọng tâm là:

- doanh thu, lợi nhuận, biên lợi nhuận hoặc kết quả theo quý/năm;
- báo cáo tài chính, dòng tiền, tài sản và nợ;
- kế hoạch kinh doanh và mức độ hoàn thành kế hoạch do doanh nghiệp công bố.

Nếu bài chủ yếu định giá hoặc khuyến nghị cổ phiếu dựa trên kết quả kinh doanh, chọn `stock_market`.

### 2.3. `business_operations` — Hoạt động kinh doanh và đầu tư

Chọn khi trọng tâm là:

- M&A, đầu tư, thoái vốn hoặc liên doanh;
- dự án, nhà máy, công suất và mở rộng thị trường;
- sản phẩm, dịch vụ, hợp đồng hoặc đối tác mới;
- hoạt động sản xuất, bán hàng, tái cấu trúc vận hành, ESG hoặc trách nhiệm xã hội.

Theo `company_awards_listing_v2`, bài vinh danh/xếp hạng một doanh nghiệp thuộc `business_operations` khi trọng tâm là thành tựu hoạt động, phát triển tổng thể hoặc ESG; các con số tài chính chỉ làm dẫn chứng. Nếu trọng tâm thực sự là công bố kết quả tài chính, vẫn chọn `financial_results`. Bài xếp hạng nhiều doanh nghiệp để phân tích xu hướng toàn ngành có thể thuộc `industry`. Việc có giải thưởng hoặc chữ “Top” không tự quyết định nhãn.

### 2.4. `legal_compliance` — Pháp lý và tuân thủ doanh nghiệp

Chọn khi một doanh nghiệp cụ thể là đối tượng của:

- xử phạt, thanh tra, điều tra hoặc truy thu;
- kiện tụng, tranh chấp hoặc vi phạm;
- quyết định pháp lý ảnh hưởng trực tiếp đến doanh nghiệp.

Quy định áp dụng chung cho một ngành hoặc chính sách vĩ mô toàn nền kinh tế thuộc `industry`.

### 2.5. `industry` — Ngành, chuỗi giá trị và kinh tế vĩ mô

Chọn khi trọng tâm là bối cảnh ngành hoặc vĩ mô bên ngoài doanh nghiệp:

- cung cầu, cạnh tranh hoặc xu hướng chung của một ngành;
- giá nguyên liệu, sản phẩm hoặc chuỗi cung ứng của ngành;
- chính sách và quy định áp dụng cho một ngành hoặc toàn nền kinh tế;
- lãi suất, tỷ giá, lạm phát, GDP, thuế quan hoặc chính sách tiền tệ/tài khóa có tác động truyền dẫn tới ngành và doanh nghiệp;
- nhiều doanh nghiệp được phân tích để phản ánh xu hướng ngành.

Quy tắc `gold_price_industry_v1` (24/10/2026): bài có nội dung chính là cập nhật giá vàng trong nước/thế giới, giá mua–bán, cung cầu hoặc diễn biến thị trường vàng thuộc `industry`. Áp dụng cả bản tin ngắn, bảng giá và bài có phân tích; không yêu cầu doanh nghiệp cụ thể phải là nhân vật chính. Việc PNJ chỉ xuất hiện như một đơn vị niêm yết giá không làm bài chuyển thành `mention_only`.

Không đổi nhãn chỉ vì tiêu đề chứa “giá vàng”. Nếu trọng tâm là doanh thu/lợi nhuận PNJ, hoạt động kinh doanh, xử lý pháp lý hoặc giá cổ phiếu, chọn nhãn sự kiện tương ứng. Bài thiếu nội dung hoặc có hai chủ đề ngang nhau vẫn phải xem lại, không tự gán theo từ khóa.

*(Lưu ý: Các bản ghi từng gán nhãn lịch sử `macro` được hợp nhất vào `industry` khi chuẩn bị dữ liệu huấn luyện để chốt 7 loại `news_type` chính).*

### 2.6. `stock_market` — Thị trường chứng khoán và định giá

Chọn khi trọng tâm là:

- diễn biến giá cổ phiếu, thanh khoản hoặc chỉ số thị trường;
- định giá, khuyến nghị mua/bán hoặc giá mục tiêu;
- giao dịch khối ngoại, phân tích kỹ thuật hoặc xu hướng thị trường;
- so sánh cơ hội đầu tư giữa các cổ phiếu.

Không chọn chỉ vì bài có nhắc đến giá cổ phiếu. Nếu nội dung chính là kết quả kinh doanh hoặc sự kiện doanh nghiệp, chọn nhãn tương ứng.

### 2.7. `mention_only` — Tổng hợp hoặc ít liên quan

Chọn khi:

- doanh nghiệp hoặc ticker chỉ được nhắc tên trong danh sách hay ví dụ;
- bài không cung cấp sự kiện hoặc thông tin đáng kể về bất kỳ doanh nghiệp được theo dõi nào;
- nội dung chính không đủ cơ sở để xếp vào sáu nhóm còn lại.

Chỉ chọn `mention_only` sau khi xác định nội dung chính không thuộc sáu nhóm còn lại. Tin thị trường vàng có thông tin giá vẫn thuộc `industry`, dù doanh nghiệp chỉ được nhắc tên. Không dùng nhãn này thay cho trường hợp khó quyết định. Trường hợp còn mơ hồ phải đặt `annotation_status = review_required`.

## 3. Quy tắc chọn nhãn chính

Ưu tiên xác định chủ đề theo thứ tự bằng chứng:

1. Tiêu đề bài viết.
2. Nội dung đoạn mở đầu hoặc `summary`.
3. Nội dung chiếm phần lớn bài viết.

Luồng quyết định:

```text
Trọng tâm là giá cổ phiếu, thị trường, định giá       → stock_market
Trọng tâm là ngành, giá hàng hóa hoặc kinh tế vĩ mô  → industry
Doanh nghiệp bị xử phạt, kiện tụng, thanh tra         → legal_compliance
Trọng tâm là doanh thu, lợi nhuận, báo cáo             → financial_results
Trọng tâm là quản trị, sở hữu, hành động vốn           → governance_capital
Trọng tâm là hoạt động kinh doanh hoặc đầu tư          → business_operations
Không thuộc các nhóm trên, chỉ nhắc tên doanh nghiệp  → mention_only
```

Nếu bài có nhiều chủ đề, chọn chủ đề được nhấn mạnh trong tiêu đề và phần mở đầu. Nếu hai chủ đề có vai trò tương đương, chưa gán `news_type` và đặt `annotation_status = review_required`.

## 4. Trạng thái gán nhãn

- `labeled`: đã xác định được một nhãn theo hướng dẫn.
- `review_required`: nội dung mơ hồ, chồng lấn hoặc cần người thứ hai kiểm tra.
- `excluded`: bài không đọc được hoặc dữ liệu nội dung không đủ để gán nhãn.

`mention_only` là một nhãn hợp lệ, không đồng nghĩa với `excluded`.

Với bài đã thuộc một tập đánh giá cố định, quyết định chưa chốt được ghi vào `pending_reviews` trong hồ sơ phiên bản dữ liệu trước. Không đổi ngay `annotation_status` của bản benchmark đang dùng, không tự loại bài hoặc chia lại tập. Ngày 25/10/2026, người gán xác nhận STT 16 và 18 trong báo cáo lỗi của lần chạy `20261024_153932_380173` cùng giữ `financial_results`; lý do được lưu ở `resolved_reviews`, danh sách chờ hiện rỗng. Nhãn, trạng thái và membership của hai bài không đổi. `review_required` là trạng thái gán nhãn, không phải lớp `news_type` thứ tám.

## 5. Kiểm soát chất lượng

- Sử dụng 7 mã nhãn chính trong `label_mapping.json` (các bản ghi lịch sử gán `macro` được chuẩn hóa tự động về `industry`).
- Không suy luận nhãn từ ticker hoặc nguồn đăng bài.
- Không sử dụng biến động giá tương lai để chọn `news_type`.
- Ghi lý do ngắn trong `annotator_note` đối với mọi dòng `review_required` hoặc `excluded`.
- Kiểm tra lại các bài có cùng tiêu đề hoặc cùng sự kiện để bảo đảm cách gán nhất quán.
- Cập nhật theo quy tắc mới phải có danh sách ID đã đọc xác nhận trong `label_corrections.csv`, lưu nhãn cũ, nhãn mới và dấu kiểm nội dung. Không dùng dự đoán model làm nhãn đúng; không thay đổi membership của train/validation/test.
- Phiên bản nhãn nằm trong `label_mapping.json`. Khi nhãn validation/test thay đổi, kết quả lịch sử vẫn thuộc phiên bản cũ; không so trực tiếp điểm trước/sau như một phép cải thiện mô hình trên cùng benchmark.
- Đợt `company_awards_listing_v2` chỉ áp dụng sáu ID đã được người dùng đọc và chốt; không tự sửa thêm bài vì model dự đoán khác nhãn. Các bài tương tự cần được đối chiếu theo cùng quy tắc trước khi có quyết định mới.
- `content_sha256` giữ dấu kiểm nội dung tại thời điểm duyệt nhãn. Cột tùy chọn `reviewed_content_sha256` chỉ ghi nhận bản làm sạch nội dung đã duyệt của đúng bài chuyển sàn MCH trong đợt này; không thay thế dấu kiểm gốc hoặc cho phép bỏ kiểm tra nội dung tùy ý.
