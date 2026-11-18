# BÁO CÁO CẤU TRÚC PROJECT VÀ QUY TRÌNH CRAWL DỮ LIỆU TIN TỨC

## 1. Tổng quan

Thư mục `news_crawler/` phục vụ đề tài:

> Xây dựng mô hình phát hiện bất thường trong giao dịch cổ phiếu và phân tích mối liên hệ với các sự kiện tin tức doanh nghiệp.

Hệ thống hiện thực hiện được các công việc chính sau:

1. Thu thập URL bài báo từ sitemap, RSS, `robots.txt` và trang chuyên mục.
2. Tải nội dung bài báo bằng HTTP scraping.
3. Xác minh ngày đăng và mức độ liên quan đến 14 mã cổ phiếu.
4. Chống crawl lại và loại bài trùng URL hoặc trùng nội dung.
5. Lưu quan hệ giữa bài báo và mã cổ phiếu.
6. Kiểm tra độ phủ dữ liệu theo nguồn và ticker.
7. Gộp dữ liệu crawler chính với phần dữ liệu hợp lệ do thành viên khác cung cấp.

Project sử dụng Python và không phụ thuộc Selenium hoặc trình duyệt tự động. Firecrawl từng được dùng ở giai đoạn tìm kiếm ban đầu, nhưng pipeline crawl chính hiện tại có thể chạy bằng Python thuần.

## 2. Phạm vi dữ liệu

### 2.1. Danh sách mã cổ phiếu

Project theo dõi 14 mã:

`ANV, DBC, DGW, DHG, FRT, IMP, MSN, MWG, PNJ, QNS, SAB, SBT, VHC, VNM`.

Các tên doanh nghiệp và thương hiệu tương ứng được cấu hình trong `config/ticker_aliases.json`. Ví dụ: MWG còn được nhận diện qua Thế Giới Di Động, Điện Máy Xanh, Bách Hóa Xanh và An Khang; FRT được nhận diện qua FPT Retail, FPT Shop và Long Châu.

### 2.2. Khoảng thời gian

- Tin tức: từ `2020-01-01` đến `2026-09-25`.

### 2.3. Quy mô dữ liệu hiện tại

| File | Số dòng | Vai trò |
|---|---:|---|
| `data/raw/news_raw.csv` | 3.633 | Bài báo do crawler chính tải và xác minh |
| `data/final/news_ticker.csv` | 3.710 | Quan hệ bài báo–ticker của crawler chính |
| `data/final/News_raw.csv` | 4.491 | Dữ liệu news cuối đã gộp với dữ liệu hợp lệ của thành viên khác |

`News_raw.csv` có 4.491 URL, `news_id` và `content_hash` duy nhất; toàn bộ bản ghi có `crawl_status = success`.

## 3. Cấu trúc thư mục

| Thư mục/file | Chức năng |
|---|---|
| `config/` | Cấu hình ngày, ticker, nguồn và alias doanh nghiệp |
| `data/raw/` | Dữ liệu crawl gốc; không nên chỉnh sửa trực tiếp |
| `data/staging/` | URL ứng viên, checkpoint SQLite và dữ liệu trung gian |
| `data/final/` | Dataset news cuối (đã gộp) và quan hệ bài báo–ticker |
| `src/crawler/` | Mã nguồn tìm URL, tải, xác minh bài báo và gắn ticker |
| `src/cleaning/` | Gộp dữ liệu crawler với dữ liệu của Thảo (`merge_thao_news.py`) |
| `src/quality/` | Báo cáo coverage theo nguồn × ticker |
| `outputs/coverage/` | Báo cáo coverage, lỗi crawl và kết quả kiểm định |
| `outputs/logs/` | Log và trạng thái pipeline |
| `.firecrawl/` | Kết quả tìm kiếm/scrape Firecrawl đã lưu từ giai đoạn trước |
| `tests/` | Kiểm thử cấu hình và pipeline |

## 4. Công cụ và thư viện

### 4.1. Công cụ crawl

- Python 3.
- `urllib.request`: gửi HTTP request trực tiếp.
- `lxml`: phân tích HTML, XML, RSS và sitemap.
- `python-dateutil`: chuẩn hóa ngày đăng.
- `ThreadPoolExecutor`: tải nhiều URL song song.
- SQLite: lưu checkpoint discovery và tránh xử lý lại tài nguyên đã duyệt.
- SHA-256: tạo `news_id` và `content_hash` để chống trùng.
- Firecrawl CLI: công cụ hỗ trợ tìm kiếm/scrape ở giai đoạn trước; không bắt buộc đối với pipeline Python native.

Pipeline không sử dụng Selenium, Playwright hay browser automation. Vì vậy, website yêu cầu JavaScript, CAPTCHA hoặc đăng nhập có thể không lấy được nội dung.

### 4.2. Công cụ kiểm thử

- `pytest`: kiểm thử.

Danh sách dependency nằm trong `requirements.txt`.

## 5. Input của hệ thống

### 5.1. Input cấu hình

1. `config/project_config.json`
   - Khoảng ngày crawl.
   - 14 mã cổ phiếu.
   - Đường dẫn input/output.
   - Retry và khoảng nghỉ request.

2. `config/sources.csv`
   - Danh sách 51 nguồn khởi tạo.
   - Domain, seed URL, loại nguồn và trạng thái bật/tắt.

3. `config/ticker_aliases.json`
   - Mã cổ phiếu.
   - Tên doanh nghiệp và thương hiệu dùng để xác minh bài liên quan.

### 5.2. Input dữ liệu

- `data/raw/news_raw.csv`: vừa là output của crawler chính, vừa là checkpoint URL đã có cho lần chạy incremental.
- `DATA_THAO - NEWS_RAW.csv`: nguồn dữ liệu bổ sung ngoài project, chỉ được đọc bởi `src/cleaning/merge_thao_news.py`; file raw gốc không bị chỉnh sửa.

## 6. Quy trình crawl dữ liệu news

### Bước 1 — Discovery URL

File chính: `src/crawler/discover_native.py`.

Crawler đọc:

- `robots.txt`;
- sitemap và sitemap index;
- RSS/feed;
- liên kết trên trang chuyên mục/seed URL.

URL được kiểm tra theo domain và metadata. Những URL đã có trong `data/raw/news_raw.csv` được bỏ qua. Trạng thái discovery được lưu trong:

- `data/staging/native_discovery.sqlite`;
- `data/staging/news_candidates_native.csv`.

Discovery chạy song song bằng `ThreadPoolExecutor` và không đặt giới hạn tổng số bài trong pipeline native.

### Bước 2 — Tải và bóc tách bài báo

Các file chính:

- `src/crawler/run_backfill.py`;
- `src/crawler/scrape_native_candidates.py`.

Hệ thống gửi HTTP request đến URL gốc, sau đó trích xuất:

- tiêu đề;
- tóm tắt;
- nội dung;
- ngày đăng;
- tác giả;
- chuyên mục;
- canonical URL.

### Bước 3 — Xác minh bài hợp lệ

Bài chỉ được chấp nhận khi:

1. URL tải thành công.
2. Tiêu đề có độ dài tối thiểu.
3. Nội dung có ít nhất 200 ký tự.
4. Có ngày đăng hợp lệ trong giai đoạn dự án.
5. Có bằng chứng liên quan tới một trong 14 ticker qua mã, tên công ty hoặc thương hiệu.

URL lỗi, nội dung thiếu hoặc không đủ bằng chứng bị ghi vào báo cáo; hệ thống không tự tạo nội dung thay thế.

### Bước 4 — Chống trùng

Hệ thống chống trùng theo hai cấp:

- Canonical URL.
- SHA-256 của nội dung đã chuẩn hóa khoảng trắng.

Nếu một bài liên quan nhiều mã, nội dung chỉ lưu một lần trong `news_raw.csv`; các quan hệ ticker được lưu thành nhiều dòng trong `news_ticker.csv`.

### Bước 5 — Ghi output và kiểm tra coverage

Pipeline ghi:

- bài báo vào `data/raw/news_raw.csv`;
- quan hệ bài–mã vào `data/final/news_ticker.csv`;
- báo cáo vào `outputs/coverage/`;
- log/trạng thái vào `outputs/logs/`.

Entry point chính là `run_native_news_pipeline.py`, chạy theo thứ tự:

`discover → scrape → build coverage → tests`.

## 7. Quy trình gộp dữ liệu news

File `src/cleaning/merge_thao_news.py` gộp dữ liệu crawler chính với phần dữ liệu hợp lệ của thành viên khác.

Quy tắc:

- Chỉ nhận dòng có trạng thái thành công, có nội dung và đúng khoảng ngày.
- Trùng URL với crawler chính: giữ bài crawler chính, chỉ bổ sung metadata và ticker có thật.
- Trùng nội bộ file bổ sung: giữ một URL và hợp nhất các ticker/metadata thực tế.
- Chuẩn hóa `news_id` và `content_hash` về SHA-256.
- Không điền giả các trường thiếu.

Output cuối là `data/final/News_raw.csv`, gồm 4.491 bài:

- 3.273 bài chỉ từ crawler chính;
- 858 bài chỉ từ crawler của thành viên khác;
- 360 bài có mặt ở cả hai nguồn và đã được hợp nhất.

## 8. Output của project

### 8.1. Output crawl chính

`data/raw/news_raw.csv` có các trường chính:

`news_id, source, url, title, summary, content, published_date, published_at_raw, author, category, crawl_time, crawl_status, error_message, content_hash`.

### 8.2. Output liên kết ticker

`data/final/news_ticker.csv` có:

`news_id, ticker, match_method, match_confidence, matched_text, published_date, source, url`.

### 8.3. Output news dùng cho project

`data/final/News_raw.csv` là file news đầu vào được khuyến nghị cho project. File đã có cột `tickers`, do đó có thể dùng mà không bắt buộc join lại `news_ticker.csv`.

## 9. Danh sách 51 nguồn cấu hình

### 9.1. Báo chí, tài chính và trang dữ liệu — 37 nguồn

| STT | Nguồn | Domain | Loại |
|---:|---|---|---|
| 1 | CafeF | cafef.vn | stock_tag |
| 2 | CafeBiz | cafebiz.vn | category |
| 3 | VnExpress | vnexpress.net | category |
| 4 | VietnamNet | vietnamnet.vn | category |
| 5 | VnEconomy | vneconomy.vn | stock_tag |
| 6 | VnBusiness | vnbusiness.vn | category |
| 7 | Dân Trí | dantri.com.vn | category |
| 8 | Doanh Nghiệp Việt Nam | doanhnghiepvn.vn | category |
| 9 | Sức khỏe & Đời sống | suckhoedoisong.vn | category |
| 10 | ZNEWS | znews.vn | category |
| 11 | Soha | soha.vn | category |
| 12 | Đầu Tư Việt Nam | dautuvietnam.com.vn | category |
| 13 | Đời Sống Việt Nam | doisongvietnam.vn | category |
| 14 | Tuổi Trẻ | tuoitre.vn | category |
| 15 | Thanh Niên | thanhnien.vn | category |
| 16 | Người Quan Sát | nguoiquansat.vn | category |
| 17 | StockBiz | stockbiz.vn | category |
| 18 | VietnamFinance | vietnamfinance.vn | category |
| 19 | MarketTimes | markettimes.vn | category |
| 20 | Nhà Đầu Tư | nhadautu.vn | category |
| 21 | Kinh tế Chứng khoán | kinhtechungkhoan.vn | category |
| 22 | VOV | vov.vn | category |
| 23 | Doanh Nhân Việt Nam | doanhnhanvn.vn | category |
| 24 | 24HMoney | 24hmoney.vn | category |
| 25 | Tạp chí Tài chính | tapchitaichinh.vn | category |
| 26 | Vietstock | vietstock.vn | category |
| 27 | Tin nhanh Chứng khoán | tinnhanhchungkhoan.vn | category |
| 28 | Báo Đầu tư | baodautu.vn | category |
| 29 | Mekong ASEAN | mekongasean.vn | category |
| 30 | TheLEADER | theleader.vn | category |
| 31 | BNews | bnews.vn | category |
| 32 | Nhân Dân | nhandan.vn | category |
| 33 | Báo Chính phủ | baochinhphu.vn | category |
| 34 | Lao Động | laodong.vn | category |
| 35 | Agro.gov.vn | agro.gov.vn | category |
| 36 | Vietcap | vietcap.com.vn | research |
| 37 | NMQuan | nmquan.vn | stock_profile |

### 9.2. Website chính thức doanh nghiệp — 14 nguồn

| STT | Nguồn | Domain | Ticker liên quan |
|---:|---|---|---|
| 38 | Navico | navicorp.com.vn | ANV |
| 39 | Dabaco | dabaco.com.vn | DBC |
| 40 | Digiworld | digiworld.com.vn | DGW |
| 41 | DHG Pharma | dhgpharma.com.vn | DHG |
| 42 | FPT Retail | frt.vn | FRT |
| 43 | Imexpharm | imexpharm.com | IMP |
| 44 | Masan Group | masangroup.com | MSN |
| 45 | Thế Giới Di Động | mwg.vn | MWG |
| 46 | PNJ | pnj.com.vn | PNJ |
| 47 | Đường Quảng Ngãi | qns.com.vn | QNS |
| 48 | Sabeco | sabeco.com.vn | SAB |
| 49 | TTC AgriS | ttcagris.com | SBT |
| 50 | Vĩnh Hoàn | vinhhoan.com | VHC |
| 51 | Vinamilk | vinamilk.com.vn | VNM |

51 là số nguồn được cấu hình để crawler kiểm tra, không phải số nguồn chắc chắn đóng góp bài. `News_raw.csv` hiện có bài từ 37 giá trị `source`; crawler chính có bài từ 30 nguồn.

## 10. Cách chạy

Tại thư mục project:

```powershell
cd news_crawler
```

Cài dependency:

```powershell
python -m pip install -r requirements.txt
```

Chạy pipeline crawl native:

```powershell
python run_native_news_pipeline.py
```

Tạo lại file news đã gộp:

```powershell
python -m src.cleaning.merge_thao_news
```

Không cần chạy lại crawler chỉ để sử dụng dataset hiện tại.

## 11. Lưu ý và hạn chế

1. `News_raw.csv` mang tên “raw” nhưng thực tế là dataset đã gộp, lọc và khử trùng.
2. `keywords`, `event_type`, `industry_group` và `note` chỉ có ở một phần dữ liệu (phần đến từ dữ liệu của Thảo).
3. HTTP crawler không lấy được mọi trang bị chặn bot, yêu cầu JavaScript, đăng nhập, CAPTCHA hoặc đã bị xóa.
4. Không được hiểu “51 nguồn cấu hình” là bảo đảm đã lấy toàn bộ mọi bài từng xuất hiện trên Internet.
5. Project này chỉ phụ trách crawl và gộp dữ liệu news; tiền xử lý, ghép với giá cổ phiếu và mô hình phát hiện bất thường nằm ngoài phạm vi.

## 12. Kết luận

Project đã có đầy đủ source code crawler, cấu hình 51 nguồn, checkpoint chống crawl lại và dataset news cuối 4.491 bài cho 14 mã cổ phiếu tại `data/final/News_raw.csv`.
