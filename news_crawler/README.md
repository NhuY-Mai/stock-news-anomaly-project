# Stock News Crawler

Project Python dùng để **crawl tin tức doanh nghiệp** cho 14 mã cổ phiếu và **gộp với dữ liệu news của Thảo** thành một dataset cuối.

- **Mã cổ phiếu (14):** ANV, DBC, DGW, DHG, FRT, IMP, MSN, MWG, PNJ, QNS, SAB, SBT, VHC, VNM
- **Khoảng thời gian:** 2020-01-01 → 2026-09-25
- **Nguồn:** 51 nguồn cấu hình (33 báo/chuyên mục, 14 website chính thức doanh nghiệp, 4 trang tag/dữ liệu chứng khoán), ngoài ra có discovery web mở
- **Kết quả cuối:** `data/final/News_raw.csv` — **4.491 bài** (3.633 bài crawl + 858 bài thêm từ dữ liệu của Thảo)

Không dùng Selenium/trình duyệt tự động: toàn bộ là HTTP request + phân tích HTML/XML bằng `lxml`.

---

## 1. Cấu trúc thư mục

```
news_crawler/
├── README.md
├── requirements.txt
├── .gitignore
├── run_native_news_pipeline.py          # Pipeline crawl chính (end-to-end)
├── run_open_web_pipeline.py             # Pipeline crawl mở rộng qua tìm kiếm web
│
├── config/
│   ├── project_config.json
│   ├── sources.csv
│   └── ticker_aliases.json
│
├── src/
│   ├── crawler/
│   │   ├── run_backfill.py
│   │   ├── discover_native.py
│   │   ├── discover_open_web.py
│   │   ├── discover_firecrawl.py
│   │   ├── discover_by_ticker.py
│   │   ├── scrape_native_candidates.py
│   │   ├── scrape_firecrawl.py
│   │   └── store.py
│   ├── cleaning/
│   │   └── merge_thao_news.py
│   └── quality/
│       └── build_coverage.py
│
├── data/
│   ├── raw/
│   │   ├── news_raw.csv
│   │   └── news_raw_before_hash_dedupe.csv
│   ├── staging/
│   │   ├── news_candidates_*.csv
│   │   ├── native_discovery.sqlite
│   │   ├── native_discovery_pre_checkpoint.sqlite
│   │   ├── open_web_discovery.sqlite
│   │   └── google_sample.html
│   └── final/
│       ├── News_raw.csv                 # ★ DATASET CUỐI
│       ├── news_ticker.csv
│       └── news_ticker_before_hash_dedupe.csv
│
├── outputs/
│   ├── coverage/
│   └── logs/
├── logs/crawl.log
├── .firecrawl/
├── docs/
│   ├── BAO_CAO_CAU_TRUC_VA_CRAWL_NEWS.md
│   └── data_dictionary.md
└── tests/test_config.py
```

### 1.1. File chạy ở thư mục gốc

| File | Chức năng |
|---|---|
| `run_native_news_pipeline.py` | Pipeline crawl chính, chạy tuần tự: `discover_native` → `scrape_native_candidates` → `build_coverage` → `pytest`. Ghi trạng thái từng bước vào `outputs/logs/native_pipeline_status.json`, dừng ngay nếu một bước lỗi. |
| `run_open_web_pipeline.py` | Pipeline mở rộng: `discover_open_web` → `scrape_native_candidates` → `build_coverage`. Tìm bài trên mọi domain (không giới hạn trong 51 nguồn). |
| `requirements.txt` | `lxml`, `python-dateutil`, `certifi`, `pytest`. |

### 1.2. `config/` — cấu hình

| File | Nội dung |
|---|---|
| `project_config.json` | Ngày bắt đầu/kết thúc, danh sách 14 ticker, đường dẫn file output, `request_delay_seconds`, `max_retries`, `max_articles_per_source`. |
| `sources.csv` | 51 nguồn, mỗi dòng gồm: `source` (tên), `domain`, `seed_url` (trang bắt đầu crawl), `enabled` (`true`/`false`), `source_type` (`category`, `official`, `stock_tag`, `stock_profile`, `research`). Muốn tắt một nguồn thì đổi `enabled` thành `false`. |
| `ticker_aliases.json` | Với mỗi ticker: danh sách tên công ty / thương hiệu dùng để nhận diện bài liên quan. Ví dụ MWG → Thế Giới Di Động, Điện Máy Xanh, Bách Hóa Xanh, An Khang; FRT → FPT Retail, FPT Shop, Long Châu. |

### 1.3. `src/crawler/` — code crawl

| File | Vai trò | Input → Output |
|---|---|---|
| `run_backfill.py` | **Module lõi.** Chứa các hàm dùng chung: tải trang (`_safe_fetch`, có retry + SSL `certifi`), chuẩn hóa URL (`_normalize_url`), bóc tách tiêu đề / tóm tắt / nội dung / ngày đăng / tác giả / chuyên mục / canonical URL. Chạy trực tiếp thì crawl kiểu cũ: vào `seed_url` của từng nguồn, lấy tối đa N bài/nguồn, **không kiểm tra bài có liên quan tới 14 mã** nên chỉ ghi ra staging và từ chối ghi vào `news_raw.csv`. | `sources.csv` → `data/staging/news_backfill.csv`, `outputs/coverage/crawl_report.csv`, `logs/crawl.log` |
| `discover_native.py` | **Tìm URL ứng viên** từ `robots.txt`, sitemap / sitemap index, RSS/feed và trang `seed_url` của từng nguồn. Chạy song song 32 luồng, lưu checkpoint SQLite để chạy lại không phải duyệt lại. | `sources.csv`, `ticker_aliases.json` → `data/staging/news_candidates_native.csv`, `native_discovery.sqlite` |
| `discover_open_web.py` | Tìm URL qua DuckDuckGo với truy vấn dạng `"<tên công ty>" tin tức <năm>` cho từng ticker × từng năm 2020–2026. Không giới hạn domain. | `ticker_aliases.json` → `data/staging/news_candidates_open_web.csv`, `open_web_discovery.sqlite` |
| `discover_firecrawl.py` | Tìm URL bằng Firecrawl CLI với truy vấn `site:<domain> (ANV OR DBC OR …)` cho từng nguồn. Cũng chứa 2 hàm dùng chung: `match_tickers` (gắn ticker) và `is_article_candidate` (lọc URL không phải bài báo). | → `data/staging/news_candidates.csv`, `.firecrawl/search-*.json` |
| `discover_by_ticker.py` | Tìm URL bằng Firecrawl, mỗi ticker một truy vấn, chỉ giữ kết quả thuộc 51 nguồn. | → `data/staging/news_candidates_by_ticker.csv`, `.firecrawl/ticker-search/` |
| `scrape_native_candidates.py` | **Tải và xác minh bài.** Đọc file ứng viên, tải từng URL (song song), bóc tách nội dung, kiểm tra hợp lệ, gắn ticker, loại trùng rồi **gộp thêm** vào dữ liệu đã có. | `news_candidates_*.csv` → `data/raw/news_raw.csv`, `data/final/news_ticker.csv` (gộp thêm), `outputs/coverage/native_more_report.csv` |
| `scrape_firecrawl.py` | Tải bài bằng Firecrawl (giới hạn 10 request/phút), cache vào `.firecrawl/articles/`. | `news_candidates.csv` → `news_raw.csv`, `news_ticker.csv` (gộp thêm), `outputs/coverage/firecrawl_scrape_report.csv` |
| `store.py` | Hàm `merge_into_dataset` dùng chung cho 2 script scrape: gộp bài mới vào `news_raw.csv` + `news_ticker.csv`, **bài đã có luôn được giữ nguyên**, loại trùng theo URL rồi theo `content_hash`. | bài mới + nhãn mới → `data/raw/news_raw.csv`, `data/final/news_ticker.csv` |

### 1.4. `src/cleaning/` và `src/quality/`

| File | Vai trò |
|---|---|
| `cleaning/merge_thao_news.py` | Gộp `data/raw/news_raw.csv` (lấy nhãn mã từ `data/final/news_ticker.csv` để điền cột `tickers`) với file của Thảo → `data/final/News_raw.csv`, báo cáo ở `outputs/coverage/merge_thao_report.json`. Không sửa file gốc nào. |
| `quality/build_coverage.py` | Đếm số bài theo từng cặp nguồn × ticker (51 × 14 = 714 dòng), đánh dấu `covered` / `missing` → `outputs/coverage/coverage.csv`. |

### 1.5. `data/` — dữ liệu

| Thư mục / file | Nội dung |
|---|---|
| `raw/news_raw.csv` | Bài báo crawl đã xác minh, **3.633 bài**, mỗi URL một dòng. Đồng thời là checkpoint: URL đã có ở đây sẽ không bị tải lại. |
| `raw/news_raw_before_hash_dedupe.csv` | Bản sao lưu trước khi loại trùng theo nội dung. |
| `staging/news_candidates_*.csv` | URL ứng viên (chưa xác minh) từ các bước discovery. Các file `_all`, `_extra`, `_more`, `_fast_verify`, `_retry_errors`, `_verify_all`, `_verify_complete_snapshot` là kết quả các lần chạy trước. |
| `staging/*.sqlite` | Checkpoint discovery: URL đã duyệt (`visited`), truy vấn đã chạy (`queries`), ứng viên (`candidates`). Xóa file này thì lần sau sẽ duyệt lại từ đầu. |
| `staging/google_sample.html` | Trang mẫu lưu lại khi thử nghiệm, không được code sử dụng. |
| `final/news_ticker.csv` | Nhãn "bài báo nói về mã nào", **3.710 dòng**: mỗi dòng gồm `news_id`, mã cổ phiếu và bằng chứng nhận diện (`matched_text`, `match_method`). Một bài nhắc nhiều công ty thì có nhiều dòng. **Không chứa dữ liệu giá cổ phiếu.** |
| `final/News_raw.csv` | **Dataset cuối** sau khi gộp với dữ liệu của Thảo, **4.491 bài**, đã có sẵn cột `tickers`. |

### 1.6. Các thư mục khác

| Thư mục | Nội dung |
|---|---|
| `outputs/coverage/` | `coverage.csv` (nguồn × ticker), `crawl_report.csv`, `native_more_report.csv`, `firecrawl_scrape_report.csv` (trạng thái từng URL: `ok` / `rejected` / `fetch_error` + lý do), `final_native_validation.json` (kiểm định dataset crawl), `merge_thao_report.json` (thống kê merge). |
| `outputs/logs/` | Log và file trạng thái của 2 pipeline (`*_pipeline.log`, `*_pipeline_status.json`). |
| `logs/crawl.log` | Log của `run_backfill.py`. |
| `.firecrawl/` | Kết quả Firecrawl đã lưu: `search-*.json` (tìm theo nguồn), `ticker-search/` (tìm theo mã), `articles/` (484 bài đã scrape, đặt tên theo SHA-256 của URL). |
| `docs/` | Báo cáo chi tiết và mô tả các cột dữ liệu. |
| `tests/` | 6 test: cấu hình, ghi CSV, chuẩn hóa URL, kiểm tra domain giả mạo, quy tắc gắn ticker, lọc trang không phải bài báo. |

---

## 2. Quy trình crawl end-to-end

```
 config/ (sources, ticker aliases, ngày)
        │
        ▼
 [1] DISCOVERY ──── tìm URL ứng viên ─────────────► data/staging/news_candidates_*.csv
        │            (sitemap / RSS / seed page / tìm kiếm web)
        ▼
 [2] TẢI & BÓC TÁCH ─ title, summary, content, ngày đăng, tác giả, canonical URL
        │
        ▼
 [3] XÁC MINH ────── đủ nội dung? đúng khoảng ngày? có nhắc tới ticker?
        │            (không đạt → ghi lý do vào outputs/coverage/*_report.csv)
        ▼
 [4] GẮN TICKER ──── match_method + match_confidence
        │
        ▼
 [5] LOẠI TRÙNG ──── theo canonical URL, rồi theo content_hash
        │
        ▼
 [6] GHI OUTPUT ──── data/raw/news_raw.csv + data/final/news_ticker.csv
        │
        ▼
 [7] COVERAGE ────── outputs/coverage/coverage.csv
        │
        ▼
 [8] MERGE THẢO ──── data/final/News_raw.csv  ★
```

### Bước 1 — Discovery: tìm URL ứng viên

Mục tiêu là thu được danh sách URL *có khả năng* là bài báo về một trong 14 mã. Có 3 cách, đều ghi ra cùng định dạng `source, ticker, url, title, description, matched_text, match_method, match_confidence, search_error`:

**a. Native (`discover_native.py`) — cách chính**
1. Với mỗi nguồn trong `sources.csv`, đưa vào hàng đợi: `robots.txt`, các đường dẫn feed phổ biến (`/rss`, `/rss.xml`, `/feed`, `/feed.xml`, `/sitemap.xml`, `/sitemap_index.xml`, `/sitemap-index.xml`) và `seed_url`.
2. Tải song song theo lô 32 URL:
   - `robots.txt` → lấy các dòng `Sitemap:` đưa tiếp vào hàng đợi.
   - Sitemap index → đưa các sitemap con vào hàng đợi, **bỏ qua sitemap có năm rõ ràng nằm ngoài 2020–2026**.
   - Sitemap / RSS / Atom → đọc từng mục (`loc`/`link`, `title`, `lastmod`/`pubDate`).
   - Trang HTML → đọc mọi thẻ `<a href>`.
3. Với mỗi URL tìm được, loại bỏ nếu: đã có trong `news_raw.csv`, khác domain của nguồn, là trang chủ, hoặc là trang dữ liệu chứng khoán (`/du-lieu/`, `/gia-co-phieu`, `/lich-su-giao-dich`, `/co-dong`, `/bao-cao-tai-chinh`, `/profile/`).
4. Chỉ giữ URL mà tiêu đề hoặc đường dẫn có nhắc tới ticker (xem quy tắc ở Bước 4).
5. Mọi URL đã duyệt được lưu vào bảng `visited` trong `native_discovery.sqlite` → chạy lại sẽ tiếp tục từ chỗ dừng.

**b. Open web (`discover_open_web.py`)**
Sinh truy vấn cho từng ticker × từng tên công ty × từng năm 2020–2026 (ví dụ `"Thế Giới Di Động" tin tức 2023`), gửi lên DuckDuckGo (2 luồng, nghỉ 0,8 giây giữa các lô), giữ kết quả có nhắc tới ticker. Truy vấn đã chạy được lưu trong `open_web_discovery.sqlite`.

**c. Firecrawl (`discover_firecrawl.py`, `discover_by_ticker.py`)**
Dùng ở giai đoạn đầu, cần cài Firecrawl CLI + Node.js. Không bắt buộc cho pipeline native.

### Bước 2 — Tải và bóc tách bài báo

`scrape_native_candidates.py` gom ứng viên theo URL (một URL có thể ứng với nhiều ticker), bỏ URL đã có trong `news_raw.csv`, rồi tải song song (mặc định 12 luồng trong pipeline). Mỗi request dùng User-Agent trình duyệt, `Accept-Language: vi-VN`, timeout 15 giây, thử lại khi gặp lỗi 408/429/5xx. Chỉ nhận nội dung HTML/XML.

Các trường được bóc tách theo thứ tự ưu tiên:

| Trường | Lấy từ |
|---|---|
| `title` | `og:title` → `twitter:title` → `<h1>` → JSON-LD `headline` |
| `summary` | `meta description` → `og:description` → phần tử class `sapo`/`summary`/`description` → JSON-LD |
| `content` | Các đoạn `<p>`, `<h2>` trong `<article>` → trong khối class `detail-content`/`article-content`/`news-content`/`content-detail` → trong `<main>` → JSON-LD `articleBody` |
| `published_date` | `article:published_time` → `pubdate` → `datePublished` → `<time datetime>` → JSON-LD → ngày dạng `dd/mm/yyyy` trong phần tử class `date`/`time`/`publish`. Chuẩn hóa bằng `dateutil` (ưu tiên ngày trước tháng). |
| `author`, `category` | Các thẻ meta tương ứng → JSON-LD |
| `url` (canonical) | `<link rel="canonical">` → `og:url`; nếu canonical trỏ sang domain khác thì giữ URL gốc |

### Bước 3 — Xác minh bài hợp lệ

Bài chỉ được nhận khi **đủ tất cả** điều kiện:
1. Tải thành công.
2. Tiêu đề ≥ 8 ký tự.
3. Nội dung ≥ 200 ký tự.
4. Có ngày đăng đọc được và nằm trong 2020-01-01 → 2026-09-25.
5. Có bằng chứng liên quan tới ticker khi kiểm tra lại trên **tiêu đề + tóm tắt + 6.000 ký tự đầu của nội dung** (không xét phần cuối trang để tránh khớp nhầm vào khung "tin liên quan").

Bài không đạt bị ghi vào `outputs/coverage/native_more_report.csv` với trạng thái `fetch_error` hoặc `rejected` và lý do. **Không tự điền hay tạo nội dung thay thế.**

### Bước 4 — Gắn ticker

Mục đích: xác định **mỗi bài báo nói về công ty nào** trong 14 mã (gắn nhãn cho bài báo). Bước này **không** dùng hay nối với dữ liệu giá cổ phiếu.

Hàm `match_tickers` (trong `discover_firecrawl.py`) so khớp sau khi bỏ dấu tiếng Việt và chuyển chữ thường:

| `match_method` | Điều kiện | `match_confidence` |
|---|---|---|
| `company_alias` | Tiêu đề/mô tả chứa tên công ty hoặc thương hiệu trong `ticker_aliases.json` | high |
| `ticker_with_market_context` | Tiêu đề/mô tả chứa mã (ví dụ `MWG`) **và** từ khóa thị trường: "cổ phiếu", "mã chứng khoán", "chứng khoán", "HOSE", "HNX", "UPCoM" | high |
| `ticker_in_url_with_market_context` | Mã xuất hiện trong đường dẫn URL **và** tiêu đề/mô tả có từ khóa thị trường | medium |
| `alias_in_url` | Mã hoặc tên công ty xuất hiện trong đường dẫn URL (chỉ dùng ở bước discovery native) | medium |

Mã đứng một mình không kèm ngữ cảnh chứng khoán sẽ **không** được tính, để tránh nhầm (ví dụ "IMP" trong một từ tiếng Anh).

### Bước 5 — Loại trùng

1. **Chuẩn hóa URL:** chữ thường cho scheme/host, bỏ `#fragment`, bỏ tham số theo dõi `utm_*`, `fbclid`, `gclid`.
2. **`news_id`** = SHA-256 của canonical URL → mỗi URL chỉ một dòng.
3. **`content_hash`** = SHA-256 của `title + summary + content` (đã gộp khoảng trắng). Hai URL khác nhau nhưng cùng nội dung → chỉ giữ một bài, các dòng ticker của bài bị loại được chuyển sang bài được giữ.

### Bước 6 — Ghi output

- `data/raw/news_raw.csv`: bài mới được **gộp** với bài cũ, sắp xếp theo ngày đăng.
- `data/final/news_ticker.csv`: nhãn mã cổ phiếu của từng bài, mỗi cặp (`news_id`, `ticker`) một dòng.
- Ghi qua file tạm `.tmp` rồi mới thay thế → không hỏng file nếu bị ngắt giữa chừng.

### Bước 7 — Coverage

`build_coverage.py` tạo ma trận 51 nguồn × 14 mã với số bài, ngày đầu/cuối, trạng thái `covered`/`missing`. Hiện có 232/714 cặp có dữ liệu. Ngoài ra `outputs/coverage/final_native_validation.json` xác nhận: 3.633 bài, 3.633 URL và 3.633 `content_hash` duy nhất, 0 bài thiếu trường bắt buộc, 0 dòng ticker mồ côi.

### Bước 8 — Merge dữ liệu của Thảo

`merge_thao_news.py` gộp dữ liệu crawl với file `C:\Users\DELL\Downloads\DATA_THAO - NEWS_RAW.csv`:

1. Nạp bài crawl, gắn cột `tickers` từ `news_ticker.csv`, đánh dấu `dataset_origin = main_crawler`.
2. Lọc file của Thảo, chỉ giữ dòng có `crawl_status = success`, có `content`, và `published_date` trong khoảng 2020-01-01 → 2026-09-25.
3. Gom các dòng của Thảo theo URL đã chuẩn hóa (bỏ `www.`, `utm_*`, `/` cuối…). Trong một nhóm trùng: giữ bản có nội dung dài nhất, hợp nhất `tickers` (chỉ nhận 14 mã hợp lệ), `keywords`, `event_type`.
4. **URL đã có trong dữ liệu crawl** → giữ nguyên bài crawl, chỉ bổ sung `tickers` và điền các trường còn trống (`industry_group`, `keywords`, `event_type`, `note`); `dataset_origin = main_crawler|thao_crawler`.
5. **URL mới** → thêm bài, tính lại `news_id` và `content_hash` bằng SHA-256; `dataset_origin = thao_crawler`.
6. Ghi `data/final/News_raw.csv` và thống kê vào `outputs/coverage/merge_thao_report.json`.

Kết quả lần chạy gần nhất:

| Chỉ số | Giá trị |
|---|---:|
| Bài crawl | 3.633 |
| Dòng trong file của Thảo | 1.372 |
| Bị loại: crawl không thành công | 81 |
| Bị loại: ngoài khoảng ngày | 19 |
| Dòng trùng nội bộ trong file của Thảo | 54 |
| Bài crawl được bổ sung thông tin từ Thảo | 360 |
| Bài mới thêm từ Thảo | 858 |
| **Tổng bài trong `News_raw.csv`** | **4.491** |

---

## 3. Cách chạy

### Cài đặt

```bash
cd news_crawler
python -m pip install -r requirements.txt
```

### Chạy toàn bộ

```bash
# Bước 1–7: crawl (discover → scrape → coverage → tests)
python run_native_news_pipeline.py

# (tuỳ chọn) crawl thêm qua tìm kiếm web mở
python run_open_web_pipeline.py

# Bước 8: merge với dữ liệu của Thảo → data/final/News_raw.csv
python -m src.cleaning.merge_thao_news
```

### Chạy từng bước riêng

```bash
python -m src.crawler.discover_native
python -m src.crawler.scrape_native_candidates --candidates data/staging/news_candidates_native.csv --workers 12
python -m src.quality.build_coverage
python -m src.cleaning.merge_thao_news
python -m pytest -q
```

Theo dõi tiến độ: xem `outputs/logs/native_pipeline_status.json` (bước đang chạy, `running`/`complete`/`failed`) và `outputs/logs/native_pipeline.log`.

**Chạy lại (incremental):** pipeline native tự bỏ qua URL đã có trong `news_raw.csv` và URL đã duyệt trong file `.sqlite`, nên chạy lại chỉ lấy thêm bài mới.

---

## 4. Lưu ý quan trọng

1. **Dữ liệu cũ không bị ghi đè.** `scrape_native_candidates.py` và `scrape_firecrawl.py` đều gộp thêm qua `store.merge_into_dataset` (bài đã có luôn được giữ). `run_backfill.py` chỉ ghi ra `data/staging/news_backfill.csv` và báo lỗi nếu `--output` trỏ vào `news_raw.csv`. Folder không dùng git, nên vẫn nên sao lưu `data/` trước khi crawl lớn.
2. **Đường dẫn file của Thảo được ghi cứng** trong biến `THAO` ở `src/cleaning/merge_thao_news.py`. Nếu file nằm chỗ khác thì sửa biến này.
3. `News_raw.csv` tên là "raw" nhưng thực chất đã gộp, lọc và loại trùng — đây là file nên dùng cho các bước sau.
4. Các cột `industry_group`, `keywords`, `event_type`, `note` chỉ có ở phần dữ liệu đến từ Thảo.
5. Crawler HTTP không lấy được trang chặn bot, cần JavaScript, đăng nhập, CAPTCHA hoặc đã bị xóa. 51 nguồn cấu hình không đảm bảo đã lấy được mọi bài từng đăng.
6. Firecrawl (`discover_firecrawl.py`, `discover_by_ticker.py`, `scrape_firecrawl.py`) cần cài Firecrawl CLI và Node.js, bị giới hạn 10 request/phút.

Mô tả chi tiết từng cột dữ liệu: `docs/data_dictionary.md`. Báo cáo đầy đủ: `docs/BAO_CAO_CAU_TRUC_VA_CRAWL_NEWS.md`.
