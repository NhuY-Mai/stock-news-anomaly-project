"""Reusable helpers for cleaning the raw news dataset."""

from __future__ import annotations

import ast
import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable
from html.parser import HTMLParser

import pandas as pd

from .validation_utils import canonicalize_url, normalize_title, parse_timestamp, split_tickers


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")

REVIEWED_CONTENT_RULE = "validation_content_review_v2"

# Exact article bodies reviewed on 25/10/2026. These hashes deliberately keep
# the corrections out of unrelated articles and refuse unreviewed new versions.
REVIEWED_CONTENT_CORRECTIONS = {
    "0924ce353ae49d611b69c2d14935624b6023ff60543e4208a02bdb20d6d65d62": {
        "old_sha256": "8bb9effb172843d76043ecbc20fb1064e0086ee50eef78d19f48e0979f777263",
        "new_sha256": "eb2003c9d05d18bae8dbe7a83c0447fbbe7e3732d954e3403f31b449d278e0d6",
        "operation": "remove_prefix",
        "boundary": "Ngày 4/12/2025, Công ty Cổ phần Hàng tiêu dùng Masan",
    },
    "8a622ba716429d810d76c613fdb3b8d163c0018af47193379b2326c4287854d9": {
        "old_sha256": "55ee78d17e1e4e553480b0097b95fb14481fb7f8591d59aab562d290bc1ae485",
        "new_sha256": "7bd7d67eb4aa87bde75c2674f9544766c87e82da6ad226adc36c5f4576e6475f",
        "operation": "remove_suffix",
        "boundary": (
            " Sản phẩm được sản xuất và phân phối bởi Công ty Cổ phần Dược Hậu Giang."
            " Đọc kỹ hướng dẫn sử dụng trước khi dùng. GPQC: 24e/2021/XNQC/QLD Doãn Phong"
        ),
    },
    "62029430c4b97773069ee32d35b42c142f819d89aed26c9dfcc55ebd17da95ab": {
        "old_sha256": "e9d9e9ecbb9b66680ae5efd5d006e332e1f30bb857ce6e42f607ccaecfcf39b3",
        "new_sha256": "a16853192cd0bc1947b95eb85bfa513c8abe9afc2362dd1d84ab12bd44c0e807",
        "operation": "remove_suffix",
        "boundary": (
            " * Lợi nhuận ròng quý 2 gần 3.2 ngàn tỷ, Vinamilk giữ vững vị thế tiền mặt"
            " ròng hơn 18 ngàn tỷ Thanh Tú FILI - 14:43 07/08/2026"
        ),
    },
}

# These patterns are specific interface fragments observed in the raw dataset.
# Generic words such as "đăng ký" are deliberately not removed because they
# frequently carry meaning in financial news.
HIGH_CONFIDENCE_BOILERPLATE_PATTERNS = [
    re.compile(r"^Bỏ qua điều hướng\s*", flags=re.IGNORECASE),
    re.compile(
        r"This site is protected by reCAPTCHA and the Google Privacy Policy "
        r"and Terms of Service apply\.?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"reCAPTCHA\s+Recaptcha requires verification\.\s*"
        r"protected by reCAPTCHA",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"Đăng nhập Đăng ký Email Mật khẩu Quên mật khẩu\? Đăng nhập "
        r"Email Mật khẩu Nhập lại mật khẩu Nhập mã xác thực Đăng ký "
        r"Khi nhấn đăng ký, bạn đã đồng ý với điều khoản của báo VnBusiness\. "
        r"Khôi phục mật khẩu Nhập email bạn đã đăng ký Chúng tôi sẽ gửi mã "
        r"OTP đến email của bạn đã đăng ký\.?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"Loading\.\.\. Top Bạn cần đăng nhập để thực hiện chức năng này! "
        r"Đăng nhập Bạn không thể gửi bình luận liên tục\. "
        r"Xin hãy đợi 60 giây nữa\.?",
        flags=re.IGNORECASE,
    ),
]

# Heading boundaries verified against this crawl. They are deliberately scoped
# to publishers: a normal sentence containing "đăng ký" is never a boundary.
SOURCE_FOOTER_MARKERS = {
    "BNews": ("Tin liên quan", "Tin cùng chuyên mục", "Tin nổi bật"),
    "MWGOfficial": ("Bài viết liên quan", "Related articles"),
    "NguoiQuanSat": ("Bài liên quan", "Link bài gốc", "Tags:", "Trang TTĐTTH của Công ty"),
    "MarketTimes": ("Bài liên quan", "Gửi bình luận", "Tin liên quan", "Cơ quan của Hội Thẩm định giá"),
    "VnBusiness": ('"Gửi email")', "Bài liên quan", "Tags:", "Cùng chuyên mục", "Đăng nhập Đăng ký Email Mật khẩu"),
    "24HMoney": ("Báo cáo Nhà đầu tư lưu ý Theo dõi 24HMoney trên", "Bấm vào mỗi từ khóa để xem bài cùng chủ đề", "Từ khóa liên quan", "Kết nối truyền thông cùng 24HMONEY", "Cơ quan chủ quản: Công ty TNHH 24HMoney", "Đang tải... Không có dữ liệu Đã xem hết tin", "✅ Nếu bạn thấy bài viết hữu ích"),
    "CafeF": ("Đọc thêm", "Copy link Link bài gốc", "CÙNG CHUYÊN MỤC", "Cùng chuyên mục Xem theo ngày", "Tags:", "Công tyTin tứcLãnh đạo Trở lên trên", "PRIVATE 100 và VNTAX 200 của năm 2026 (dựa trên số liệu thực nộp năm 2025):"),
    "CafeBiz": ("Đọc thêm", "Copy link Link bài gốc", "Cùng chuyên mục Xem theo ngày"),
    "DoanhNhanVN": ("Bài viết liên quan", "Có thể bạn quan tâm", "button share.zalo.me"),
    "DoanhNghiepVN": ("Copy link Link bài gốc", "Từ khóa:", "End of content", "Cột tin quảng cáo"),
    "ZNEWS": ("Tri Thức Znews giới thiệu độc giả Tủ sách", "Bài liên quan", "Chúng tôi sử dụng cookie"),
    "VnEconomy": ("Đọc bài theo từ khoá", "Đọc bài theo từ khóa", "Bài liên quan", "Mới nhất trên VnEconomy"),
    "VOV": ("Mời quý độc giả theo dõi VOV.VN trên", "Thêm VOV trên Google", "Viết bình luận Gửi tin", "Tin liên quan"),
    "DanTri": ("Thêm Dân trí trên Google", "Bình luận (", "Đọc nhiều trong", "Tags:"),
    "ThanhNien": ("Tin liên quan", "Bình luận (", "Khám phá thêm chủ đề"),
    "TuoiTre": ("Tối đa: 1500 ký tự", "Hiện chưa có bình luận nào"),
    "SucKhoeDoiSong": ("Quan tâm nhấtMới nhất", "Quan tâm nhất Mới nhất", "Ý kiến của bạn Hãy nhập", "Link bài gốcCopy Link", "Gửi bình luận Bình luận của bạn", "button share.zalo.me", "Chia sẻ facebook twitter Bình luận ("),
    "VietnamFinance": ("Google News", "Tags:", "TIN LIÊN QUAN"),
    "VietnamNet": ("Xem thêm về:",),
    "VnExpress": ("Chọn VnExpress làm nguồn ưu tiên trên Google Search", "Ý kiến ("),
    "TapChiTaiChinh": ("Tin tức khác", "Xem phiên bản di động Twitter Widget Iframe"),
    "StockBiz": ("Đang tải nội dung... Tải tệp đính kèm", "1d1m3m6m1y3y5y", "THUẬT NGỮ TRONG BÀI", "Sao chép liên kết để chia sẻ bài viết", "Cùng chuyên mục"),
    "Vietcap": ("Powered by Froala Editor",),
    "ImexpharmOfficial": ("Thông báo: Sự kiện công bố KQKD",),
    "TheLeader": (
        "Chuỗi điện máy Erablue hiện đang dần hoàn thiện mô hình kinh doanh",
        "Lãnh đạo Pharma Group, ông Burak Pekmezci tin rằng",
        "Giá vàng hôm nay 30/5 tăng 1 triệu đồng",
        "MWG PayLater - Dịch vụ mua trước trả sau",
        "Thế Giới Di Động đang cho thấy sức bật mạnh mẽ sau giai đoạn tái cấu trúc",
        "SSI, BVSC, VDSC, VCBS vừa dự báo kết quả kinh doanh quý III/2025",
        "Vinamilk đứng đầu danh sách “Top 25 thương hiệu dẫn đầu",
        "Tổng doanh thu hợp nhất quý IV/2025 của Vinamilk đạt hơn 17.000 tỷ đồng",
        "Nhiều tin vui đến với doanh nghiệp thủy sản ngay đầu năm 2026",
        "Ngân hàng TMCP Quốc tế Việt Nam (VIB) công bố kết quả kinh doanh sáu tháng đầu năm 2026",
        "Một trong những điểm đáng chú ý trong báo cáo phân tích mới nhất của VCBS",
        "Hội đồng quản trị Công ty cổ phần Masan High-Tech Materials đã thông qua việc tổ chức lấy ý kiến cổ đông bằng văn bản",
        "50 năm sau ngày thành lập, Vinamilk không chỉ ghi dấu hành trình",
        "Hòa mình trong màn đồng diễn quy mô tại Hà Nội",
        "50 năm của Vinamilk là hành trình liên tục nâng chuẩn",
    ),
    "NhaDauTu": (
        "Tài chính Tài chính Tài chính Tài chính",
        "Tài chính Tài chính Tài chính",
        "Doanh nghiệp Doanh nghiệp Doanh nghiệp Doanh nghiệp",
    ),
}

# Three-card current-news widgets in the 30 full CafeF pages. Exact strings are
# retained here because the flattened crawler output contains no link tags.
CAFEF_WIDGET_HEADLINES = (
    'Cổ phiếu PNJ "có biến"',
    "VinFast sắp nhận sáp nhập một công ty vốn 136.000 tỷ đồng",
    "Dự báo lợi nhuận ngân hàng quý III: Một nhà băng được kỳ vọng lãi gấp rưỡi cùng kỳ, MB, Techcombank, VPBank Vietcombank, VietinBank và BIDV có thể tăng trưởng hai con số",
    "Novaland làm sao thế?",
    "Người phụ nữ trình báo mất 2,4 tỷ đồng cất trong tủ quần áo: Ngỡ ngàng danh tính thủ phạm",
    "IR Awards 2026: Dấu ấn minh bạch trong hành trình tăng trưởng của Masan",
    "Doanh nghiệp đua trả cổ tức, cao nhất lên tới 50%",
    'Chứng khoán phát tín hiệu mới, CTCK chỉ ra nhóm cổ phiếu có thể "hút tiền" nhờ câu chuyện riêng',
    "Thị trường sau nâng hạng: Cơ hội nào cho dòng tiền đang chờ giải ngân?",
    "Giải mã nội lực tăng trưởng của LPBank trong chu kỳ mới",
    "Lãi lớn tại Việt Nam, một ngân hàng liên tục chuyển lợi nhuận khủng về nước",
    'muốn bán cổ phiếu riêng lẻ giá cao hơn 60% thị giá") PVcomBank (PCB) muốn bán cổ phiếu riêng lẻ giá cao hơn 60% thị giá muốn bán cổ phiếu riêng lẻ giá cao hơn 60% thị giá")',
    "VietinBank lần thứ 5 bán cổ phiếu Cảng Sài Gòn",
    "Chủ tịch AgriS Đặng Huỳnh Ức My gom thành công 11 triệu cổ phiếu SBT",
    "Doanh nghiệp bất động sản loay hoay tìm đường xoay vốn",
    "Việt Nam nâng hạng thị trường chứng khoán: Cơ hội và thách thức nào cho doanh nghiệp Việt?",
    "Ưu đãi thuế 3 năm cho doanh nghiệp nhỏ và vừa, ngân sách chi 10.000 tỷ đồng/năm",
    "Trúng đậm cá cơm trái vụ, ngư dân Quảng Ngãi 'bỏ túi' bạc triệu mỗi ngày",
    "Địa phương đầu tiên công bố tăng trưởng GRDP ước tính 9 tháng đầu năm 2026",
    "Lịch chốt quyền trả cổ tức trong tuần từ ngày 28/9 2/10, tỷ lệ cao nhất 50%",
    "Điều tra loạt giao dịch tổng 2,8 tỷ đồng, công an khởi tố Vũ Quang Huy SN 1991 và 2 bị can",
    "Chủ tịch Kosy Group giải trình gì khi cổ phiếu giảm sàn 5 phiên liên tiếp",
    "PNJ đã vay 700 tỷ từ gia đình bà Cao Thị Ngọc Dung, tính đóng 30 cửa hàng sau cú sốc “kim cương”",
    "Con trai tỷ phú Nguyễn Thị Phương Thảo đóng vai trò gì trong việc đưa Wi Fi của Elon Musk lên 120 máy bay Vietjet?",
    "Chân dung cháu gái tỷ phú Jensen Huang ngồi cùng bàn nghìn tỷ USD với Chủ tịch Trung Quốc Tập Cận Bình",
)

# Additional recommendations verified in the raw pages, absent from the ticker
# corpus itself. These may ONLY be removed as exact, anchored trailing titles.
CAFEF_VERIFIED_TRAILING_HEADLINES = (
    "Profile các nhà tư vấn tái cấu trúc Novaland: 1 công ty tư vấn toàn cầu, 1 hãng luật góp mặt tại loạt thương vụ M&A tỷ đô của Masan, VPBank",
    'Top 10 Công ty Thức ăn chăn nuôi uy tín năm 2022: C.P Việt Nam tiếp tục dẫn đầu, một "ông lớn" từ Hà Lan bứt phá ấn tượng còn Dabaco đuối sức',
    'Top người giàu tuần đầu năm: Vị trí số 2 cạnh tranh quyết liệt giữa Chủ tịch Sunshine Homes, Techcombank, Masan và Hòa Phát, "người nhà" VPBank áp đảo',
    "19 gương mặt công bố ước tính lợi nhuận quý 4: BIDV lãi gấp đôi trong khi Vietinbank giảm lãi 20%, Vietnam Airlines, BSR, Hải An dự kiến lỗ lớn",
    'Lão bà Kỷ Mão lớn tuổi nhất sàn chứng Việt: DN cán đích lợi nhuận ngay quý đầu niên độ, trả cổ tức cả trăm tỷ, EPS top đầu, cổ phiếu trụ vững trong CLB "giá 3 chữ số"',
    "Cập nhật BCTC quý 4 ngày 30/1: Dabaco lần đầu báo lỗ kể từ quý 3/2017, Habeco ghi nhận LNTT quý 4 tăng 86 lần cùng kỳ",
    "Ông Nguyễn Đức Tài: Từ 2 lần khởi nghiệp thất bại tới 'giấc mộng' 10 tỉ USD của MWG",
    "Masan Group báo lãi ròng 3.567 tỷ đồng trong năm 2022, mở mới 730 siêu thị WinMart+, doanh thu Phúc Long vươn gần 1.600 tỷ đồng",
    "Digiworld (DGW) tự tin với mức tăng trưởng 2 chữ số, BVSC lại đánh giá: Khó bứt phá",
    'Chủ tịch Thế giới Di Động "trải lòng" với cổ đông và nhân viên: Tâm lý tiêu dùng tiết kiệm, MWG sẽ chấm dứt những thử nghiệm không có tiềm năng tăng trưởng',
)

# Legal declarations and contact appendices, NOT the investment analysis above.
PDF_APPENDIX_MARKERS = (
    "CAM KẾT PHÂN TÍCH",
    "Xác nhận của chuyên viên phân tích",
    "Update Report Hệ thống khuyến nghị của Vietcap",
    "Update Report Liên hệ",
    "Update Report Khuyến cáo",
)


class _VisibleHTMLText(HTMLParser):
    """Keep visible text while ignoring HTML scripts, styles and comments."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "iframe"}:
            self.hidden_depth += 1
        self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "iframe"} and self.hidden_depth:
            self.hidden_depth -= 1
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def _article_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    for _ in range(8):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    # Only invoke an HTML parser for actual tags, not comparisons such as <5%.
    if re.search(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?/?>", text):
        parser = _VisibleHTMLText()
        parser.feed(text)
        parser.close()
        text = " ".join(parser.parts)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://www\.w3\.org/2000/svg[\"'/>]*", " ", text)
    text = ZERO_WIDTH_PATTERN.sub("", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _publisher_name(source: object) -> str:
    name = "" if source is None or pd.isna(source) else str(source)
    if name.startswith("API:CafeF"):
        return "CafeF"
    return {"RSS:Thanh Niên": "ThanhNien", "RSS:Dân trí": "DanTri", "RSS:VnExpress": "VnExpress"}.get(name, name)


def _cut_footer(text: str, publisher: str) -> str:
    boundaries = [
        position for marker in SOURCE_FOOTER_MARKERS.get(publisher, ())
        if (position := text.find(marker)) >= (200 if publisher == "TheLeader" else 0)
    ]
    return text[:min(boundaries)].strip() if boundaries else text


def _strip_title_header(text: str, title: object) -> str:
    """Remove only a publisher header ending in an observed timestamp."""
    title_text = _article_text(title)
    if not title_text or title_text not in text[:1800]:
        return text
    title_end = text.find(title_text) + len(title_text)
    timestamp = re.match(
        r".{0,160}?(?:\d{1,2}[:h]\d{2}\s*[\\|]?\s*\d{1,2}/\d{1,2}/\d{4}|"
        r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?:\s*\(?GMT\+7\)?)?)\s*",
        text[title_end:],
    )
    return text[title_end + timestamp.end():].strip() if timestamp else text


def parse_serialized_list(value: object) -> list[object] | None:
    """Parse only genuine Python list/tuple strings; leave text like '[PDF]' alone."""
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)

    text = str(value).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return list(parsed) if isinstance(parsed, (list, tuple)) else None


def select_scalar_value(value: object) -> object:
    """Return the first non-empty item from a serialized list or the scalar itself."""
    parsed_values = parse_serialized_list(value)
    if parsed_values is not None:
        candidates = [str(item).strip() for item in parsed_values if str(item).strip()]
        return candidates[0] if candidates else pd.NA

    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def normalize_inline_text(value: object) -> object:
    """Normalize Unicode and whitespace for one-line text fields."""
    scalar_value = select_scalar_value(value)
    if pd.isna(scalar_value):
        return pd.NA

    text = unicodedata.normalize("NFKC", str(scalar_value))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text if text else pd.NA


def clean_article_content(
    value: object,
    source: object = None,
    title: object = None,
    summary: object = None,
    known_article_titles: Iterable[str] = (),
) -> object:
    """Keep article prose, removing only verified publisher-specific UI boundaries.

    Missing article bodies are not reconstructed from a headline or another news
    card. ``assess_content_quality`` identifies these records before export.
    """
    text = _article_text(value)
    if not text:
        return pd.NA
    publisher = _publisher_name(source)
    source_text = "" if source is None or pd.isna(source) else str(source)
    title_text = _article_text(title)
    summary_text = _article_text(summary)
    original_text = text
    article_author = ""
    if publisher == "CafeF" and title_text and text.startswith(title_text):
        author_match = re.match(
            r"\s*(?:Theo\s+)?([^|\\]{2,70}?)\s*[|\\]\s*(?:\d{1,2}[- /]){2}\d{4}",
            text[len(title_text):],
        )
        article_author = author_match.group(1).strip() if author_match else ""

    # This raw VnExpress fragment is JavaScript, not a sentence. Its exact end
    # matters because valid article paragraphs follow it in the same string.
    text = re.sub(
        r'try\{try\{for\(const iframe of document\.querySelectorAll\("iframe\[data-src\]"\)\)'
        r'.*?console\.log\("error_replace_script",e\);\}',
        " ", text,
    )
    text = re.sub(r"Created with Highcharts\s+\d+(?:\.\d+)+", " ", text)
    for pattern in HIGH_CONFIDENCE_BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    # The raw crawl inserts standalone backslashes between UI words. Normalize
    # them BEFORE testing a publisher header, not only at the final export.
    text = re.sub(r"(?:^|\s)\\+(?=\s|$)", " ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    if publisher == "BNews":
        text = re.sub(r"^BNEWS\s+", "", text)
    elif publisher == "NguoiQuanSat" and text.startswith("Market Watch list"):
        # The title is used as an anchor so navigation dates cannot select a
        # false start. PDF reports without this page shell are untouched.
        text = _strip_title_header(text, title)
    elif publisher == "MarketTimes":
        text = _strip_title_header(text, title)
    elif publisher == "VietnamFinance" and text.startswith("home Tài chính"):
        text = _strip_title_header(text, title)
        if text.startswith("home Tài chính"):
            text = re.sub(
                r"^home Tài chính.*?\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+\(GMT\+7\)\s*",
                "", text,
            )
        inline_related = re.search(
            r"TIN LIÊN QUAN (.{10,300}?) \d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}\s*",
            text[:1500],
        )
        if inline_related:
            related_title = inline_related.group(1)
            text = text[:inline_related.start()] + text[inline_related.end():]
            repeated_card = text.find(f"{related_title} {related_title}")
            if repeated_card > 200:
                text = text[:repeated_card].strip()
        related_card = re.search(
            r"(?P<headline>[^.!?]{12,240}?) (?P=headline) Tài chính\d+ ngày trước \(VNF\)",
            text,
        )
        if related_card and related_card.start() > 200:
            text = text[:related_card.start()].strip()
        text = re.split(r"(?:\\\s*)?Google News\s*(?:\\\s*)?Tiktok", text, maxsplit=1)[0].strip()
    elif publisher == "VOV":
        text = re.sub(r"^Nhảy đến nội dung .*?Facebook Copy link thành công!\s*VOV\.VN\s*[-–]?\s*", "", text)
    elif publisher == "DanTri":
        if "(Dân trí)" in text[:1500]:
            text = text[text.index("(Dân trí)") + len("(Dân trí)"):].lstrip(" -–")
    elif publisher == "ZNEWS":
        text = re.sub(r"^Bạn có thể chuyển sang phiên bản mobile rút gọn .*?nếu mạng chậm\.\s*Đóng\s*", "", text)
    elif publisher == "TapChiTaiChinh":
        text = re.sub(r"^.*?Theo dõi Tapchikinhtetaichinh\.vn trên\s*", "", text)
    elif publisher == "StockBiz":
        marker = "Tóm tắt tin tức bằng AI"
        if marker in text[:1500]:
            text = text[text.index(marker) + len(marker):].strip()
    elif publisher == "SucKhoeDoiSong" and text.startswith("Bỏ qua đến nội dung chính"):
        text = re.sub(
            r"^Bỏ qua đến nội dung chính.*?Theo dõi Gia đình và Xã hội trên\s*\\*\s*News\s*",
            "", text,
        )
    elif publisher == "VnExpress" and text.startswith("50 năm phụng sự khát vọng Việt"):
        text = re.sub(
            r"^.*?\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}\s*\(GMT\+7\)\s*",
            "", text, count=1,
        )
        # This exact related card starts after the reporter's closing byline.
        text = re.split(
            r"(?<=Thi Hà) Vinamilk ghi nhận doanh thu cao kỷ lục Năm 2025, doanh thu bán hàng và cung cấp dịch vụ",
            text, maxsplit=1,
        )[0].strip()
    elif publisher == "24HMoney":
        approved = text.find("24HMONEY đã kiểm duyệt")
        title_start = text.find(title_text, approved) if title_text and approved >= 0 else -1
        if title_start >= 0:
            text = text[title_start + len(title_text):].strip()
        text = re.sub(r"^Mã liên quan\s*(?:Thành viên gắn bó\s*)?", "", text)
        text = re.sub(r"^Thành viên gắn bó\s+", "", text)
        text = text.replace("Được tài trợ", " ")

    if publisher == "CafeF" and "TIN MỚI" in text:
        before_news, _, after_news = text.partition("TIN MỚI")
        # API pages place dated recommendations/chart metadata before this
        # heading, followed by the actual article. Do not cut arbitrary prose
        # merely because it happens to contain the same two words.
        api_header = source_text.startswith("API:CafeF") and (
            "Chia sẻ" in before_news or "Xem hồ sơ doanh nghiệp" in before_news
        )
        widget_body = after_news.strip()
        removed_cards = 0
        while widget_body.startswith(CAFEF_WIDGET_HEADLINES):
            card = next(card for card in CAFEF_WIDGET_HEADLINES if widget_body.startswith(card))
            widget_body = widget_body[len(card):].strip()
            removed_cards += 1
        full_page_header = (
            source_text == "CafeF" and "Chia sẻ" in before_news
            and (title_text and title_text in before_news or "Xem hồ sơ doanh nghiệp" in before_news)
            and removed_cards >= 2
        )
        if api_header or full_page_header:
            text = widget_body if removed_cards >= 2 else after_news.strip()
            if summary_text and not text.startswith(summary_text):
                text = f"{summary_text} {text}".strip()
    if publisher == "CafeBiz" and "Theo dõi Cafebiz.vn trên Liên hệ" in text[:1800]:
        text = text.split("Theo dõi Cafebiz.vn trên Liên hệ", 1)[1].strip()

    if publisher == "ThanhNien":
        # The Google preference tutorial is inline at the top and repeats at
        # the end. Decode entities before this step to recover Vietnamese prose.
        text = re.sub(
            r'Chia sẻ Chọn Báo Thanh Niên làm nguồn ưu tiên trên Google tìm kiếm\. '
            r'Xem hướng dẫn\.\s*(?:BƯỚC 1.*?Chúc bạn đọc báo vui vẻ!\s*)?(?:-->)?',
            " ", text,
        )
        text = _strip_title_header(text, title)

    if publisher == "NguoiQuanSat" and title_text.startswith("[PDF]"):
        appendix_positions = [
            position for marker in PDF_APPENDIX_MARKERS
            if (position := text.find(marker)) > 200
        ]
        if appendix_positions:
            text = text[:min(appendix_positions)].strip()

    if publisher == "CafeF" and "Copy link Link bài gốc" in text:
        # Byline is recovered from this page's header, never a generic "Theo"
        # sentence. The source attribution immediately before the copy widget
        # confirms that the final repeated author is a footer, not body prose.
        copy_position = text.index("Copy link Link bài gốc")
        attribution = text[:copy_position]
        author_position = attribution.rfind(" " + article_author + " ") if article_author else -1
        if author_position > 200 and "Theo " in attribution[author_position:]:
            text = attribution[:author_position].strip()
            if text.endswith(" Theo"):
                text = text[:-len(" Theo")].rstrip()
        elif " | " in original_text[:1200]:
            # A differently punctuated title may prevent the exact-title header
            # anchor. Recover the byline from its separate first date field.
            header_author = re.search(
                r"\s(?:Theo )?([A-ZĐ][^|\\]{1,45}?)\s*\|\s*\d{1,2}-\d{1,2}-\d{4}",
                original_text[:1200],
            )
            if header_author:
                author_words = header_author.group(1).strip().split()
                author_words = author_words[-3:] if len(author_words) > 3 else author_words
                recovered_author = " ".join(author_words)
                byline = re.search(
                    r"\s+(?:Theo\s+)?" + re.escape(recovered_author)
                    + r"\s+(?P<publication>[^.!?|]{2,65}?)\s+Theo\s+(?P=publication)\s*$",
                    attribution,
                    flags=re.IGNORECASE,
                )
                if byline and byline.start() > 200:
                    text = attribution[:byline.start()].strip()
    text = _cut_footer(text, publisher)
    if publisher == "CafeF":
        candidates = tuple(
            candidate for candidate in (*known_article_titles, *CAFEF_VERIFIED_TRAILING_HEADLINES)
            if isinstance(candidate, str) and len(candidate) >= 35 and candidate != title_text
        )
        # Flattened scrapes lose the link boundary of some final recommendations.
        # Only an exact known headline anchored at the END can be removed, and
        # at least 200 characters of preceding article prose must remain.
        while candidates and text.endswith(candidates):
            headline = max((candidate for candidate in candidates if text.endswith(candidate)), key=len)
            preceding = text[:-len(headline)].strip()
            if len(preceding) < 200 or not re.search(r"[.!?。][\"”’']?$", preceding):
                break
            text = preceding
    if publisher == "24HMoney":
        text = re.sub(r"\s+Thứ (?:hai|ba|tư|năm|sáu|bảy|Hai|Ba|Tư|Năm|Sáu|Bảy), ngày \d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [AP]M \(GMT\+7\)\s*$", "", text)
        text = re.sub(r"\s*-{10,}\s*PHÙNG HƯƠNG\b.*$", "", text)
        text = re.sub(r"^\(Chuyên viên Tư vấn Đầu tư\)\s*|\s*\(Chuyên viên Tư vấn Đầu tư\)$", "", text)
    if publisher == "MarketTimes":
        text = re.sub(r"\s+Facebook\s*$", "", text)
    # Capture full inline forms even when an earlier cleaner already removed
    # only the first half of the VnBusiness login panel.
    text = re.sub(r"Gửi mã OTP Quay lại Xác thực mã OTP.*$", "", text)
    text = re.sub(r"Loading\.\.\. Top Bạn cần đăng nhập.*$", "", text)
    text = re.sub(r"\s+reCAPTCHA\s*$", "", text)
    text = re.sub(r"(?:^|\s)\\+(?=\s|$)", " ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip(" \\|")
    return text if text else pd.NA


def clean_reviewed_article_content(news_id: object, content: object) -> object:
    """Apply only a reviewed article's exact prefix/suffix removal, idempotently.

    Run after ``clean_article_content``. Unknown IDs are untouched; changed
    versions of an approved article fail rather than being cut by a broad rule.
    """
    if not isinstance(news_id, str) or news_id not in REVIEWED_CONTENT_CORRECTIONS:
        return content
    if not isinstance(content, str):
        raise ValueError(f"Reviewed article {news_id} must have a text body.")
    correction = REVIEWED_CONTENT_CORRECTIONS[news_id]
    current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if current_hash == correction["new_sha256"]:
        return content
    if current_hash != correction["old_sha256"]:
        raise ValueError(f"Reviewed article {news_id} has an unapproved content version.")

    boundary = correction["boundary"]
    if correction["operation"] == "remove_prefix":
        boundary_position = content.find(boundary)
        if boundary_position <= 0:
            raise ValueError(f"Reviewed prefix boundary is missing for {news_id}.")
        cleaned = content[boundary_position:]
    elif correction["operation"] == "remove_suffix" and content.endswith(boundary):
        cleaned = content[:-len(boundary)].rstrip()
    else:
        raise ValueError(f"Reviewed suffix boundary is missing for {news_id}.")
    if hashlib.sha256(cleaned.encode("utf-8")).hexdigest() != correction["new_sha256"]:
        raise ValueError(f"Reviewed content result does not match the approved hash for {news_id}.")
    return cleaned


def apply_approved_content_corrections(
    news_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return corrected content and its audit log, preserving all other fields."""
    required_columns = {"news_id", "content"}
    if not required_columns.issubset(news_dataframe.columns):
        raise ValueError("Content correction requires news_id and content columns.")
    if not news_dataframe.index.is_unique:
        raise ValueError("Content correction requires a unique row index.")
    selected = news_dataframe["news_id"].isin(REVIEWED_CONTENT_CORRECTIONS)
    if news_dataframe.loc[selected, "news_id"].duplicated().any():
        raise ValueError("Reviewed article IDs must occur at most once.")

    corrected = news_dataframe.copy(deep=True)
    changes = []
    for index, row in news_dataframe.loc[selected].iterrows():
        original_content = row["content"]
        cleaned_content = clean_reviewed_article_content(row["news_id"], original_content)
        if cleaned_content == original_content:
            continue
        corrected.at[index, "content"] = cleaned_content
        changes.append({
            "news_id": row["news_id"], "rule_id": REVIEWED_CONTENT_RULE,
            "old_content_sha256": hashlib.sha256(original_content.encode("utf-8")).hexdigest(),
            "new_content_sha256": hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest(),
            "removed_character_count": len(original_content) - len(cleaned_content),
        })
    log_columns = [
        "news_id", "rule_id", "old_content_sha256", "new_content_sha256", "removed_character_count",
    ]
    return corrected, pd.DataFrame(changes, columns=log_columns)


def residual_boilerplate_issues(value: object, source: object = None) -> list[str]:
    """Report known unresolved page shells; this is not a proof of all cleanliness."""
    text = _article_text(value)
    publisher = _publisher_name(source)
    issues: list[str] = []
    if any(text.find(marker) >= (200 if publisher == "TheLeader" else 0) for marker in SOURCE_FOOTER_MARKERS.get(publisher, ())):
        issues.append("RESIDUAL_SOURCE_WIDGET")
    if publisher == "NguoiQuanSat" and any(marker in text for marker in PDF_APPENDIX_MARKERS):
        issues.append("RESIDUAL_PDF_APPENDIX")
    if re.search(
        r"Market Watch list|Copy link Link bài gốc|Xem hồ sơ doanh nghiệp|"
        r"home Tài chính Chứng khoán|"
        r"Created with Highstock|TIN MỚI|Đăng nhập Đăng ký Email Mật khẩu|"
        r"Nhận OTP Bạn từng có tài khoản social|POWERED BY ONECMS|"
        r"This site can[’']t be reached|error_replace_script|"
        r"Chọn Báo Thanh Niên làm nguồn ưu tiên|M\.AIgợi ý câu hỏi liên quan|"
        r"Bấm vào mỗi từ khóa để xem bài cùng chủ đề|"
        r"Xem theo ngàyNgày123456789|Twitter Widget Iframe",
        text,
    ):
        issues.append("RESIDUAL_PAGE_SHELL")
    if publisher in {"SucKhoeDoiSong", "VnExpress"} and re.search(
        r"Bỏ qua đến nội dung chính|50 năm phụng sự khát vọng Việt Hoạt động Hành trình|"
        r"Trả lờiBáo vi phạm|This page has been blocked by an extension",
        text,
    ):
        issues.append("RESIDUAL_PAGE_SHELL")
    return issues


def assess_content_quality(
    content: object,
    source: object = None,
    title: object = None,
    summary: object = None,
    url: object = None,
) -> list[str]:
    """Return auditable issue codes; candidates are review flags, not proven errors.

    ``UNRESOLVED_TRAILING_FRAGMENT`` conservatively quarantines a headline-like
    API tail without deleting it. An unfinished legitimate sentence can also
    match, so this code must never be described as confirmed boilerplate.
    """
    text = _article_text(content)
    publisher = _publisher_name(source)
    title_text = _article_text(title)
    summary_text = _article_text(summary)
    issues = residual_boilerplate_issues(text, source)
    if isinstance(source, str) and source.startswith("API:CafeF"):
        boundaries = list(re.finditer(r"[.!?。][\"”’']?\s+", text))
        if boundaries:
            last_boundary = boundaries[-1]
            fragment = text[last_boundary.end():].strip()
            if (
                last_boundary.end() >= 200 and 35 <= len(fragment) <= 450
                and len(fragment.split()) >= 7 and fragment[0].isupper()
                and not re.search(r"[.!?。][\"”’']?$", fragment)
            ):
                issues.append("UNRESOLVED_TRAILING_FRAGMENT")
    normalized = lambda value: re.sub(r"[^\w]+", "", value.casefold())
    normalized_content = normalized(text)
    if not normalized_content:
        issues.append("EMPTY_BODY")
    elif normalized_content == normalized(title_text):
        issues.append("TITLE_ONLY")
    elif summary_text and normalized_content in {
        normalized(summary_text), normalized(title_text + summary_text),
    }:
        issues.append("SUMMARY_ONLY")

    if publisher == "TinNhanhChungKhoan" and (
        text.count("Ngày GDKHQ") >= 3
        and ("Báo Đầu tư Chứng khoán số" in text or "Vietnam Investment Review số" in text)
    ):
        issues.append("RELATED_LINKS_WITHOUT_BODY")
    if publisher == "ImexpharmOfficial" and (
        "Thông báo: Sự kiện công bố KQKD" in text
        or (title_text and normalized_content == normalized(title_text))
    ):
        issues.append("INFOGRAPHIC_WITHOUT_TEXT")
    if publisher == "DoiSongVN" and title_text == "Thế giới số" and "Tin Hot" in text:
        issues.append("CATEGORY_PAGE")
    if publisher == "24HMoney" and title_text.endswith(": Sự kiện liên quan") and (
        "Tổng quan Bài viết PTKT" in text and "Báo cáo tài chính" in text
    ):
        issues.append("CATEGORY_PAGE")
    if publisher == "VietnamNet" and text.startswith("Bản tin Podcast thời sự tổng hợp") and (
        "gồm các tin chính:" in text and len(text) < 500
    ):
        issues.append("PODCAST_WITHOUT_TRANSCRIPT")

    # These three crawler mismatches were verified against the supplied records.
    # Guard with actual body cues so a later corrected body can pass validation.
    url_text = "" if url is None or pd.isna(url) else str(url)
    mismatches = (
        ("the-gioi-di-dong-giam-doanh-thu-va-loi-nhuan-i5013777", "Giả mạo website Thế Giới Di Động"),
        ("nha-mang-wintel-cua-masan-am-tham-thay-tuong-2146102", "Chiều ngày 3/6/2020, Công ty Cổ phần Mobicast"),
        ("thong-tin-chi-tiet-ve-don-vi-cung-ung-gia-do-u-chat-cam-cho-bach-hoa-xanh-2357084", "vừa khai trương siêu thị ‘size 5 tỷ’"),
    )
    if any(url_fragment in url_text and cue.casefold() in text.casefold() for url_fragment, cue in mismatches):
        issues.append("MISMATCHED_BODY")
    return list(dict.fromkeys(issues))


def normalize_url(value: object) -> object:
    """Select a scalar URL and convert it to the project's canonical URL form."""
    scalar_value = select_scalar_value(value)
    if pd.isna(scalar_value):
        return pd.NA
    canonical_url = canonicalize_url(scalar_value)
    return canonical_url if canonical_url else pd.NA


def normalize_tickers(value: object, excluded: set[str] | None = None) -> str:
    """Return sorted unique ticker tokens after removing explicitly excluded tokens."""
    excluded_tokens = {token.upper() for token in (excluded or set())}
    tokens = {
        token for token in split_tickers(value)
        if token not in excluded_tokens
    }
    return ", ".join(sorted(tokens))


def resolve_published_date(published_date: object, published_at_raw: object) -> object:
    """Prefer the local date parsed from the source timestamp, otherwise use date-only data."""
    published_timestamp = parse_timestamp(published_at_raw)
    if not pd.isna(published_timestamp):
        return published_timestamp.strftime("%Y-%m-%d")

    fallback_date = pd.to_datetime(published_date, format="%Y-%m-%d", errors="coerce")
    return fallback_date.strftime("%Y-%m-%d") if not pd.isna(fallback_date) else pd.NA


def normalize_timestamp(value: object) -> object:
    """Convert a known timestamp representation to ISO 8601 in Vietnam time."""
    timestamp = parse_timestamp(value)
    return timestamp.isoformat() if not pd.isna(timestamp) else pd.NA


def sha256_text(value: object) -> object:
    """Hash cleaned text with SHA-256; return missing for missing text."""
    if value is None or pd.isna(value):
        return pd.NA
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _split_delimited(values: Iterable[object], delimiter: str) -> list[str]:
    tokens: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        tokens.update(token.strip() for token in str(value).split(delimiter) if token.strip())
    return sorted(tokens)


def _record_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic quality signals for selecting a duplicate representative."""
    quality = pd.DataFrame(index=data.index)
    quality["metadata_fields"] = data[["summary", "author", "category"]].notna().sum(axis=1)
    quality["has_source_timestamp"] = data["published_at"].notna().astype(int)
    quality["content_length"] = data["content"].fillna("").astype(str).str.len()
    return quality


def consolidate_url_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge rows with the same canonical URL while preserving ticker and lineage unions."""
    working = data.copy()
    quality = _record_quality(working)
    working = working.assign(
        _metadata_fields=quality["metadata_fields"],
        _has_source_timestamp=quality["has_source_timestamp"],
        _content_length=quality["content_length"],
    )

    kept_rows: list[pd.Series] = []
    merge_rows: list[dict[str, object]] = []
    for canonical_url, group in working.groupby("url", sort=False, dropna=False):
        ranked_group = group.sort_values(
            ["_metadata_fields", "_has_source_timestamp", "_content_length", "news_id"],
            ascending=[False, False, False, True],
        )
        representative = ranked_group.iloc[0].copy()

        if len(group) > 1:
            ticker_tokens: set[str] = set()
            for ticker_value in group["tickers"]:
                ticker_tokens.update(split_tickers(ticker_value))
            representative["tickers"] = ", ".join(sorted(ticker_tokens))
            representative["dataset_origin"] = "|".join(
                _split_delimited(group["dataset_origin"], "|")
            )
            representative["source_news_ids"] = "|".join(
                _split_delimited(group["source_news_ids"], "|")
            )
            representative["source_content_hashes"] = "|".join(
                _split_delimited(group["source_content_hashes"], "|")
            )
            merge_rows.append({
                "canonical_url": canonical_url,
                "kept_news_id": representative["news_id"],
                "merged_news_ids": "|".join(sorted(group["news_id"].astype(str))),
                "rows_merged": len(group),
            })
        kept_rows.append(representative)

    result = pd.DataFrame(kept_rows).drop(
        columns=["_metadata_fields", "_has_source_timestamp", "_content_length"]
    )
    result = result.reset_index(drop=True)
    merge_log = pd.DataFrame(
        merge_rows,
        columns=["canonical_url", "kept_news_id", "merged_news_ids", "rows_merged"],
    )
    return result, merge_log


def consolidate_article_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge components sharing a URL, exact content, or normalized title.

    Tickers and lineage fields are unioned, while the most complete row supplies the
    remaining values.
    """
    working = data.reset_index(drop=True).copy()
    row_count = len(working)
    parents = list(range(row_count))

    def find(position: int) -> int:
        while parents[position] != position:
            parents[position] = parents[parents[position]]
            position = parents[position]
        return position

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    normalized_titles = working["title"].fillna("").map(normalize_title)
    duplicate_keys = {
        "canonical_url": working["url"].fillna("").astype(str),
        "exact_content": working["content"].fillna("").astype(str),
        "normalized_title": normalized_titles,
    }

    for key_name, keys in duplicate_keys.items():
        first_position: dict[object, int] = {}
        for position, key in enumerate(keys):
            if key == "":
                continue
            if key in first_position:
                union(first_position[key], position)
            else:
                first_position[key] = position

    component_positions: dict[int, list[int]] = {}
    for position in range(row_count):
        component_positions.setdefault(find(position), []).append(position)

    quality = _record_quality(working)
    ranked = working.assign(
        _metadata_fields=quality["metadata_fields"],
        _has_source_timestamp=quality["has_source_timestamp"],
        _content_length=quality["content_length"],
    )
    kept_rows: list[pd.Series] = []
    merge_rows: list[dict[str, object]] = []

    for component_number, positions in enumerate(component_positions.values(), start=1):
        group = ranked.iloc[positions]
        representative = group.sort_values(
            ["_metadata_fields", "_has_source_timestamp", "_content_length", "news_id"],
            ascending=[False, False, False, True],
        ).iloc[0].copy()

        if len(group) > 1:
            ticker_tokens: set[str] = set()
            for ticker_value in group["tickers"]:
                ticker_tokens.update(split_tickers(ticker_value))
            representative["tickers"] = ", ".join(sorted(ticker_tokens))

            for column, delimiter in (
                ("dataset_origin", "|"),
                ("source_news_ids", "|"),
                ("source_content_hashes", "|"),
            ):
                if column in group.columns:
                    representative[column] = "|".join(_split_delimited(group[column], delimiter))

            reasons: list[str] = []
            if group["url"].duplicated(keep=False).any():
                reasons.append("canonical_url")
            if group["content"].duplicated(keep=False).any():
                reasons.append("exact_content")
            group_titles = normalized_titles.iloc[positions]
            if group_titles.duplicated(keep=False).any():
                reasons.append("normalized_title")

            merge_rows.append({
                "duplicate_group": component_number,
                "duplicate_reasons": "|".join(reasons),
                "kept_news_id": representative["news_id"],
                "merged_news_ids": "|".join(sorted(group["news_id"].astype(str))),
                "rows_in_group": len(group),
                "rows_removed": len(group) - 1,
            })
        kept_rows.append(representative)

    result = pd.DataFrame(kept_rows).drop(
        columns=["_metadata_fields", "_has_source_timestamp", "_content_length"]
    )
    result = result.reset_index(drop=True)
    merge_log = pd.DataFrame(merge_rows, columns=[
        "duplicate_group", "duplicate_reasons", "kept_news_id",
        "merged_news_ids", "rows_in_group", "rows_removed",
    ])
    return result, merge_log
