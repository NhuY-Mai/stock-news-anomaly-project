import hashlib
from pathlib import Path

import pandas as pd
import pytest

import src.cleaning_utils as cleaning_helpers
from src.cleaning_utils import (
    REVIEWED_CONTENT_CORRECTIONS,
    apply_approved_content_corrections,
    assess_content_quality,
    clean_article_content,
    clean_reviewed_article_content,
    consolidate_article_duplicates,
    normalize_tickers,
    residual_boilerplate_issues,
    resolve_published_date,
    select_scalar_value,
)


def test_select_scalar_value_parses_real_list_only():
    assert select_scalar_value('["a", "b"]') == "a"
    assert select_scalar_value("[PDF] report") == "[PDF] report"


def test_normalize_tickers_removes_only_excluded_token():
    assert normalize_tickers("MSN, QNS, SAB", {"QNS"}) == "MSN, SAB"


def test_resolve_published_date_prefers_source_timestamp():
    assert (
        resolve_published_date("2020-01-04", "2020-04-01T09:32:51+07:00")
        == "2020-04-01"
    )


def _article(**overrides):
    article = {
        "news_id": "id-1",
        "url": "https://example.com/a",
        "title": "Kết quả kinh doanh",
        "summary": pd.NA,
        "author": pd.NA,
        "category": pd.NA,
        "content": "Nội dung thứ nhất",
        "published_date": "2025-01-10",
        "published_at": "2025-01-10T09:00:00+07:00",
        "tickers": "MSN",
        "dataset_origin": "crawler_a",
        "source_news_ids": "id-1",
        "source_content_hashes": "hash-1",
    }
    article.update(overrides)
    return article


def test_consolidate_article_duplicates_merges_normalized_title_across_dates():
    data = pd.DataFrame([
        _article(),
        _article(
            news_id="id-2",
            url="https://example.com/b",
            title="  kết quả kinh doanh ",
            summary="Bản đầy đủ",
            content="Nội dung dài hơn và đầy đủ hơn",
            tickers="VNM",
            source_news_ids="id-2",
            source_content_hashes="hash-2",
        ),
        _article(
            news_id="id-3",
            url="https://example.com/c",
            published_date="2025-01-11",
            content="Nội dung ngày tiếp theo",
            source_news_ids="id-3",
            source_content_hashes="hash-3",
        ),
    ])

    result, merge_log = consolidate_article_duplicates(data)

    assert len(result) == 1
    merged = result.loc[result["news_id"].eq("id-2")].iloc[0]
    assert merged["tickers"] == "MSN, VNM"
    assert merge_log.iloc[0]["duplicate_reasons"] == "normalized_title"


def test_consolidate_article_duplicates_merges_exact_content_across_dates():
    data = pd.DataFrame([
        _article(),
        _article(
            news_id="id-2",
            url="https://example.com/b",
            title="Tiêu đề khác",
            published_date="2025-01-11",
            source_news_ids="id-2",
            source_content_hashes="hash-2",
        ),
    ])

    result, merge_log = consolidate_article_duplicates(data)

    assert len(result) == 1
    assert merge_log.iloc[0]["duplicate_reasons"] == "exact_content"


def test_content_cleaning_decodes_entities_and_normalizes_unicode():
    content = "Doanh&nbsp;thu\u200b tăng 12% &amp; lợi nhuận đạt 300 tỷ đồng. &#160;"

    assert clean_article_content(content) == (
        "Doanh thu tăng 12% & lợi nhuận đạt 300 tỷ đồng."
    )


@pytest.mark.parametrize("value", [None, pd.NA, "", " \n\t "])
def test_content_cleaning_preserves_missing_values(value):
    assert pd.isna(clean_article_content(value))


@pytest.mark.parametrize("source", [None, "CafeF", "VnBusiness", "NguoiQuanSat"])
def test_financial_registration_and_management_quotes_are_not_boilerplate(source):
    content = (
        "Chủ tịch đăng ký mua 2 triệu cổ phiếu để nâng tỷ lệ sở hữu. "
        "Ban lãnh đạo chia sẻ kế hoạch kinh doanh, thông qua phương án "
        "phát hành cổ phiếu và đăng ký đầu tư dự án mới."
    )

    assert clean_article_content(content, source=source) == content
    assert residual_boilerplate_issues(content, source=source) == []


@pytest.mark.parametrize("related_heading", ["Tin liên quan", "Tin cùng chuyên mục"])
def test_bnews_removes_appended_related_news(related_heading):
    body = (
        "BNEWS Vinamilk ra mắt sản phẩm mới tại Philippines. "
        "Liên doanh có vốn đầu tư ban đầu 6 triệu USD./."
    )
    content = (
        f"{body} {related_heading} VN-Index khởi đầu tuần mới trong sắc đỏ. "
        "AMC Robotics đặt mục tiêu đưa dây chuyền tại Bắc Ninh vào vận hành."
    )

    result = clean_article_content(content, source="BNews")

    assert "Vinamilk ra mắt sản phẩm mới tại Philippines" in result
    assert "6 triệu USD" in result
    assert "VN-Index" not in result
    assert "AMC Robotics" not in result
    assert residual_boilerplate_issues(result, source="BNews") == []


def test_related_information_in_financial_prose_is_not_a_widget_heading():
    content = (
        "BNEWS Hội đồng quản trị xem xét các thông tin liên quan đến dự án mới. "
        "Doanh nghiệp dự kiến giải ngân 500 tỷ đồng trong năm tới."
    )

    assert clean_article_content(content, source="BNews") == content.removeprefix("BNEWS ")
    assert residual_boilerplate_issues(content, source="BNews") == []


@pytest.mark.parametrize(
    ("related_heading", "related_text"),
    [
        (
            "Bài viết liên quan",
            "MWG Tiếp Tục Vào Top 3 Doanh Nghiệp 25 tháng 9, 2026 · 146 lượt xem",
        ),
        (
            "Related articles",
            "MWG Continues To Rank Among The Top 3 25 September 2026 · 27 views",
        ),
    ],
)
def test_mwg_removes_vietnamese_and_english_related_articles(
    related_heading, related_text
):
    body = "TopZone ghi nhận doanh thu 15 nghìn tỷ đồng và dự kiến mở thêm cửa hàng."

    result = clean_article_content(
        f"{body} {related_heading} {related_text}", source="MWGOfficial"
    )

    assert result == body
    assert residual_boilerplate_issues(result, source="MWGOfficial") == []


def test_nqs_uses_article_anchor_to_remove_menu_and_footer():
    title = "Vinamilk lập kỷ lục doanh thu quý II/2025"
    body = "Doanh thu quý II đạt 16.000 tỷ đồng, tăng 8% so với cùng kỳ."
    content = (
        "Market Watch list Chứng khoán Chuyển động thị trường Doanh nghiệp A Z "
        f"Bất động sản Thị trường nhà đất {title} 30/07/2025 10:00 {body} "
        "Trang TTĐTTH của Công ty cổ phần Đầu tư đổi mới công nghệ INTECH "
        "Giấy phép số 2326/GP-TTĐT Liên hệ quảng cáo Email: sales@nguoiquansat.vn "
        "POWERED BY ONECMS & INTECH"
    )

    result = clean_article_content(content, source="NguoiQuanSat", title=title)

    assert body in result
    assert "Market Watch list" not in result
    assert "Liên hệ quảng cáo" not in result
    assert "ONECMS" not in result
    assert residual_boilerplate_issues(result, source="NguoiQuanSat") == []


@pytest.mark.parametrize("source", ["CafeF", "API:CafeF DN", "API:CafeF Vĩ mô"])
def test_cafef_removes_terminal_related_widget_without_losing_article(source):
    body = "MWG ghi nhận doanh thu quý I tăng 20%, lợi nhuận đạt 900 tỷ đồng."
    content = (
        f"{body} Theo Nhịp sống thị trường Tags: MWG kết quả kinh doanh "
        "Nổi bật Việt Nam nâng hạng thị trường chứng khoán "
        "11 giờ trước Công tyTin tứcLãnh đạo Trở lên trên"
    )

    result = clean_article_content(content, source=source)

    assert body in result
    assert "Công tyTin tứcLãnh đạo" not in result
    assert "Trở lên trên" not in result


def test_inline_latest_news_marker_does_not_discard_following_article_body():
    conclusion = (
        "Trên khắp các diễn đàn chứng khoán, cổ đông VNM lo ngại giá nguyên liệu. "
        "Doanh nghiệp dự kiến mở rộng thị trường xuất khẩu trong năm sau."
    )
    content = (
        "Cổ phiếu Vinamilk giảm về mức thấp nhất một năm. "
        'TIN MỚI Cổ phiếu PNJ "có biến" VinFast sắp nhận sáp nhập một công ty. '
        f"Một năm thua lỗ của cổ đông VNM {conclusion}"
    )

    result = clean_article_content(content, source="CafeF")

    assert conclusion in result


def test_vnbusiness_removes_complete_authentication_widget():
    body = "Nhà đầu tư đăng ký mua cổ phiếu để hưởng cổ tức bằng tiền mặt."
    content = (
        f"{body} Đăng nhập Đăng ký Email Mật khẩu Quên mật khẩu? "
        "Đăng nhập Email Mật khẩu Nhập lại mật khẩu Nhập mã xác thực Đăng ký "
        "Khi nhấn đăng ký, bạn đã đồng ý với điều khoản của báo VnBusiness. "
        "Khôi phục mật khẩu Nhập email bạn đã đăng ký Chúng tôi sẽ gửi mã OTP "
        "đến email của bạn. Gửi mã OTP Quay lại Xác thực mã OTP Nhập mã OTP "
        "Vui lòng nhập mã OTP đã được gửi đến email của bạn. Xác thực Quay lại "
        "Đặt lại mật khẩu Mật khẩu mới Nhập lại mật khẩu mới Đặt lại mật khẩu "
        "Quay lại Hoặc đăng nhập bằng tài khoản 1 / 1"
    )

    result = clean_article_content(content, source="VnBusiness")

    assert result == body
    assert residual_boilerplate_issues(result, source="VnBusiness") == []


def test_inline_javascript_is_removed_without_dropping_later_paragraph():
    before = "Vinamilk trình cổ đông phương án chia cổ tức 43,5% bằng tiền."
    after = "Năm 2025, Vinamilk đặt mục tiêu doanh thu 64.505 tỷ đồng."
    script = (
        'try{try{for(const iframe of document.querySelectorAll("iframe[data-src]"))'
        '{iframe.removeAttribute("sandbox");iframe.setAttribute("src",'
        'iframe.getAttribute("data-src"));}}catch(e){}}catch(e)'
        '{console.log("error_replace_script",e);}'
    )

    result = clean_article_content(f"{before} {script} {after}", source="VnExpress")

    assert before in result
    assert after in result
    assert "document.querySelectorAll" not in result
    assert "error_replace_script" not in result


def test_html_scripts_are_removed_but_numeric_comparisons_are_preserved():
    result = clean_article_content(
        '<p>Biên lợi nhuận &lt;5% trong quý I.</p>'
        '<script>console.log("advertisement")</script>'
        '<p>Lợi nhuận sau thuế đạt 20 tỷ đồng.</p>'
    )

    assert "Biên lợi nhuận <5% trong quý I." in result.replace("< ", "<")
    assert "Lợi nhuận sau thuế đạt 20 tỷ đồng." in result
    assert "advertisement" not in result


def test_residual_checker_detects_known_widgets_before_cleaning():
    contaminated = (
        "Doanh nghiệp công bố lợi nhuận. Related articles "
        "MWG Continues To Rank Among The Top 3 25 September 2026 · 27 views"
    )

    assert residual_boilerplate_issues(contaminated, source="MWGOfficial")
    cleaned = clean_article_content(contaminated, source="MWGOfficial")
    assert residual_boilerplate_issues(cleaned, source="MWGOfficial") == []


def test_quality_checker_flags_title_only_but_not_a_valid_short_news_brief():
    title = "Dược Hậu Giang khám bệnh, phát thuốc cho người dân Nghệ An"
    brief = (
        "Dược Hậu Giang vừa đồng hành khám bệnh, phát thuốc cho 1.000 người dân "
        "xã Quỳnh Long, huyện Quỳnh Lưu, tỉnh Nghệ An, nơi bà con thường mắc "
        "vấn đề về huyết áp do thói quen ăn mặn và đi biển dài ngày."
    )

    assert assess_content_quality(title, source="VietnamNet", title=title)
    assert assess_content_quality(brief, source="VietnamNet", title=title) == []


def test_quality_checker_flags_official_page_with_only_title_and_related_notices():
    title = "1Q2025 - Imexpharm Con số & Sự kiện"
    content = (
        f"{title} Thông báo: Sự kiện công bố KQKD 6 tháng đầu năm 2026 "
        "Thông báo: Sự kiện công bố KQKD Quý 1 năm 2026 "
        "IMP - Thông cáo báo chí: ĐHĐCĐ thường niên năm 2026 "
        "1Q2026-Imexpharm Con số và Sự kiện"
    )
    cleaned = clean_article_content(content, source="ImexpharmOfficial", title=title)

    assert assess_content_quality(cleaned, source="ImexpharmOfficial", title=title)


def test_quality_checker_flags_verified_mismatched_body_but_allows_correction():
    title = "Thế Giới Di Động giảm doanh thu và lợi nhuận"
    url = (
        "https://vietnamnet.vn/"
        "the-gioi-di-dong-giam-doanh-thu-va-loi-nhuan-i5013777.html"
    )
    mismatched = (
        "Giả mạo website Thế Giới Di Động, MoMo để lừa đảo người dùng. "
        "NCSC vừa khuyến nghị người dùng không truy cập các trang giả mạo."
    )
    corrected = (
        "Khách hàng thắt chặt chi tiêu khiến doanh thu của Thế Giới Di Động "
        "giảm 10%, lợi nhuận giảm 20% so với cùng kỳ năm trước."
    )

    assert "MISMATCHED_BODY" in assess_content_quality(
        mismatched, source="SEARCH:VietnamNet", title=title, url=url
    )
    assert assess_content_quality(
        corrected, source="SEARCH:VietnamNet", title=title, url=url
    ) == []


@pytest.mark.parametrize(
    ("source", "content"),
    [
        ("BNews", "BNEWS Lợi nhuận tăng 15%./. Tin cùng chuyên mục AMC Robotics mở nhà máy."),
        (
            "MWGOfficial",
            "Doanh thu đạt 20 tỷ đồng. Bài viết liên quan "
            "Tin mới 25 tháng 9, 2026 · 146 lượt xem",
        ),
        (
            "MWGOfficial",
            "Revenue rose 15%. Related articles "
            "Annual report 25 September 2026 · 27 views",
        ),
        (None, "Doanh&nbsp;thu tăng 10% &amp; lợi nhuận tăng 15%."),
    ],
)
def test_article_cleaning_is_idempotent(source, content):
    cleaned = clean_article_content(content, source=source)

    assert clean_article_content(cleaned, source=source) == cleaned


def test_unresolved_api_tail_is_review_candidate_without_deletion():
    body = "Doanh nghiệp ghi nhận doanh thu tăng 15% và lợi nhuận tăng 20% trong quý. " * 5
    fragment = "Một doanh nghiệp khác bất ngờ công bố kế hoạch đầu tư mới"
    content = body + fragment

    assert "UNRESOLVED_TRAILING_FRAGMENT" in assess_content_quality(
        content, source="API:CafeF DN"
    )
    assert fragment in clean_article_content(content, source="API:CafeF DN")
    assert "UNRESOLVED_TRAILING_FRAGMENT" not in assess_content_quality(
        body + "Doanh nghiệp tiếp tục duy trì kết quả kinh doanh khả quan.",
        source="API:CafeF DN",
    )


def test_verified_cafef_recommendation_is_removed_only_at_tail():
    body = "Doanh nghiệp ghi nhận doanh thu tăng 15% và lợi nhuận tăng 20% trong quý. " * 5
    headline = (
        "Profile các nhà tư vấn tái cấu trúc Novaland: 1 công ty tư vấn toàn cầu, "
        "1 hãng luật góp mặt tại loạt thương vụ M&A tỷ đô của Masan, VPBank"
    )

    assert clean_article_content(body + headline, source="API:CafeF DN") == body.strip()
    assert headline in clean_article_content(
        headline + " là tiêu đề được nhắc lại trong bài. " + body,
        source="API:CafeF DN",
    )


@pytest.mark.parametrize(
    "appendix",
    [
        "CAM KẾT PHÂN TÍCH Chuyên viên phân tích cam kết về quan điểm cá nhân.",
        "Xác nhận của chuyên viên phân tích Chúng tôi xác nhận quan điểm cá nhân.",
        "Update Report Hệ thống khuyến nghị của Vietcap Chúng tôi xác nhận quan điểm. Update Report Liên hệ CTCP Chứng khoán Vietcap.",
    ],
)
def test_pdf_appendix_removal_preserves_recommendation_and_financial_analysis(appendix):
    body = "MUA Giá mục tiêu 71.600 VND. " + "Doanh thu và lợi nhuận dự kiến tăng 20%. " * 8
    cleaned = clean_article_content(
        body + appendix, source="NguoiQuanSat", title="[PDF] CTCP Vĩnh Hoàn"
    )

    assert cleaned == body.strip()
    assert residual_boilerplate_issues(cleaned, source="NguoiQuanSat") == []
    assert body.strip() in clean_article_content(
        body + appendix, source="NguoiQuanSat", title="Doanh nghiệp công bố thông tin"
    )


def test_skds_family_site_shell_removed_without_changing_article():
    body = "Masan Consumer ghi nhận doanh thu tăng 20% và chuẩn bị niêm yết trên HoSE."
    content = (
        "Bỏ qua đến nội dung chính Chia sẻ twitter In Tiêu đề Thứ năm, 13:00 13/11/2025 "
        "Sản phẩm Dịch vụ Theo dõi Gia đình và Xã hội trên\\\\ \\\\ News " + body
        + " Masan Chia sẻ facebook twitter Bình luận (0)"
    )

    cleaned = clean_article_content(content, source="SucKhoeDoiSong")

    assert cleaned == body + " Masan"
    assert residual_boilerplate_issues(cleaned, source="SucKhoeDoiSong") == []


def test_vnexpress_anniversary_menu_and_reader_comments_removed():
    body = "CEO Mai Kiều Liên nói doanh nghiệp ưu tiên ổn định giá bán để hỗ trợ sức mua."
    content = (
        "50 năm phụng sự \\\\ \\\\ khát vọng Việt Hoạt động Hành trình 50 năm Lời chúc "
        "Thứ tư, 22/4/2026, 13:54 (GMT+7) " + body
        + " Thi Hà Vinamilk ghi nhận doanh thu cao kỷ lục Năm 2025, doanh thu bán hàng và cung cấp dịch vụ đạt kỷ lục. "
        "Ý kiến (11) Bình luận độc giả Trả lờiBáo vi phạm This page has been blocked by an extension"
    )

    cleaned = clean_article_content(content, source="VnExpress")

    assert cleaned == body + " Thi Hà"
    assert residual_boilerplate_issues(cleaned, source="VnExpress") == []


def test_cafef_submission_promotion_removed_not_normal_ranking_discussion():
    body = "Long Châu phát triển hệ sinh thái y tế số, giúp người dân tiếp cận dịch vụ chăm sóc sức khỏe."
    promotion = (
        "PRIVATE 100 và VNTAX 200 của năm 2026 (dựa trên số liệu thực nộp năm 2025): "
        "Bảng VNTAX 200 Các bài viết liên quan: https://cafef.vn/cafef-lists.html "
        "Doanh nghiệp gửi thông tin vào email info@cafef.vn."
    )

    assert clean_article_content(body + " " + promotion, source="CafeF") == body
    discussion = "Doanh nghiệp được xếp hạng PRIVATE 100 nhờ đóng góp ngân sách lớn."
    assert clean_article_content(discussion, source="CafeF") == discussion


@pytest.fixture(scope="module")
def reviewed_content_articles():
    """Use immutable raw input, not a generated validation report or edited clean CSV."""
    raw_path = Path(__file__).resolve().parents[1] / "data/raw/news_raw.csv"
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    articles = raw.loc[raw["news_id"].isin(REVIEWED_CONTENT_CORRECTIONS)].copy()
    assert len(articles) == 3
    articles["content"] = articles.apply(
        lambda row: clean_article_content(
            row["content"], row["source"], row["title"], row["summary"],
        ), axis=1,
    )
    return articles.set_index("news_id", drop=False)


@pytest.mark.parametrize("news_id", tuple(REVIEWED_CONTENT_CORRECTIONS))
def test_reviewed_content_cleanup_matches_approved_hash_and_is_idempotent(
    reviewed_content_articles, news_id,
):
    content = reviewed_content_articles.at[news_id, "content"]
    correction = REVIEWED_CONTENT_CORRECTIONS[news_id]
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == correction["old_sha256"]

    cleaned = clean_reviewed_article_content(news_id, content)

    assert hashlib.sha256(cleaned.encode("utf-8")).hexdigest() == correction["new_sha256"]
    assert clean_reviewed_article_content(news_id, cleaned) == cleaned
    assert len(cleaned) < len(content)


def test_reviewed_prefix_cleanup_preserves_listing_article_without_decoding_junk(reviewed_content_articles):
    news_id = "0924ce353ae49d611b69c2d14935624b6023ff60543e4208a02bdb20d6d65d62"
    content = reviewed_content_articles.at[news_id, "content"]

    cleaned = clean_reviewed_article_content(news_id, content)

    assert cleaned.startswith("Ngày 4/12/2025, Công ty Cổ phần Hàng tiêu dùng Masan")
    assert content.endswith(cleaned)
    assert "%20" not in cleaned
    assert '"Liên hệ")' not in cleaned
    assert "220.000 đồng/cổ phiếu" in cleaned
    assert "Go Global" in cleaned
    assert cleaned.endswith("Đăng Quang")


def test_reviewed_advertising_cleanup_preserves_products_and_operations(reviewed_content_articles):
    news_id = "8a622ba716429d810d76c613fdb3b8d163c0018af47193379b2326c4287854d9"
    content = reviewed_content_articles.at[news_id, "content"]

    cleaned = clean_reviewed_article_content(news_id, content)

    assert content.startswith(cleaned)
    assert "GPQC:" not in cleaned
    assert "Đọc kỹ hướng dẫn sử dụng trước khi dùng." not in cleaned
    assert "Doãn Phong" not in cleaned
    assert "NattoEnzym" in cleaned
    assert "Naturenz" in cleaned
    assert "Bocalex Multi" in cleaned
    assert "(ERP)" in cleaned
    assert "Japan-GMP" in cleaned
    assert cleaned.endswith("sẵn sàng lên sàn thương mại điện tử.")


def test_reviewed_related_link_cleanup_preserves_buyback_and_capex_facts(reviewed_content_articles):
    news_id = "62029430c4b97773069ee32d35b42c142f819d89aed26c9dfcc55ebd17da95ab"
    content = reviewed_content_articles.at[news_id, "content"]

    cleaned = clean_reviewed_article_content(news_id, content)

    assert content.startswith(cleaned)
    assert "* Lợi nhuận ròng quý 2" not in cleaned
    assert "FILI - 14:43 07/08/2026" not in cleaned
    assert "Về khả năng mua lại cổ phiếu quỹ" in cleaned
    assert "doanh nghiệp chưa có kế hoạch thực hiện" in cleaned
    assert "2,500-3,500 tỷ đồng" in cleaned
    assert cleaned.endswith("tùy tiến độ giải ngân.")


@pytest.mark.parametrize("news_id", tuple(REVIEWED_CONTENT_CORRECTIONS))
def test_reviewed_fragments_are_untouched_in_unapproved_articles(reviewed_content_articles, news_id):
    content = reviewed_content_articles.at[news_id, "content"]
    assert clean_reviewed_article_content("another-article", content) == content


@pytest.mark.parametrize("content", [None, pd.NA, "", "A different article body."])
def test_reviewed_content_requires_exact_approved_version(content):
    news_id = next(iter(REVIEWED_CONTENT_CORRECTIONS))
    with pytest.raises(ValueError, match="text body|unapproved content version"):
        clean_reviewed_article_content(news_id, content)


def test_reviewed_content_result_hash_protects_against_wrong_boundary(
    reviewed_content_articles, monkeypatch,
):
    news_id = next(iter(REVIEWED_CONTENT_CORRECTIONS))
    wrong_correction = {**REVIEWED_CONTENT_CORRECTIONS[news_id], "boundary": "Chủ tịch HĐQT"}
    monkeypatch.setattr(cleaning_helpers, "REVIEWED_CONTENT_CORRECTIONS", {news_id: wrong_correction})
    with pytest.raises(ValueError, match="approved hash"):
        clean_reviewed_article_content(news_id, reviewed_content_articles.at[news_id, "content"])


def test_reviewed_content_dataframe_updates_only_content_and_returns_auditable_log(reviewed_content_articles):
    data = reviewed_content_articles[["news_id", "title", "content", "tickers"]].reset_index(drop=True)
    data["news_type"] = ["governance_capital", "business_operations", "business_operations"]
    data["annotation_status"] = "labeled"
    unrelated = data.iloc[[0]].copy()
    unrelated["news_id"] = "unapproved-copy"
    data = pd.concat([unrelated, data], ignore_index=True)
    original = data.copy(deep=True)

    corrected, log = apply_approved_content_corrections(data)

    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(corrected.drop(columns="content"), original.drop(columns="content"))
    assert corrected.at[0, "content"] == original.at[0, "content"]
    assert corrected.columns.tolist() == original.columns.tolist()
    assert corrected.index.equals(original.index)
    assert set(log["news_id"]) == set(REVIEWED_CONTENT_CORRECTIONS)
    assert set(log["rule_id"]) == {"validation_content_review_v2"}
    assert sorted(log["removed_character_count"].tolist()) == [127, 148, 384]
    for row in log.itertuples(index=False):
        assert row.old_content_sha256 == REVIEWED_CONTENT_CORRECTIONS[row.news_id]["old_sha256"]
        assert row.new_content_sha256 == REVIEWED_CONTENT_CORRECTIONS[row.news_id]["new_sha256"]

    repeated, repeated_log = apply_approved_content_corrections(corrected)
    pd.testing.assert_frame_equal(repeated, corrected)
    assert repeated_log.empty
    assert repeated_log.columns.tolist() == log.columns.tolist()


def test_reviewed_content_dataframe_leaves_unrelated_and_empty_data_unchanged():
    for data in [pd.DataFrame(columns=["news_id", "content"]), pd.DataFrame([{"news_id": "other", "content": pd.NA}])]:
        corrected, log = apply_approved_content_corrections(data)
        pd.testing.assert_frame_equal(corrected, data)
        assert log.empty


@pytest.mark.parametrize("missing_column", ["news_id", "content"])
def test_reviewed_content_dataframe_requires_identity_and_body(missing_column):
    data = pd.DataFrame([{"news_id": "other", "content": "Body"}]).drop(columns=missing_column)
    with pytest.raises(ValueError, match="news_id and content"):
        apply_approved_content_corrections(data)


def test_reviewed_content_dataframe_rejects_ambiguous_rows_without_mutating_input(reviewed_content_articles):
    data = reviewed_content_articles[["news_id", "content"]].reset_index(drop=True)
    original = data.copy(deep=True)
    with pytest.raises(ValueError, match="at most once"):
        apply_approved_content_corrections(pd.concat([data, data.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="unique row index"):
        apply_approved_content_corrections(pd.concat([data, data.iloc[[0]]]))

    data.loc[1, "content"] = "Unreviewed replacement body."
    changed_input = data.copy(deep=True)
    with pytest.raises(ValueError, match="unapproved content version"):
        apply_approved_content_corrections(data)
    pd.testing.assert_frame_equal(data, changed_input)
    assert original.loc[0, "content"] == data.loc[0, "content"]
