"""Script to build train_v2.csv by adding 26 verified clean review38 articles to baseline train.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Determine root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.model_data_utils import MODEL_SPLIT_COLUMNS, build_phobert_input_text


def main() -> None:
    train_path = PROJECT_DIR / "data/splits/news_type/train.csv"
    resolution_path = PROJECT_DIR / "data/annotation/news_type/review38/review38_resolution.csv"
    clean_path = PROJECT_DIR / "data/processed/news_clean.csv"
    mapping_path = PROJECT_DIR / "models/news_type_phobert_cnn/label_mapping.json"
    duplicates_path = PROJECT_DIR / "data/interim/duplicate_merge_log.csv"
    output_v2_path = PROJECT_DIR / "data/splits/news_type/train_v2.csv"

    print("1. Đọc dữ liệu đầu vào...")
    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    resolution_df = pd.read_csv(resolution_path, dtype=str, keep_default_na=False)
    clean_df = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    label_mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    label2id = label_mapping["label2id"]
    duplicate_merge_log = pd.read_csv(duplicates_path, dtype=str, keep_default_na=False)

    lineage: dict[str, str] = {}
    for group in duplicate_merge_log.itertuples(index=False):
        for alias in str(group.merged_news_ids).split("|"):
            if alias:
                lineage[alias] = group.kept_news_id

    # Filter 26 articles with v2_action == 'TRAIN_V2'
    train_v2_disposition = resolution_df[resolution_df["v2_action"] == "TRAIN_V2"].copy()
    print(f"Số bài từ review38 đạt điều kiện đưa vào train: {len(train_v2_disposition)}")
    assert len(train_v2_disposition) == 26, f"Kỳ vọng 26 bài nhưng tìm thấy {len(train_v2_disposition)}"

    clean_map = clean_df.set_index("news_id")

    new_rows = []
    for _, row in train_v2_disposition.iterrows():
        news_id = row["news_id"]
        final_type = row["final_news_type"]
        assert news_id in clean_map.index, f"news_id {news_id} không có trong news_clean.csv"
        clean_row = clean_map.loc[news_id]

        title = clean_row["title"]
        summary = clean_row["summary"]
        content = clean_row["content"]
        input_text = build_phobert_input_text(title, summary, content)

        canonical_id = lineage.get(news_id, news_id)
        label_id = str(label2id[final_type])

        new_rows.append({
            "news_id": news_id,
            "canonical_news_id": canonical_id,
            "source": clean_row["source"],
            "url": clean_row["url"],
            "published_date": clean_row["published_date"],
            "tickers": clean_row["tickers"],
            "title": title,
            "summary": summary,
            "content": content,
            "input_text": input_text,
            "news_type": final_type,
            "label_id": label_id,
            "split": "train",
        })

    new_train_df = pd.DataFrame(new_rows, columns=train_df.columns)
    
    # Check for duplicate news_id between baseline train and new articles
    overlap = set(train_df["news_id"]).intersection(set(new_train_df["news_id"]))
    assert not overlap, f"Phát hiện trùng lặp news_id với train cũ: {overlap}"

    # Combine baseline + new
    combined_train_df = pd.concat([train_df, new_train_df], ignore_index=True)
    print(f"Số bài train ban đầu (v1): {len(train_df)}")
    print(f"Số bài thêm mới (TRAIN_V2): {len(new_train_df)}")
    print(f"Tổng số bài train_v2: {len(combined_train_df)}")

    # Export train_v2.csv
    combined_train_df.to_csv(output_v2_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu thành công: {output_v2_path}")

    # Summary by class
    print("\nPhân bổ lớp trong train_v2 so với train v1:")
    dist_v1 = train_df["news_type"].value_counts()
    dist_new = new_train_df["news_type"].value_counts()
    dist_v2 = combined_train_df["news_type"].value_counts()
    summary_df = pd.DataFrame({
        "Train v1": dist_v1,
        "Thêm mới": dist_new.fillna(0).astype(int),
        "Train v2": dist_v2,
    }).fillna(0).astype(int)
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
