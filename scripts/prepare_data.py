from __future__ import annotations

import pandas as pd

from src.config import CFG
from src.data_collection import collect_dataset
from src.data_cleaning import clean_metadata, save_cleaning_outputs
from src.split import stratified_group_split
from src.utils import save_json

TAXA = {
    "monarch": 48662,
    "admiral": 49133,
    "cabbage_white": 55626,
    "silvery_checkerspot": 49150,
    "painted_lady": 48548,
    "tiger_swallowtail": 60551,
    "peacock": 207977,
    "common_blue": 55641,
}


def create_class_map(df: pd.DataFrame) -> dict:
    """Create mapping from label string to integer index."""
    labels = sorted(df["label"].unique())
    class_map = {label: idx for idx, label in enumerate(labels)}
    save_json(class_map, CFG.class_map_json)
    return class_map


def main() -> None:
    CFG.ensure_dirs()

    # 1. Сбор данных
    df = collect_dataset(
        taxa=TAXA,
        target_count=CFG.target_per_class,
    )
    print("Done collecting data.")
    print(f"Total images downloaded: {len(df)}")
    print(f"Saved metadata to: {CFG.raw_metadata_csv}")

    # 2. Очистка
    cleaned_df, removed_df = clean_metadata(
        df,
        min_width=CFG.min_width,
        min_height=CFG.min_height,
        deduplicate=True,
    )
    save_cleaning_outputs(cleaned_df, removed_df)
    print(f"Cleaned: {len(cleaned_df)} images")
    print(f"Removed: {len(removed_df)} images")
    print(f"Saved cleaned metadata to: {CFG.cleaned_metadata_csv}")
    print(f"Saved cleaning report to: {CFG.cleaning_report_json}")

    # 3. Разделение на train/val/test
    if cleaned_df.empty:
        print("No cleaned data to split.")
        return

    train_df, val_df, test_df = stratified_group_split(cleaned_df)

    train_df.to_csv(CFG.train_csv, index=False)
    val_df.to_csv(CFG.val_csv, index=False)
    test_df.to_csv(CFG.test_csv, index=False)
    print(f"Saved train/val/test to:\n  {CFG.train_csv}\n  {CFG.val_csv}\n  {CFG.test_csv}")

    # 4. Сохранение словаря классов
    class_map = create_class_map(cleaned_df)
    print(f"Class map saved to {CFG.class_map_json}")

    # 5. Вывод статистики
    print("\nClass distribution in splits:")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df["label"].value_counts().sort_index()
        print(f"{name}:\n{counts.to_string()}")


if __name__ == "__main__":
    main()