from __future__ import annotations

from src.config import CFG
from src.data_collection import collect_dataset
from src.data_cleaning import clean_metadata, save_cleaning_output

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


def main() -> None:
    CFG.ensure_dirs()

    df = collect_dataset(
        taxa=TAXA,
        target_count=CFG.target_per_class,
    )

    print("Done collecting data.")
    print(f"Total images downloaded: {len(df)}")
    print(f"Saved metadata to: {CFG.raw_metadata_csv}")
    cleaned_df, removed_df = clean_metadata(
        raw_df,
        min_width=CFG.min_width,
        min_height=CFG.min_height,
        deduplicate=True,
    )
    save_cleaning_outputs(cleaned_df, removed_df)

    print(f"Cleaned: {len(cleaned_df)} images")
    print(f"Removed: {len(removed_df)} images")
    print(f"Saved cleaned metadata to: {CFG.cleaned_metadata_csv}")
    print(f"Saved cleaning report to: {CFG.cleaning_report_json}")


if __name__ == "__main__":
    main()
