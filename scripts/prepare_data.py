from __future__ import annotations

from src.config import CFG
from src.data_collection import collect_dataset

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


if __name__ == "__main__":
    main()
