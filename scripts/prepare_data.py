from __future__ import annotations
import pandas as pd
from src.config import CFG
from src.data_collection import collect_dataset
from src.data_cleaning import clean_metadata, save_cleaning_outputs
from src.split import stratified_group_split
from src.utils import save_json, sha256sum

TAXA = {"monarch":48662,"admiral":49133,"cabbage_white":55626,"silvery_checkerspot":49150,
        "painted_lady":48548,"tiger_swallowtail":60551,"peacock":207977,"common_blue":55641}

def create_class_map(df: pd.DataFrame) -> dict:
    labels = sorted(df["label"].unique())
    m = {lbl:i for i,lbl in enumerate(labels)}
    save_json(m, CFG.class_map_json)
    return m

def main() -> None:
    CFG.ensure_dirs()

    if CFG.raw_metadata_csv.exists():
        df = pd.read_csv(CFG.raw_metadata_csv)
        print(f"Loaded existing raw metadata: {CFG.raw_metadata_csv} ({len(df)} rows)")
    else:
        df = collect_dataset(TAXA, target_count=CFG.target_per_class)
        print(f"Collected metadata rows: {len(df)}")

    cleaned_df, removed_df = clean_metadata(df, deduplicate=True)
    save_cleaning_outputs(cleaned_df, removed_df)
    if cleaned_df.empty:
        print("No cleaned data to split."); return

    train_df, val_df, test_df = stratified_group_split(cleaned_df)
    train_df.to_csv(CFG.train_csv, index=False)
    val_df.to_csv(CFG.val_csv, index=False)
    test_df.to_csv(CFG.test_csv, index=False)

    with (CFG.splits_dir / "splits_sha256.txt").open("w") as f:
        f.write(f"train.csv {sha256sum(CFG.train_csv)}\n")
        f.write(f"val.csv {sha256sum(CFG.val_csv)}\n")
        f.write(f"test.csv {sha256sum(CFG.test_csv)}\n")

    create_class_map(cleaned_df)
    print("Prepared data, splits, class_indices.json and split hashes.")

if __name__ == "__main__":
    main()
