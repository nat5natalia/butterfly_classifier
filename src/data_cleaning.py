from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import CFG


def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Простой perceptual hash.
    """
    img = img.convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(img, dtype=np.float32)
    mean = pixels.mean()
    bits = pixels > mean
    return "".join("1" if b else "0" for b in bits.flatten())


def hamming_distance(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def validate_image(filepath: str, min_width: int, min_height: int) -> Tuple[bool, Dict]:
    """
    Проверяет:
      - существует ли файл
      - открывается ли он
      - достаточно ли он большой
    """
    path = Path(filepath)

    if not path.exists():
        return False, {"reason": "missing_file"}

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size

            if width < min_width or height < min_height:
                return False, {
                    "reason": "too_small",
                    "width": width,
                    "height": height,
                }

            return True, {"width": width, "height": height}

    except (UnidentifiedImageError, OSError):
        return False, {"reason": "unreadable"}
    except Exception:
        return False, {"reason": "error"}


def clean_metadata(
    df: pd.DataFrame,
    min_width: int = CFG.min_width,
    min_height: int = CFG.min_height,
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Очищает raw metadata:
      - удаляет отсутствующие файлы
      - удаляет битые файлы
      - удаляет слишком маленькие изображения
      - удаляет дубликаты по perceptual hash

    Возвращает:
      cleaned_df, removed_df
    """
    required_cols = {"filepath", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy().reset_index(drop=True)
    df["filepath"] = df["filepath"].astype(str)
    df["label"] = df["label"].astype(str)

    cleaned_rows: List[dict] = []
    removed_rows: List[dict] = []
    seen_hashes: Dict[str, str] = {}

    for _, row in df.iterrows():
        filepath = row["filepath"]

        ok, info = validate_image(
            filepath=filepath,
            min_width=min_width,
            min_height=min_height,
        )

        if not ok:
            removed_rows.append({**row.to_dict(), **info})
            continue

        if deduplicate:
            try:
                with Image.open(filepath) as img:
                    phash = average_hash(img)

                duplicate_of = None
                for existing_hash, existing_file in seen_hashes.items():
                    if hamming_distance(phash, existing_hash) <= 3:
                        duplicate_of = existing_file
                        break

                if duplicate_of is not None:
                    removed_rows.append(
                        {
                            **row.to_dict(),
                            "reason": "duplicate",
                            "duplicate_of": duplicate_of,
                        }
                    )
                    continue

                seen_hashes[phash] = filepath

            except Exception:
                removed_rows.append({**row.to_dict(), "reason": "hash_error"})
                continue

        new_row = row.to_dict()
        new_row.update(info)
        cleaned_rows.append(new_row)

    cleaned_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)
    removed_df = pd.DataFrame(removed_rows).reset_index(drop=True)

    return cleaned_df, removed_df


def save_cleaning_outputs(
    cleaned_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    cleaned_csv: Path = CFG.cleaned_metadata_csv,
    report_json: Path = CFG.cleaning_report_json,
) -> None:
    """
    Сохраняет:
      - cleaned_metadata.csv
      - cleaning_report.json
    """
    cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_csv, index=False)

    report = {
        "n_cleaned": int(len(cleaned_df)),
        "n_removed": int(len(removed_df)),
        "class_counts": (
            cleaned_df["label"].value_counts().sort_index().to_dict()
            if not cleaned_df.empty
            else {}
        ),
        "removed_reasons": (
            removed_df["reason"].value_counts().to_dict()
            if not removed_df.empty and "reason" in removed_df.columns
            else {}
        ),
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)