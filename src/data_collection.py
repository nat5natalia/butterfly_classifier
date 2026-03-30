from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from PIL import Image

from src.config import CFG
from src.utils import ensure_dir


def get_best_photo_url(photo: Dict[str, Any]) -> Optional[str]:
    """
    Возвращает лучший доступный URL фотографии.
    """
    for key in ["original_url", "large_url", "medium_url", "small_url", "url"]:
        url = photo.get(key)
        if url:
            if key == "url":
                url = url.replace("square", "large").replace("thumb", "large")
            return url
    return None


def download_image(session: requests.Session, url: str, timeout: int = 30) -> Optional[Image.Image]:
    """
    Скачивает изображение и проверяет, что его можно открыть через Pillow.
    """
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def collect_for_taxon(
    species_name: str,
    taxon_id: int,
    target_count: int = CFG.target_per_class,
    out_root: Path = CFG.raw_dir,
    base_url: str = CFG.iNat_base_url,
    quality_grade: str = CFG.quality_grade,
    per_page: int = CFG.per_page,
) -> pd.DataFrame:
    """
    Скачивает изображения для одного вида и возвращает таблицу метаданных.
    """
    out_dir = ensure_dir(out_root / species_name)
    session = requests.Session()

    rows: List[dict] = []
    downloaded = 0
    page = 1

    while downloaded < target_count:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": quality_grade,
            "has[]": "photos",
            "page": page,
            "per_page": per_page,
            "order_by": "observed_on",
            "order": "desc",
        }

        response = session.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        for obs in results:
            if downloaded >= target_count:
                break

            photos = obs.get("photos") or []
            if not photos:
                continue

            photo = photos[0]
            photo_url = get_best_photo_url(photo)
            if not photo_url:
                continue

            obs_id = obs.get("id")
            photo_id = photo.get("id")
            observed_on = obs.get("observed_on")
            license_code = photo.get("license_code") or obs.get("license_code")
            source = obs.get("uri") or "inaturalist"

            filename = f"{species_name}_{obs_id}_{photo_id}.jpg"
            filepath = out_dir / filename

            if filepath.exists():
                continue

            try:
                img = download_image(session, photo_url)
                width, height = img.size

                # сохраняем только валидные изображения
                img.save(filepath, format="JPEG", quality=95, optimize=True)

                rows.append(
                    {
                        "filepath": str(filepath.resolve()),
                        "label": species_name,
                        "taxon_id": taxon_id,
                        "observation_id": obs_id,
                        "photo_id": photo_id,
                        "photo_url": photo_url,
                        "source": source,
                        "license": license_code,
                        "width": width,
                        "height": height,
                        "observed_on": observed_on,
                    }
                )

                downloaded += 1

            except Exception:
                continue

        page += 1

    return pd.DataFrame(rows)


def collect_dataset(
    taxa: Dict[str, int],
    target_count: int = CFG.target_per_class,
    metadata_csv: Path = CFG.raw_metadata_csv,
) -> pd.DataFrame:
    """
    Скачивает данные по всем классам и сохраняет raw_metadata.csv.
    """
    CFG.ensure_dirs()

    all_parts: List[pd.DataFrame] = []

    for species_name, taxon_id in taxa.items():
        print(f"Collecting: {species_name} ({taxon_id})")
        part = collect_for_taxon(
            species_name=species_name,
            taxon_id=taxon_id,
            target_count=target_count,
        )
        all_parts.append(part)

    df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_csv, index=False)

    return df