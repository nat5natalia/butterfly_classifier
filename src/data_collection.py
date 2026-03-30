from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import pandas as pd
import requests
from PIL import Image

from src.config import CFG
from src.utils import ensure_dir


# ---------- helpers ----------

def get_best_photo_url(photo: Dict[str, Any]) -> Optional[str]:
    for key in ["original_url", "large_url", "medium_url", "small_url", "url"]:
        url = photo.get(key)
        if url:
            if key == "url":
                url = url.replace("square", "large").replace("thumb", "large")
            return url
    return None


def safe_request(session, url, params=None, retries=5, sleep=2):
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                print("Rate limit hit, sleeping...")
                time.sleep(5)
            else:
                print(f"HTTP {response.status_code}, retry...")
        except Exception as e:
            print(f"Request error: {e}")

        time.sleep(sleep * (attempt + 1))

    return None


def download_image(session, url, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
        except Exception:
            time.sleep(1 + attempt)

    return None


# ---------- main logic ----------

def collect_for_taxon(
    species_name: str,
    taxon_id: int,
    target_count: int,
) -> pd.DataFrame:

    out_dir = ensure_dir(CFG.raw_dir / species_name)
    session = requests.Session()

    rows = []
    downloaded = len(list(out_dir.glob("*.jpg")))
    page = 1

    print(f"{species_name}: already have {downloaded}")

    while downloaded < target_count:
        print(f"{species_name}: page {page}")

        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "has[]": "photos",
            "page": page,
            "per_page": 30,
        }

        response = safe_request(session, CFG.iNat_base_url, params)

        if response is None:
            print("Skipping page due to errors")
            page += 1
            continue

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
            url = get_best_photo_url(photo)
            if not url:
                continue

            obs_id = obs.get("id")
            photo_id = photo.get("id")

            filename = f"{species_name}_{obs_id}_{photo_id}.jpg"
            path = out_dir / filename

            if path.exists():
                continue

            img = download_image(session, url)

            if img is None:
                continue

            w, h = img.size

            # фильтр по размеру
            if w < CFG.min_width or h < CFG.min_height:
                continue

            try:
                img.save(path, "JPEG", quality=90)
            except Exception:
                continue

            rows.append({
                "filepath": str(path.resolve()),
                "label": species_name,
                "taxon_id": taxon_id,
                "observation_id": obs_id,
                "photo_id": photo_id,
                "width": w,
                "height": h,
            })

            downloaded += 1

            if downloaded % 10 == 0:
                print(f"{species_name}: {downloaded}/{target_count}")

            time.sleep(0.2)

        page += 1
        time.sleep(1)

    return pd.DataFrame(rows)


def collect_dataset(taxa: Dict[str, int], target_count: int) -> pd.DataFrame:
    CFG.ensure_dirs()

    all_data = []

    for name, taxon_id in taxa.items():
        df = collect_for_taxon(name, taxon_id, target_count)
        all_data.append(df)

        # пауза между классами
        time.sleep(3)

    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    final_df.to_csv(CFG.raw_metadata_csv, index=False)

    return final_df