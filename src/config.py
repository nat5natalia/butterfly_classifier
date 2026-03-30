from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    metadata_dir: Path = Path("data/metadata")
    cleaned_dir: Path = Path("data/cleaned")
    splits_dir: Path = Path("data/splits")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    metrics_dir: Path = Path("reports/metrics")
    app_dir: Path = Path("app")
    src_dir: Path = Path("src")
    scripts_dir: Path = Path("scripts")

    raw_metadata_csv: Path = Path("data/metadata/raw_metadata.csv")
    cleaned_metadata_csv: Path = Path("data/cleaned/cleaned_metadata.csv")
    cleaning_report_json: Path = Path("data/cleaned/cleaning_report.json")
    class_map_json: Path = Path("data/splits/class_indices.json")

    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    group_col: str = "observation_id"

    min_width: int = 400
    min_height: int = 300

    target_per_class: int = 150
    sleep_between_requests_sec: float = 0.25
    sleep_between_pages_sec: float = 0.8

    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    iNat_base_url: str = "https://api.inaturalist.org/v1/observations"
    quality_grade: str = "research"
    per_page: int = 30

    def ensure_dirs(self) -> None:
        for p in [
            self.data_dir,
            self.raw_dir,
            self.metadata_dir,
            self.cleaned_dir,
            self.splits_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.metrics_dir,
            self.app_dir,
            self.src_dir,
            self.scripts_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


CFG = ProjectConfig()


def save_config(path: Path = Path("data/config.json")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in asdict(CFG).items():
        payload[k] = str(v) if isinstance(v, Path) else v
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)