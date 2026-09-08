from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    timezone: str = "America/Chicago"
    current_season: int | None = None
    cfbd_api_key: str | None = None

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def team_logos_dir(self) -> Path:
        return self.project_root / "frontend" / "team_logos"

    @property
    def frontend_dir(self) -> Path:
        return self.project_root / "frontend"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def season(self) -> int:
        if self.current_season:
            return self.current_season
        now = datetime.now(ZoneInfo(self.timezone))
        return now.year

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.predictions_dir,
            self.team_logos_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or PROJECT_ROOT).resolve()
    load_dotenv(root / ".env")
    raw_season = os.getenv("CFB_CURRENT_SEASON", "").strip()
    settings = Settings(
        project_root=root,
        timezone=os.getenv("CFB_TIMEZONE", "America/Chicago"),
        current_season=int(raw_season) if raw_season else None,
        cfbd_api_key=os.getenv("CFBD_API_KEY", "").strip() or None,
    )
    settings.ensure_directories()
    return settings
