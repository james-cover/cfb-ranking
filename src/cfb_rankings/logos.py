from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
MAX_LOGO_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
BRANDING_COLUMNS = [
    "team",
    "team_id",
    "logo_url",
    "logo_path",
    "team_color",
    "alt_color",
]


def _logo_urls(value: Any) -> list[str]:
    if isinstance(value, list):
        payload = value
    elif pd.isna(value) or value in (None, ""):
        return []
    else:
        try:
            payload = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return []
    return [str(url) for url in payload if str(url).startswith("https://")]


def _safe_stem(team_id: Any, team: str) -> str:
    if team_id is not None and not pd.isna(team_id):
        value = str(team_id).removesuffix(".0")
        if value.isdigit():
            return value
    slug = re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")
    return slug or "team"


def _clean_text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _download_logo(
    url: str,
    destination_stem: Path,
    session: requests.Session,
    timeout_seconds: int,
) -> Path | None:
    if urlparse(url).scheme != "https":
        return None
    try:
        response = session.get(url, timeout=timeout_seconds, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        suffix = ALLOWED_CONTENT_TYPES.get(content_type)
        if suffix is None:
            LOGGER.warning("Skipping unsupported logo type %s from %s", content_type, url)
            return None
        content_length = int(response.headers.get("Content-Length", "0") or 0)
        if content_length > MAX_LOGO_BYTES:
            LOGGER.warning("Skipping oversized logo from %s", url)
            return None
        destination = destination_stem.with_suffix(suffix)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        total = 0
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_LOGO_BYTES:
                    raise ValueError("logo exceeded 5 MB")
                output.write(chunk)
        temporary.replace(destination)
        return destination
    except (OSError, requests.RequestException, ValueError) as exc:
        # Logo failures should never prevent rankings or game predictions from being produced.
        LOGGER.warning("Could not cache team logo %s: %s", url, exc)
        for candidate in destination_stem.parent.glob(f"{destination_stem.name}.*.tmp"):
            candidate.unlink(missing_ok=True)
        return None


def build_team_branding(
    teams: pd.DataFrame,
    season: int,
    logos_dir: Path,
    *,
    session: requests.Session | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """Return current-season branding and cache each available logo exactly once."""
    if teams.empty or "team" not in teams.columns:
        return pd.DataFrame(columns=BRANDING_COLUMNS)
    current = teams.copy()
    if "season" in current.columns:
        current = current[pd.to_numeric(current["season"], errors="coerce").eq(season)]
    current = current.dropna(subset=["team"]).drop_duplicates("team", keep="last")
    logos_dir.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    http.headers.setdefault("User-Agent", "cfb-ranking-model/0.3")
    rows: list[dict[str, Any]] = []
    for _, team in current.iterrows():
        team_name = str(team["team"])
        urls = _logo_urls(team.get("logos_json"))
        logo_url = urls[0] if urls else ""
        stem = _safe_stem(team.get("team_id"), team_name)
        existing = next(
            (
                path
                for suffix in ALLOWED_CONTENT_TYPES.values()
                if (path := logos_dir / f"{stem}{suffix}").is_file()
            ),
            None,
        )
        cached = existing
        if cached is None and logo_url:
            cached = _download_logo(logo_url, logos_dir / stem, http, timeout_seconds)
        rows.append(
            {
                "team": team_name,
                "team_id": team.get("team_id"),
                "logo_url": logo_url,
                "logo_path": f"team_logos/{cached.name}" if cached else "",
                "team_color": _clean_text(team.get("color")),
                "alt_color": _clean_text(team.get("alternate_color")),
            }
        )
    return pd.DataFrame(rows, columns=BRANDING_COLUMNS)
