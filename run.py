"""Installation-free launcher for the College Football Ranking Lab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
runtime_base = Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT / ".runtime"))
RUNTIME_PACKAGES = runtime_base / "CFBRankingRuntime" / "py314" / "packages"
SOURCE = PROJECT_ROOT / "src"

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(RUNTIME_PACKAGES))

try:
    from cfb_rankings.cli import main
except ModuleNotFoundError as exc:
    missing = exc.name or "a required dependency"
    raise SystemExit(
        f"Missing {missing}. Run .\\setup-offline.ps1 once, then retry this command."
    ) from exc


if __name__ == "__main__":
    raise SystemExit(main())

