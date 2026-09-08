from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from .cfbd import CFBDClient
from .config import load_settings
from .ingest import IngestionPipeline
from .pipeline import build_features, create_data_audit, generate_predictions, train_models


def _json_print(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfb-rankings",
        description="Live AP forecast, independent rankings, and game-margin predictions",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser("bootstrap", help="Download historical and current data")
    bootstrap.add_argument("--start-year", type=int, default=2014)
    bootstrap.add_argument("--end-year", type=int, default=None)

    refresh = commands.add_parser("refresh", help="Refresh one season from the API")
    refresh.add_argument("--season", type=int, default=None)

    commands.add_parser("audit", help="Write and display the CSV data-quality audit")
    commands.add_parser("build-features", help="Create leakage-safe model features")
    commands.add_parser("train", help="Select and train both model families")
    commands.add_parser("predict", help="Generate current rankings and next-game forecasts")

    run_all = commands.add_parser("run-all", help="Refresh current data through predictions")
    run_all.add_argument("--season", type=int, default=None)
    commands.add_parser("dashboard", help="Launch the Streamlit dashboard")
    serve = commands.add_parser("serve", help="Serve the HTML frontend and JSON API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.project_root)

    if args.command in {"bootstrap", "refresh", "run-all"}:
        if not settings.cfbd_api_key:
            raise SystemExit(
                "CFBD_API_KEY is missing. Copy .env.example to .env and add your API key."
            )
        client = CFBDClient(settings.cfbd_api_key)
        ingestion = IngestionPipeline(client, settings)

    if args.command == "bootstrap":
        _json_print(ingestion.bootstrap(args.start_year, args.end_year or settings.season))
    elif args.command == "refresh":
        season = args.season or settings.season
        _json_print(ingestion.bootstrap(season, season))
    elif args.command == "audit":
        print(create_data_audit(settings).to_string(index=False))
    elif args.command == "build-features":
        _json_print(build_features(settings))
    elif args.command == "train":
        _json_print(train_models(settings))
    elif args.command == "predict":
        _json_print(generate_predictions(settings))
    elif args.command == "run-all":
        season = args.season or settings.season
        results = {
            "ingestion": ingestion.bootstrap(season, season),
            "audit": create_data_audit(settings).to_dict(orient="records"),
            "features": build_features(settings),
            "predictions": generate_predictions(settings),
        }
        _json_print(results)
    elif args.command == "dashboard":
        executable = shutil.which("streamlit")
        if not executable:
            raise SystemExit("Streamlit is not installed. Run: pip install -e .")
        dashboard = Path(__file__).with_name("dashboard.py")
        return subprocess.call([executable, "run", str(dashboard)])
    elif args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(
            create_app(settings),
            host=args.host,
            port=args.port,
            log_level="info",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
