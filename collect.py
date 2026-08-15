from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipeline.contracts import load_collection_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect randomized Isaac episodes")
    parser.add_argument("--config", required=True, help="Collection YAML path")
    args, kit_args = parser.parse_known_args()
    config_path = Path(args.config).resolve()
    load_collection_config(config_path)
    main_path = Path(__file__).resolve().with_name("main.py")
    command = [
        sys.executable,
        str(main_path),
        "--config-dir",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
        "--headless",
        *kit_args,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
