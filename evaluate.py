from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.contracts import load_evaluation_config
from pipeline.dataset import atomic_json
from pipeline.metrics import summarize_evaluation


ROOT = Path(__file__).resolve().parent


def create_run_directory(root: Path, now: datetime | None = None) -> tuple[str, Path]:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for suffix in range(100):
        run_id = timestamp if suffix == 0 else f"{timestamp}_{suffix:02d}"
        path = runs_dir / run_id
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return run_id, path
    raise RuntimeError("could not allocate a unique evaluation run directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ACT policy in Isaac")
    parser.add_argument("--config", required=True)
    args, kit_args = parser.parse_known_args()
    config = load_evaluation_config(args.config)
    output_root = ROOT / config.output_dir
    run_id, output_dir = create_run_directory(output_root)
    run_status = {
        "run_id": run_id,
        "run_dir": str(output_dir.relative_to(ROOT)),
        "status": "running",
    }
    atomic_json(output_dir / "status.json", run_status)
    atomic_json(output_root / "latest.json", run_status)
    exit_state = {"completed": False}

    def mark_incomplete() -> None:
        if exit_state["completed"]:
            return
        incomplete = {**run_status, "status": "incomplete"}
        atomic_json(output_dir / "status.json", incomplete)
        atomic_json(output_root / "latest.json", incomplete)

    atexit.register(mark_incomplete)
    resolved_dir = output_dir / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    portable_config = config.model_dump(mode="json")
    for field in ("base_collection_config", "checkpoint_path"):
        portable_config[field] = str(
            Path(portable_config[field]).relative_to(ROOT)
        )
    (output_dir / "resolved_evaluation.yaml").write_text(
        yaml.safe_dump(portable_config, sort_keys=False),
        encoding="utf-8",
    )
    base = yaml.safe_load(
        Path(config.base_collection_config).read_text(encoding="utf-8")
    )
    all_reports = []
    reports_by_seed_set = {}
    for name, seed_set in config.seed_sets.items():
        stage = json.loads(json.dumps(base))
        stage["name"] = f"Level2_Protocol1_{name}_evaluation"
        stage["mode"] = "evaluate"
        stage["controller_type"] = "policy"
        stage["max_episodes"] = seed_set.episodes
        stage["multi_run"]["run_dir"] = str(output_dir / name)
        stage["hydra"]["run"]["dir"] = str(output_dir / name / "hydra")
        stage["collector"]["video"]["every_n_episodes"] = (
            config.video_every_n_episodes
        )
        stage["evaluation"] = {
            "checkpoint_path": str(Path(config.checkpoint_path)),
            "output_dir": str(output_dir / name / "episodes"),
            "max_steps": config.max_steps,
            "max_joint_step": config.max_joint_step,
            "dataset_split": name if name != "reference" else "reference",
            "seed_set": name,
        }
        if seed_set.randomized:
            stage["randomization"]["seed"] = seed_set.root_seed
            stage["randomization"]["episodes"] = seed_set.episodes
        else:
            stage.pop("randomization", None)
        stage_path = resolved_dir / f"{name}.yaml"
        stage_path.write_text(
            yaml.safe_dump(stage, sort_keys=False), encoding="utf-8"
        )
        report_dir = output_dir / name / "episodes"
        command = [
            sys.executable,
            str(ROOT / "main.py"),
            "--config-dir",
            str(resolved_dir),
            "--config-name",
            name,
            "--headless",
            *kit_args,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        reports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(report_dir.glob("episode_*.json"))
        ]
        if len(reports) != seed_set.episodes:
            raise RuntimeError(
                f"{name} evaluation produced {len(reports)} reports; "
                f"expected {seed_set.episodes}"
            )
        reports_by_seed_set[name] = reports
        all_reports.extend(reports)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_id = f"{checkpoint_path.parent.parent.name}/{checkpoint_path.name}"
    summary = {
        **summarize_evaluation(all_reports),
        "status": "completed",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "config_id": Path(args.config).stem,
        "dataset_split": "validation,test,reference",
        "evaluation_seed_sets": {
            name: {
                **seed_set.model_dump(mode="json"),
                "metrics": summarize_evaluation(reports_by_seed_set[name]),
            }
            for name, seed_set in config.seed_sets.items()
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    completed = {**run_status, "status": "completed"}
    atomic_json(output_dir / "status.json", completed)
    atomic_json(output_root / "latest.json", completed)
    exit_state["completed"] = True
    atexit.unregister(mark_incomplete)


if __name__ == "__main__":
    main()
