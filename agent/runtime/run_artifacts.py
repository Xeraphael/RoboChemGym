from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.planning.models import AgentPlan


_PLAN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_PLAN_ID_LENGTH = 128


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path

    @classmethod
    def open(cls, run_dir: Path) -> "RunArtifacts":
        run_dir = Path(run_dir).resolve()
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError("resume run directory must be a regular directory")
        required = (
            "agent_plan.json",
            "validation_report.json",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(
                "resume run directory is incomplete: " + ", ".join(missing)
            )
        return cls(run_dir=run_dir)

    @classmethod
    def create(
        cls,
        root: Path,
        plan_id: str,
        *,
        now: datetime | None = None,
    ) -> "RunArtifacts":
        if (
            not isinstance(plan_id, str)
            or len(plan_id) > _MAX_PLAN_ID_LENGTH
            or _PLAN_ID_PATTERN.fullmatch(plan_id) is None
        ):
            raise ValueError("plan_id must match ^[a-z][a-z0-9_]*$ and be at most 128 characters")

        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        now = now or datetime.now()
        base_name = f"{now:%Y%m%d_%H%M%S}_{plan_id}"
        suffix = 0
        while True:
            candidate_name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
            run_dir = root / candidate_name
            try:
                run_dir.mkdir(exist_ok=False)
            except FileExistsError:
                suffix += 1
                continue
            break
        (run_dir / "legacy").mkdir(exist_ok=False)
        return cls(run_dir=run_dir)

    @property
    def plan_path(self) -> Path:
        return self.run_dir / "agent_plan.json"

    @property
    def validation_path(self) -> Path:
        return self.run_dir / "validation_report.json"

    @property
    def repair_history_path(self) -> Path:
        return self.run_dir / "repair_history.json"

    @property
    def execution_report_path(self) -> Path:
        return self.run_dir / "execution_report.json"

    @property
    def scene_preflight_path(self) -> Path:
        return self.run_dir / "scene_preflight.json"

    @property
    def trajectory_path(self) -> Path:
        return self.run_dir / "trajectory.json"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def legacy_equipment_path(self) -> Path:
        return self.run_dir / "legacy" / "equipment.txt"

    @property
    def legacy_actions_path(self) -> Path:
        return self.run_dir / "legacy" / "actions.txt"

    def write_protocol(self, text: str) -> None:
        (self.run_dir / "input_protocol.txt").write_text(text, encoding="utf-8")

    def write_plan(self, plan: AgentPlan) -> None:
        self.plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    def write_json(self, path: Path, value: Any) -> None:
        run_dir = self.run_dir.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError("JSON artifact path must be contained within run_dir") from exc

        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        candidate.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_legacy_exports(self, plan: AgentPlan) -> None:
        equipment = "\n".join(obj.instance_name for obj in plan.scene.objects) + "\n"
        actions = "\n".join(
            f"{step.id}: {step.type.value} {step.object or ''} {step.target or ''}".rstrip()
            for step in plan.actions
        ) + "\n"
        self.legacy_equipment_path.write_text(equipment, encoding="utf-8")
        self.legacy_actions_path.write_text(actions, encoding="utf-8")
