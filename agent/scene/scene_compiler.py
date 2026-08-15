from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.planning.models import AgentPlan
from agent.planning.registry import CapabilityRegistry
from agent.planning.validator import PlanValidator
from agent.runtime.run_artifacts import RunArtifacts
from agent.scene.isaac_scene_worker import SceneWorkerError
from agent.scene.scene_preflight import ScenePreflightReport


class SceneCompileError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        self.code = code
        self.returncode = returncode
        self.stdout = str(stdout or "")[-4000:]
        self.stderr = str(stderr or "")[-4000:]
        super().__init__(message)


class ResolvedSceneObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    asset_id: str
    instance_name: str
    category: str
    usd_path: str
    supported_actions: list[str]
    required_anchors: dict[str, list[str]]
    required_capabilities: list[str] = Field(default_factory=list)


class SceneBackend(Protocol):
    def build(self, objects, *, output_usd: Path, output_json: Path, layout_profile: dict) -> None:
        raise NotImplementedError

    def preflight(self, objects, *, usd_path: Path, scene_json_path: Path, layout_profile: dict) -> ScenePreflightReport:
        raise NotImplementedError


class SceneCompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    usd_path: Path
    scene_json_path: Path
    config_path: Path
    preflight: ScenePreflightReport


class SceneCompiler:
    def __init__(self, registry: CapabilityRegistry, backend: SceneBackend, root: Path):
        self.registry = registry
        self.backend = backend
        self.root = root
        profiles = yaml.safe_load((root / "agent/scene/layout_profiles.yaml").read_text(encoding="utf-8"))
        self._layout_profile = profiles["profiles"]["lab_table_franka"]

    @property
    def layout_profile(self) -> dict:
        return deepcopy(self._layout_profile)

    def compile(self, plan: AgentPlan, artifacts: RunArtifacts) -> SceneCompileResult:
        usd_path = artifacts.run_dir / "scene.usd"
        scene_json_path = artifacts.run_dir / "scene.json"
        config_path = artifacts.run_dir / "config.yaml"
        self._remove_stale_file(config_path)

        artifacts.write_plan(plan)
        validation = PlanValidator(self.registry).validate(plan)
        artifacts.write_json(artifacts.validation_path, validation)
        if not validation.valid:
            raise SceneCompileError("PLAN_INVALID", "plan validation failed")

        objects = tuple(self.resolve_objects(plan))
        layout_profile = self._layout_for_plan(plan, objects)

        self._remove_stale_file(usd_path)
        self._remove_stale_file(scene_json_path)
        try:
            self.backend.build(
                self._copy_objects(objects),
                output_usd=usd_path,
                output_json=scene_json_path,
                layout_profile=deepcopy(layout_profile),
            )
        except SceneWorkerError as exc:
            self._remove_scene_outputs(usd_path, scene_json_path)
            raise SceneCompileError(
                exc.code,
                str(exc),
                returncode=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            ) from exc
        except Exception:
            self._remove_scene_outputs(usd_path, scene_json_path)
            raise
        if not self._is_regular_file(usd_path) or not self._is_regular_file(scene_json_path):
            self._remove_scene_outputs(usd_path, scene_json_path)
            raise SceneCompileError(
                "SCENE_OUTPUT_MISSING",
                "scene backend must produce regular scene.usd and scene.json files",
            )
        preflight = self._preflight_resolved(
            objects,
            artifacts,
            layout_profile=layout_profile,
        )
        artifacts.write_json(artifacts.scene_preflight_path, preflight)
        if not preflight.passed:
            raise SceneCompileError("SCENE_PREFLIGHT_FAILED", "scene preflight failed")

        config = yaml.safe_load(
            (self.root / "config/example_protocol.yaml").read_text(encoding="utf-8")
        )
        config.update({
            "name": plan.plan_id,
            "task_type": "all",
            "controller_type": "plan_executor",
            "mode": "execute",
            "usd_path": str(usd_path),
            "max_episodes": 1,
        })
        config["agent"] = {
            "execution_backend": "plan_executor",
            "plan_path": str(artifacts.plan_path),
            "validation_report_path": str(artifacts.validation_path),
            "scene_preflight_path": str(artifacts.scene_preflight_path),
            "execution_report_path": str(artifacts.execution_report_path),
            "trajectory_path": str(artifacts.trajectory_path),
            "state_anchors": {
                obj.instance_name: sorted({
                    anchor
                    for capability in obj.required_capabilities
                    for anchor in obj.required_anchors.get(capability, [])
                })
                for obj in objects
            },
        }
        config["task"] = {
            "obj_paths": [
                {"path": f"/World/{obj.instance_name}"} for obj in objects
            ],
            "placement_alignments": self._placement_alignments(plan, objects),
        }
        config["multi_run"]["run_dir"] = str(artifacts.run_dir / "data")
        config["hydra"]["run"]["dir"] = str(artifacts.run_dir / "hydra")
        result = SceneCompileResult(
            usd_path=usd_path,
            scene_json_path=scene_json_path,
            config_path=config_path,
            preflight=preflight,
        )
        self._write_config_atomic(config_path, config)
        return result

    def resolve_objects(self, plan: AgentPlan) -> list[ResolvedSceneObject]:
        objects = []
        for obj in plan.scene.objects:
            asset = self.registry.assets.resolve(obj.asset_id, obj.properties)
            required_capabilities = set()
            for step in plan.actions:
                if step.object == obj.id:
                    required_capabilities.add(step.type.value)
                if step.target == obj.id:
                    required_capabilities.add("place_target" if step.type.value == "place" else step.type.value)
            objects.append(ResolvedSceneObject(
                id=obj.id,
                asset_id=obj.asset_id,
                instance_name=obj.instance_name,
                category=asset.category,
                usd_path=asset.usd_path,
                supported_actions=asset.supported_actions,
                required_anchors=asset.required_anchors,
                required_capabilities=sorted(required_capabilities),
            ))
        return objects

    @staticmethod
    def _placement_alignments(
        plan: AgentPlan,
        objects: Sequence[ResolvedSceneObject],
    ) -> list[dict[str, str]]:
        by_id = {obj.id: obj for obj in objects}
        alignments = []
        seen = set()
        for step in plan.actions:
            source = by_id.get(step.object)
            target = by_id.get(step.target)
            if (
                step.type.value != "place"
                or source is None
                or target is None
                or target.category != "placement_target"
            ):
                continue
            pair = (source.instance_name, target.instance_name)
            if pair in seen:
                continue
            seen.add(pair)
            alignments.append({
                "object_path": f"/World/{source.instance_name}",
                "target_path": f"/World/{target.instance_name}",
            })
        return alignments

    def preflight(self, plan: AgentPlan, artifacts: RunArtifacts) -> ScenePreflightReport:
        objects = tuple(self.resolve_objects(plan))
        return self._preflight_resolved(
            objects,
            artifacts,
            layout_profile=self._layout_for_plan(plan, objects),
        )

    def _preflight_resolved(
        self,
        objects: Sequence[ResolvedSceneObject],
        artifacts: RunArtifacts,
        *,
        layout_profile: dict | None = None,
    ) -> ScenePreflightReport:
        try:
            backend_report = self.backend.preflight(
                self._copy_objects(objects),
                usd_path=artifacts.run_dir / "scene.usd",
                scene_json_path=artifacts.run_dir / "scene.json",
                layout_profile=deepcopy(
                    self._layout_profile
                    if layout_profile is None
                    else layout_profile
                ),
            )
        except SceneWorkerError as exc:
            raise SceneCompileError(
                exc.code,
                str(exc),
                returncode=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            ) from exc
        try:
            report_data = backend_report.model_dump(mode="python", warnings="error")
            return ScenePreflightReport(**report_data)
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise SceneCompileError(
                "SCENE_PREFLIGHT_INVALID",
                "scene backend returned an invalid preflight report",
            ) from exc

    def _layout_for_plan(
        self,
        plan: AgentPlan,
        objects: Sequence[ResolvedSceneObject],
    ) -> dict:
        profile = deepcopy(self._layout_profile)
        slots = profile.pop("control_slots", {})
        by_id = {obj.id: obj for obj in objects}
        preferred: dict[str, list[float]] = {}

        for obj in objects:
            if obj.category == "device" and "device" in slots:
                preferred[obj.instance_name] = deepcopy(slots["device"])
            elif obj.category == "placement_target" and "placement_target" in slots:
                preferred[obj.instance_name] = deepcopy(slots["placement_target"])

        pour_targets = {
            step.target for step in plan.actions
            if step.type.value == "pour" and step.target is not None
        }
        pour_sources = {
            step.object for step in plan.actions
            if step.type.value == "pour" and step.object is not None
        }
        assigned_containers = set()
        for object_id, slot_name in (
            *((object_id, "primary_container") for object_id in pour_targets),
            *((object_id, "secondary_container") for object_id in pour_sources),
        ):
            obj = by_id.get(object_id)
            if obj is not None and slot_name in slots:
                preferred[obj.instance_name] = deepcopy(slots[slot_name])
                assigned_containers.add(object_id)

        available_container_slots = iter(
            name for name in ("primary_container", "secondary_container")
            if name in slots
        )
        for obj in objects:
            if obj.category != "container" or obj.id in assigned_containers:
                continue
            slot_name = next(available_container_slots, None)
            if slot_name is None:
                break
            preferred[obj.instance_name] = deepcopy(slots[slot_name])

        profile["preferred_positions"] = preferred
        return profile

    @staticmethod
    def _copy_objects(objects: Sequence[ResolvedSceneObject]) -> list[ResolvedSceneObject]:
        return [obj.model_copy(deep=True) for obj in objects]

    @staticmethod
    def _remove_stale_file(path: Path) -> None:
        if path.is_file() or path.is_symlink():
            path.unlink()

    @classmethod
    def _remove_scene_outputs(cls, usd_path: Path, scene_json_path: Path) -> None:
        for path in (usd_path, scene_json_path):
            try:
                cls._remove_stale_file(path)
            except OSError:
                pass

    @staticmethod
    def _write_config_atomic(config_path: Path, config: dict) -> None:
        temporary_path = config_path.with_name(f".{config_path.name}.tmp")
        try:
            temporary_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            temporary_path.replace(config_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _is_regular_file(path: Path) -> bool:
        return path.is_file() and not path.is_symlink()
