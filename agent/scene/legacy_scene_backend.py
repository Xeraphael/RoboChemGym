from pathlib import Path
import shutil

from agent.scene.anchor_resolver import matching_anchor_prims
from agent.scene.scene_compiler import ResolvedSceneObject
from agent.scene.scene_preflight import ScenePreflightIssue, ScenePreflightReport


class LegacySceneBackend:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def equipment_requests(
        objects: list[ResolvedSceneObject],
    ) -> list[tuple[str, Path, bool]]:
        return [
            (obj.instance_name, Path(obj.usd_path), "pick" in obj.supported_actions)
            for obj in objects
        ]

    def build(
        self,
        objects,
        *,
        output_usd: Path,
        output_json: Path,
        layout_profile: dict,
    ) -> None:
        from agent.scene.extractor.scene_extractor import SceneExtractor
        from agent.scene.optimization.position_optimizer import PositionOptimizer
        from agent.scene.optimization.position_updater import PositionUpdater

        reference_scene = self.root / "protocols/Level2_Protocol1/scene.usd"
        if not reference_scene.is_file():
            raise FileNotFoundError(
                f"public reference scene does not exist: {reference_scene}"
            )
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_scene, output_usd)

        extractor = SceneExtractor(scenes_dir=str(output_usd.parent))
        extracted = extractor.extract(output_usd.name)
        if extracted is None:
            raise RuntimeError("scene extraction failed")

        optimizer = PositionOptimizer.from_profile(Path(extracted), layout_profile)
        positions = optimizer.optimize(
            grid_resolution=float(layout_profile["grid_resolution"])
        )
        if positions is None:
            raise RuntimeError("scene layout failed")

        preferred = layout_profile.get("preferred_positions", {})
        for obj in objects:
            if obj.instance_name in preferred:
                positions[f"/World/{obj.instance_name}"] = list(
                    preferred[obj.instance_name]
                )
        optimizer.save_optimized_positions(positions, output_json)
        PositionUpdater(scenes_dir=str(output_usd.parent)).apply_positions_to_usd(
            output_json,
            output_usd,
            in_place=True,
        )

    @staticmethod
    def _required_anchor_match_count(
        Usd,
        instance_prim,
        instance_path: str,
        anchor: str,
    ) -> int:
        return len(
            matching_anchor_prims(
                Usd,
                instance_prim,
                instance_path,
                anchor,
            )
        )

    @staticmethod
    def _scene_json_validation_error(scene_json_path: Path) -> str | None:
        import json
        import math

        try:
            scene_data = json.loads(
                Path(scene_json_path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "file is missing, unreadable, or invalid JSON"

        if not isinstance(scene_data, dict):
            return "root must be an object"

        def is_finite_vector3(value) -> bool:
            return (
                isinstance(value, list)
                and len(value) == 3
                and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                    for item in value
                )
            )

        for prim_path, data in scene_data.items():
            if not isinstance(data, dict):
                return f"entry {prim_path} must be an object"
            if not isinstance(data.get("prim_name"), str):
                return f"entry {prim_path} must have a string prim_name"
            if not is_finite_vector3(data.get("position")):
                return f"entry {prim_path} must have a finite 3D position"

            if "bounding_box" in data:
                bounding_box = data["bounding_box"]
                if not isinstance(bounding_box, dict):
                    return f"entry {prim_path} bounding_box must be an object"
                if "size" in bounding_box and not is_finite_vector3(
                    bounding_box["size"]
                ):
                    return (
                        f"entry {prim_path} bounding_box size must be a finite 3D vector"
                    )

        return None

    def preflight(
        self,
        objects,
        *,
        usd_path: Path,
        scene_json_path: Path,
        layout_profile: dict,
    ) -> ScenePreflightReport:
        from pxr import Usd
        import numpy as np

        from agent.scene.optimization.position_optimizer import PositionOptimizer

        try:
            stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
        except Exception:
            stage = None
        if stage is None:
            return ScenePreflightReport(
                passed=False,
                issues=(
                    ScenePreflightIssue(
                        code="INVALID_USD",
                        message=f"cannot open {usd_path}",
                    ),
                ),
            )

        issues = []
        object_ids = {obj.instance_name: obj.id for obj in objects}
        for obj in objects:
            prim_path = f"/World/{obj.instance_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                issues.append(ScenePreflightIssue(
                    code="MISSING_PRIM",
                    object_id=obj.id,
                    message=f"missing {prim_path}",
                ))
            for capability in obj.required_capabilities:
                for anchor in obj.required_anchors.get(capability, []):
                    match_count = self._required_anchor_match_count(
                        Usd,
                        prim,
                        prim_path,
                        anchor,
                    )
                    if match_count == 0:
                        issues.append(ScenePreflightIssue(
                            code="MISSING_ANCHOR",
                            object_id=obj.id,
                            message=f"missing anchor {anchor}",
                        ))
                    elif match_count > 1:
                        issues.append(ScenePreflightIssue(
                            code="AMBIGUOUS_ANCHOR",
                            object_id=obj.id,
                            message=f"ambiguous anchor {anchor}",
                        ))

        scene_json_error = self._scene_json_validation_error(scene_json_path)
        if scene_json_error is not None:
            issues.append(ScenePreflightIssue(
                code="INVALID_SCENE_JSON",
                message=f"invalid scene JSON {scene_json_path}: {scene_json_error}",
            ))
            return ScenePreflightReport(passed=False, issues=tuple(issues))
        optimizer = PositionOptimizer.from_profile(
            scene_json_path,
            layout_profile,
        )
        scene_names = {
            data.get("prim_name")
            for data in optimizer.scene_data.values()
            if isinstance(data, dict)
        }
        for obj in objects:
            if obj.instance_name not in scene_names:
                issues.append(ScenePreflightIssue(
                    code="SCENE_JSON_NAME_MISMATCH",
                    object_id=obj.id,
                    message=f"{obj.instance_name} is missing from scene JSON",
                ))

        for scene_obj in optimizer.objects:
            position = np.asarray(scene_obj.position, dtype=float)
            object_id = object_ids.get(scene_obj.prim_name)
            if not optimizer.ellipse_constraint.contains(position[0], position[1]):
                issues.append(ScenePreflightIssue(
                    code="OUT_OF_REACH",
                    object_id=object_id,
                    message=(
                        f"{scene_obj.prim_name} is outside the configured reachable region"
                    ),
                ))
            if abs(position[2] - optimizer.z_height) > 0.001:
                issues.append(ScenePreflightIssue(
                    code="INVALID_SURFACE_HEIGHT",
                    object_id=object_id,
                    message=f"{scene_obj.prim_name} has z={position[2]}",
                ))

        for index, first in enumerate(optimizer.objects):
            for second in optimizer.objects[index + 1:]:
                first_position = np.asarray(first.position, dtype=float)
                second_position = np.asarray(second.position, dtype=float)
                if optimizer.check_collision(
                    first,
                    first_position,
                    second,
                    second_position,
                ):
                    issues.append(ScenePreflightIssue(
                        code="INITIAL_COLLISION",
                        object_id=object_ids.get(first.prim_name),
                        message=f"{first.prim_name} collides with {second.prim_name}",
                    ))
                center_distance = float(
                    np.linalg.norm(
                        first_position[:2] - second_position[:2]
                    )
                )
                if center_distance + 1e-12 < optimizer.minimum_spacing:
                    issues.append(ScenePreflightIssue(
                        code="INSUFFICIENT_SPACING",
                        object_id=object_ids.get(first.prim_name),
                        message=(
                            f"{first.prim_name} and {second.prim_name} are "
                            f"{center_distance:.6f}m apart; minimum spacing is "
                            f"{optimizer.minimum_spacing:.6f}m"
                        ),
                    ))

        return ScenePreflightReport(passed=not issues, issues=issues)
