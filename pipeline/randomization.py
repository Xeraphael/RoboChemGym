from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.contracts import RandomizationConfig


def derive_episode_seed(root_seed: int, episode_index: int) -> int:
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    sequence = np.random.SeedSequence([root_seed, episode_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def resolve_randomization(
    config: RandomizationConfig, episode_index: int
) -> dict[str, Any]:
    if episode_index >= config.episodes:
        raise ValueError("episode_index exceeds configured episode count")
    episode_seed = derive_episode_seed(config.seed, episode_index)
    rng = np.random.default_rng(episode_seed)
    positions = {}
    for path, ranges in config.scene.object_position.items():
        positions[path] = [
            float(rng.uniform(axis.minimum, axis.maximum))
            for axis in (ranges.x, ranges.y, ranges.z)
        ]
    yaw = {
        path: float(rng.uniform(value.minimum, value.maximum))
        for path, value in config.scene.object_yaw.items()
    }
    lighting = {}
    for path, values in config.scene.lighting.items():
        resolved = {
            "intensity": float(
                rng.uniform(values.intensity.minimum, values.intensity.maximum)
            )
        }
        if values.color_temperature is not None:
            resolved["color_temperature"] = float(
                rng.uniform(
                    values.color_temperature.minimum,
                    values.color_temperature.maximum,
                )
            )
        lighting[path] = resolved
    material = {}
    for path, values in config.scene.material.items():
        candidate_index = int(rng.integers(0, len(values.candidates)))
        material[path] = {
            "candidate_index": candidate_index,
            "material_path": values.candidates[candidate_index],
        }
    return {
        "root_seed": config.seed,
        "episode_seed": episode_seed,
        "episode_index": episode_index,
        "rng": "numpy.PCG64",
        "requested": config.model_dump(mode="json"),
        "resolved": {
            "object_position": positions,
            "object_yaw": yaw,
            "lighting": lighting,
            "material": material,
        },
    }


class SceneRandomizer:
    def __init__(self, config: RandomizationConfig, stage, object_utils):
        self.config = config
        self.stage = stage
        self.object_utils = object_utils
        self.current: dict[str, Any] | None = None
        self._validate_stage()

    def _validate_stage(self) -> None:
        configured = (
            self.config.scene.object_position
            | self.config.scene.object_yaw
            | self.config.scene.camera_pose
            | self.config.scene.lighting
            | self.config.scene.material
            | self.config.physics.friction
            | self.config.physics.mass_scale
        )
        missing = [path for path in configured if not self.stage.GetPrimAtPath(path).IsValid()]
        if missing:
            raise ValueError("randomization prims do not exist: " + ", ".join(missing))
        unsupported = {
            "camera_pose": self.config.scene.camera_pose,
            "friction": self.config.physics.friction,
            "mass_scale": self.config.physics.mass_scale,
        }
        enabled = [name for name, values in unsupported.items() if values]
        if enabled:
            raise ValueError(
                "configured randomization is not implemented for existing assets: "
                + ", ".join(enabled)
            )
        for path, values in self.config.scene.lighting.items():
            prim = self.stage.GetPrimAtPath(path)
            if not prim.GetAttribute("inputs:intensity").IsValid():
                raise ValueError(f"light has no intensity attribute: {path}")
            if values.color_temperature is not None and (
                not prim.GetAttribute("inputs:colorTemperature").IsValid()
                or not prim.GetAttribute("inputs:enableColorTemperature").IsValid()
            ):
                raise ValueError(f"light has no color temperature attributes: {path}")
        if self.config.scene.material:
            from pxr import UsdShade

            for target_path, values in self.config.scene.material.items():
                for material_path in values.candidates:
                    material = UsdShade.Material.Get(self.stage, material_path)
                    if not material:
                        raise ValueError(
                            f"material candidate is not a USD material: {material_path}"
                        )

    def apply(self, episode_index: int) -> dict[str, Any]:
        sample = resolve_randomization(self.config, episode_index)
        for path, position in sample["resolved"]["object_position"].items():
            self.object_utils.set_object_position(
                object_path=path, position=np.asarray(position, dtype=float)
            )
        if sample["resolved"]["object_yaw"]:
            raise ValueError("object_yaw requires an existing rotation authoring adapter")
        for path, values in sample["resolved"]["lighting"].items():
            prim = self.stage.GetPrimAtPath(path)
            prim.GetAttribute("inputs:intensity").Set(values["intensity"])
            if "color_temperature" in values:
                prim.GetAttribute("inputs:enableColorTemperature").Set(True)
                prim.GetAttribute("inputs:colorTemperature").Set(
                    values["color_temperature"]
                )
        if sample["resolved"]["material"]:
            from pxr import UsdShade

            for target_path, values in sample["resolved"]["material"].items():
                target = self.stage.GetPrimAtPath(target_path)
                material = UsdShade.Material.Get(
                    self.stage, values["material_path"]
                )
                UsdShade.MaterialBindingAPI(target).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants
                )
        self.current = sample
        return sample
