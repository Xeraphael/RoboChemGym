from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class NumericRange(StrictModel):
    minimum: float
    maximum: float

    @model_validator(mode="before")
    @classmethod
    def accept_pair(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"minimum": value[0], "maximum": value[1]}
        return value

    @model_validator(mode="after")
    def validate_range(self) -> "NumericRange":
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("range endpoints must be finite")
        if self.minimum > self.maximum:
            raise ValueError("range minimum must not exceed maximum")
        return self


class PositionRange(StrictModel):
    x: NumericRange
    y: NumericRange
    z: NumericRange


class LightRandomization(StrictModel):
    intensity: NumericRange
    color_temperature: NumericRange | None = None

    @model_validator(mode="after")
    def validate_light(self) -> "LightRandomization":
        if self.intensity.minimum < 0:
            raise ValueError("light intensity must be non-negative")
        if self.color_temperature is not None and (
            self.color_temperature.minimum < 1000
            or self.color_temperature.maximum > 20000
        ):
            raise ValueError("light color temperature must be within 1000-20000 K")
        return self


class MaterialRandomization(StrictModel):
    candidates: list[str] = Field(min_length=1)

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, value: list[str]) -> list[str]:
        if any(not path or not path.startswith("/") for path in value):
            raise ValueError("material candidate paths must be absolute")
        if len(set(value)) != len(value):
            raise ValueError("material candidate paths must be unique")
        return value


class ReachableWorkspace(StrictModel):
    center: tuple[float, float]
    semi_axes: tuple[float, float]
    rotation: float = 0.0
    z: NumericRange

    @model_validator(mode="after")
    def validate_workspace(self) -> "ReachableWorkspace":
        values = (*self.center, *self.semi_axes, self.rotation)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("workspace values must be finite")
        if any(axis <= 0 for axis in self.semi_axes):
            raise ValueError("workspace semi_axes must be positive")
        return self

    def contains(self, x: float, y: float, z: float) -> bool:
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        dx, dy = x - self.center[0], y - self.center[1]
        local_x = cos_r * dx + sin_r * dy
        local_y = -sin_r * dx + cos_r * dy
        ellipse = (local_x / self.semi_axes[0]) ** 2 + (
            local_y / self.semi_axes[1]
        ) ** 2
        return ellipse <= 1.0 + 1e-12 and self.z.minimum <= z <= self.z.maximum


class SceneRandomization(StrictModel):
    object_position: dict[str, PositionRange] = Field(default_factory=dict)
    object_yaw: dict[str, NumericRange] = Field(default_factory=dict)
    camera_pose: dict[str, dict[str, Any]] = Field(default_factory=dict)
    lighting: dict[str, LightRandomization] = Field(default_factory=dict)
    material: dict[str, MaterialRandomization] = Field(default_factory=dict)

    @field_validator(
        "object_position", "object_yaw", "camera_pose", "lighting", "material"
    )
    @classmethod
    def validate_paths(cls, value: dict[str, Any]) -> dict[str, Any]:
        for path in value:
            if not path or not path.startswith("/"):
                raise ValueError(f"USD prim path must be absolute: {path!r}")
        return value


class PhysicsRandomization(StrictModel):
    friction: dict[str, NumericRange] = Field(default_factory=dict)
    mass_scale: dict[str, NumericRange] = Field(default_factory=dict)

    @field_validator("friction", "mass_scale")
    @classmethod
    def validate_paths(cls, value: dict[str, Any]) -> dict[str, Any]:
        for path in value:
            if not path or not path.startswith("/"):
                raise ValueError(f"USD prim path must be absolute: {path!r}")
        return value


class RandomizationConfig(StrictModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    episodes: int = Field(gt=0)
    retry_policy: str = "reuse_episode_sample"
    reachable_workspace: ReachableWorkspace
    scene: SceneRandomization = Field(default_factory=SceneRandomization)
    physics: PhysicsRandomization = Field(default_factory=PhysicsRandomization)

    @field_validator("retry_policy")
    @classmethod
    def validate_retry_policy(cls, value: str) -> str:
        if value != "reuse_episode_sample":
            raise ValueError("retry_policy must be 'reuse_episode_sample'")
        return value

    @model_validator(mode="after")
    def validate_positions_are_reachable(self) -> "RandomizationConfig":
        workspace = self.reachable_workspace
        for path, position in self.scene.object_position.items():
            corners = (
                (x, y, z)
                for x in (position.x.minimum, position.x.maximum)
                for y in (position.y.minimum, position.y.maximum)
                for z in (position.z.minimum, position.z.maximum)
            )
            if not all(workspace.contains(*corner) for corner in corners):
                raise ValueError(f"object position range is outside workspace: {path}")
        return self


def load_collection_config(path: str | Path) -> tuple[dict[str, Any], RandomizationConfig]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("collection config must contain a YAML mapping")
    randomization = RandomizationConfig.model_validate(data.get("randomization"))
    max_episodes = data.get("max_episodes")
    if max_episodes is not None and max_episodes != randomization.episodes:
        raise ValueError("max_episodes must match randomization.episodes")
    usd_path = data.get("usd_path")
    if not isinstance(usd_path, str) or not usd_path:
        raise ValueError("usd_path must be a non-empty string")
    resolved_usd = Path(usd_path)
    if not resolved_usd.is_absolute():
        resolved_usd = config_path.parents[2] / resolved_usd
    if not resolved_usd.is_file():
        raise ValueError(f"USD asset does not exist: {resolved_usd}")
    return data, randomization


class TrainingModelConfig(StrictModel):
    type: str = "act"
    horizon: int = Field(gt=0)
    arguments: dict[str, Any]

    @field_validator("type")
    @classmethod
    def act_only(cls, value: str) -> str:
        if value != "act":
            raise ValueError("the release-candidate training path supports model.type=act")
        return value


class OptimizerConfig(StrictModel):
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0, default=0.0)


class SchedulerConfig(StrictModel):
    type: str = "cosine"
    warmup_steps: int = Field(ge=0, default=0)

    @field_validator("type")
    @classmethod
    def supported_scheduler(cls, value: str) -> str:
        if value not in {"constant", "cosine"}:
            raise ValueError("scheduler.type must be constant or cosine")
        return value


class CheckpointConfig(StrictModel):
    metric: str = "val_loss"
    mode: str = "min"
    every_n_epochs: int = Field(gt=0, default=1)
    keep_numbered: int = Field(ge=0, default=3)

    @field_validator("mode")
    @classmethod
    def supported_mode(cls, value: str) -> str:
        if value not in {"min", "max"}:
            raise ValueError("checkpoint.mode must be min or max")
        return value


class TrainingConfig(StrictModel):
    dataset_manifest: str
    split_version: str
    model: TrainingModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0, default=0)
    sample_stride: int = Field(gt=0, default=1)
    epochs: int = Field(gt=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    device: str
    output_dir: str
    evaluation_interval: int = Field(gt=0)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    resume: bool = True


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = TrainingConfig.model_validate(data)
    manifest = Path(config.dataset_manifest)
    if not manifest.is_absolute():
        manifest = config_path.parents[2] / manifest
    if not manifest.is_file():
        raise ValueError(f"dataset manifest does not exist: {manifest}")
    return config.model_copy(update={"dataset_manifest": str(manifest)})


class EvaluationSeedSet(StrictModel):
    episodes: int = Field(gt=0)
    root_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    randomized: bool = True

    @model_validator(mode="after")
    def require_randomized_seed(self) -> "EvaluationSeedSet":
        if self.randomized and self.root_seed is None:
            raise ValueError("randomized evaluation seed sets require root_seed")
        if not self.randomized and self.root_seed is not None:
            raise ValueError("non-randomized evaluation seed sets cannot set root_seed")
        return self


class EvaluationConfig(StrictModel):
    base_collection_config: str
    checkpoint_path: str
    output_dir: str
    max_steps: int = Field(gt=0)
    max_joint_step: float = Field(gt=0, le=0.5, default=0.05)
    video_every_n_episodes: int = Field(gt=0, default=10)
    seed_sets: dict[str, EvaluationSeedSet]

    @model_validator(mode="after")
    def require_release_seed_sets(self) -> "EvaluationConfig":
        required = {"validation", "test", "reference"}
        if set(self.seed_sets) != required:
            raise ValueError(f"evaluation seed_sets must be exactly {sorted(required)}")
        if self.seed_sets["reference"].randomized:
            raise ValueError("reference seed set must be non-randomized")
        if (
            self.seed_sets["validation"].root_seed
            == self.seed_sets["test"].root_seed
        ):
            raise ValueError("validation and test seed sets must be distinct")
        return self


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = EvaluationConfig.model_validate(data)
    updates = {}
    for field in ("base_collection_config", "checkpoint_path"):
        value = Path(getattr(config, field))
        if not value.is_absolute():
            value = config_path.parents[2] / value
        if not value.is_file():
            raise ValueError(f"evaluation artifact does not exist: {value}")
        updates[field] = str(value)
    return config.model_copy(update=updates)
