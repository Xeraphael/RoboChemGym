from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class DuplicateAliasError(ValueError):
    pass


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, f"expected a mapping node, but found {node.id}", node.start_mark)
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_DuplicateKeySafeLoader)


def _validate_asset_path(usd_path: str) -> None:
    path = Path(usd_path)
    if not usd_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"asset path must be a contained repository-relative path: {usd_path!r}")


class AssetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aliases: list[str] = Field(default_factory=list)
    category: str
    usd_path: str | None = None
    variants: dict[str, str] = Field(default_factory=dict)
    variant_property: str | None = None
    supported_actions: list[str]
    required_anchors: dict[str, list[str]] = Field(default_factory=dict)
    action_defaults: dict[str, dict[str, object]] = Field(default_factory=dict)
    variant_action_defaults: dict[str, dict[str, dict[str, object]]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_path_contract(self):
        if (self.usd_path is not None) == bool(self.variants):
            raise ValueError("asset must define exactly one of usd_path or variants")
        if self.variants and (not self.variant_property or "default" not in self.variants):
            raise ValueError("variant assets require variant_property and a default variant")
        for usd_path in [self.usd_path] if self.usd_path is not None else self.variants.values():
            _validate_asset_path(usd_path)
        return self

    def all_usd_paths(self) -> list[str]:
        return [self.usd_path] if self.usd_path else list(self.variants.values())

    def select_usd_path(self, properties: dict[str, Any]) -> str:
        if self.usd_path:
            return self.usd_path
        variant = str(properties.get(self.variant_property or "", "default"))
        if variant not in self.variants:
            variant = "default"
        return self.variants[variant]


class ResolvedAsset(BaseModel):
    asset_id: str
    category: str
    usd_path: str
    supported_actions: list[str]
    required_anchors: dict[str, list[str]]
    action_defaults: dict[str, dict[str, object]]


class ParameterConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["number", "string"]
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] = Field(default_factory=list)

    def validate_value(self, name: str, value: Any) -> None:
        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{name} must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"{name} must be <= {self.maximum}")
        elif not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        if self.choices and value not in self.choices:
            raise ValueError(f"{name} must be one of {self.choices}")


class ActionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_categories: list[str]
    target_categories: list[str]
    adapter: str
    verifier: str
    required_object: bool
    required_target: bool
    max_frames: int = Field(gt=0)
    preconditions: list[str]
    effects: list[str]
    supported_parameters: list[str]
    tunable_parameters: list[str]
    parameter_constraints: dict[str, ParameterConstraint]
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    degradable_modifiers: list[str]

    def validate_parameters(self, parameters: dict[str, Any], *, tunable_only: bool = False) -> None:
        allowed = set(self.tunable_parameters if tunable_only else self.supported_parameters)
        rejected = set(parameters) - allowed
        if rejected:
            raise ValueError(f"unsupported parameters: {sorted(rejected)}")
        for name, value in parameters.items():
            self.parameter_constraints[name].validate_value(name, value)


class AssetRegistry:
    def __init__(self, definitions: dict[str, AssetDefinition]):
        self._definitions = deepcopy(definitions)
        self._aliases: dict[str, str] = {}
        for asset_id, definition in self._definitions.items():
            if not set(definition.required_anchors).issubset(definition.supported_actions):
                raise ValueError(f"{asset_id} anchors reference unsupported capabilities")
            for alias in [asset_id, *definition.aliases]:
                key = alias.casefold()
                if key in self._aliases and self._aliases[key] != asset_id:
                    raise DuplicateAliasError(f"duplicate asset alias: {alias}")
                self._aliases[key] = asset_id

    @property
    def definitions(self) -> dict[str, AssetDefinition]:
        return deepcopy(self._definitions)

    def canonical_id(self, name: str) -> str:
        return self._aliases[name.casefold()]

    def get(self, asset_id: str) -> AssetDefinition:
        return deepcopy(self._definitions[asset_id])

    def resolve(self, asset_id: str, properties: dict[str, Any]) -> ResolvedAsset:
        definition = self.get(asset_id)
        variant = str(properties.get(definition.variant_property or "", "default"))
        if variant not in definition.variants:
            variant = "default"
        action_defaults = deepcopy(definition.action_defaults)
        for action_name, values in definition.variant_action_defaults.get(
            variant, {}
        ).items():
            action_defaults.setdefault(action_name, {}).update(deepcopy(values))
        return ResolvedAsset(
            asset_id=asset_id,
            category=definition.category,
            usd_path=definition.select_usd_path(properties),
            supported_actions=deepcopy(definition.supported_actions),
            required_anchors=deepcopy(definition.required_anchors),
            action_defaults=action_defaults,
        )


class ActionRegistry:
    def __init__(self, definitions: dict[str, ActionDefinition]):
        self._definitions = deepcopy(definitions)
        allowed_preconditions = {"gripper_empty", "holding_object", "target_closed", "target_open"}
        allowed_effects = {
            "hold_object",
            "release_object",
            "object_at_target",
            "target_activated",
            "target_open",
            "target_closed",
        }
        for name, definition in self._definitions.items():
            if set(definition.parameter_constraints) != set(definition.supported_parameters):
                raise ValueError(f"{name} parameter constraints must match supported_parameters")
            if not set(definition.tunable_parameters).issubset(definition.supported_parameters):
                raise ValueError(f"{name} tunable_parameters must be supported")
            if not set(definition.preconditions).issubset(allowed_preconditions):
                raise ValueError(f"{name} has unknown symbolic preconditions")
            if not set(definition.effects).issubset(allowed_effects):
                raise ValueError(f"{name} has unknown symbolic effects")
            definition.validate_parameters(definition.default_parameters)

    @property
    def definitions(self) -> dict[str, ActionDefinition]:
        return deepcopy(self._definitions)

    def get(self, action_type: str) -> ActionDefinition:
        return deepcopy(self._definitions[action_type])


class CapabilityRegistry:
    def __init__(self, assets: AssetRegistry, actions: ActionRegistry, root: Path):
        self.assets = assets
        self.actions = actions
        self.root = root
        self._validate_asset_action_defaults()

    def _validate_asset_action_defaults(self) -> None:
        for asset_id, asset in self.assets.definitions.items():
            for variant in asset.variant_action_defaults:
                if variant not in asset.variants:
                    raise ValueError(
                        f"Asset {asset_id} has defaults for unknown variant "
                        f"{variant!r}"
                    )
            self._validate_supported_capabilities(asset_id, asset)
            self._validate_defaults_mapping(
                asset_id,
                asset,
                asset.action_defaults,
                context="action_defaults",
            )
            for variant, defaults in asset.variant_action_defaults.items():
                self._validate_defaults_mapping(
                    asset_id,
                    asset,
                    defaults,
                    context=f"variant {variant!r}",
                )

    def _validate_supported_capabilities(
        self, asset_id: str, asset: AssetDefinition
    ) -> None:
        for capability in asset.supported_actions:
            action_name = "place" if capability == "place_target" else capability
            try:
                action = self.actions.get(action_name)
            except KeyError:
                raise ValueError(
                    f"Asset {asset_id} references unknown action "
                    f"{capability!r}"
                ) from None

            if capability == "place_target":
                valid_categories = action.target_categories
                role = "target"
            else:
                valid_categories = (
                    action.object_categories + action.target_categories
                )
                role = "object or target"
            if asset.category not in valid_categories:
                raise ValueError(
                    f"Asset {asset_id} category {asset.category!r} cannot be "
                    f"the {role} of action {action_name!r}"
                )

    def _validate_defaults_mapping(
        self,
        asset_id: str,
        asset: AssetDefinition,
        defaults_by_action: dict[str, dict[str, object]],
        *,
        context: str,
    ) -> None:
        for action_name, defaults in defaults_by_action.items():
            try:
                action = self.actions.get(action_name)
            except KeyError:
                raise ValueError(
                    f"Asset {asset_id} {context} references unknown action "
                    f"{action_name!r}"
                ) from None
            if action_name not in asset.supported_actions:
                raise ValueError(
                    f"Asset {asset_id} {context} configures unsupported action "
                    f"{action_name!r}"
                )
            if asset.category not in action.object_categories:
                raise ValueError(
                    f"Asset {asset_id} category {asset.category!r} cannot be "
                    f"the object of action {action_name!r}"
                )
            try:
                action.validate_parameters(deepcopy(defaults))
            except ValueError as exc:
                raise ValueError(
                    f"Asset {asset_id} {context} has invalid defaults for "
                    f"action {action_name!r}: {exc}"
                ) from exc

    @classmethod
    def load_default(cls, root: Path) -> "CapabilityRegistry":
        registry_dir = root / "agent" / "planning" / "registry"
        asset_data = _load_yaml(registry_dir / "assets.yaml")["assets"]
        action_data = _load_yaml(registry_dir / "actions.yaml")["actions"]
        return cls(
            assets=AssetRegistry({k: AssetDefinition.model_validate(v) for k, v in asset_data.items()}),
            actions=ActionRegistry({k: ActionDefinition.model_validate(v) for k, v in action_data.items()}),
            root=root,
        )
