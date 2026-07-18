"""Pure contracts for manifest-driven private production parity and release gating."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Tuple


class PrivateReleaseGateError(ValueError):
    """Raised when private fixture metadata or release evidence is invalid."""


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PrivateReleaseGateError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, field_name: str, *, allow_empty: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PrivateReleaseGateError(f"{field_name} must be an array of strings")
    result = tuple(_non_empty_string(item, field_name) for item in value)
    if not allow_empty and not result:
        raise PrivateReleaseGateError(f"{field_name} cannot be empty")
    if len(result) != len(set(result)):
        raise PrivateReleaseGateError(f"{field_name} cannot contain duplicates")
    return result


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrivateReleaseGateError(f"{field_name} must be an object")
    return value


def _relative_path(value: Any, field_name: str) -> Path:
    resolved = Path(_non_empty_string(value, field_name))
    if resolved.is_absolute():
        raise PrivateReleaseGateError(f"{field_name} must be relative to the manifest")
    if any(part == ".." for part in resolved.parts):
        raise PrivateReleaseGateError(f"{field_name} cannot escape the manifest directory")
    return resolved


def _finite_non_negative(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise PrivateReleaseGateError(f"{field_name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class PrivateImagePairSpec:
    expected: Path
    actual: Path
    maximum_absolute_error: float = 0.0
    mean_absolute_error: float = 0.0
    alpha_maximum_absolute_error: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], field_name: str) -> "PrivateImagePairSpec":
        mapping = _mapping(value, field_name)
        return cls(
            expected=_relative_path(mapping.get("expected"), f"{field_name}.expected"),
            actual=_relative_path(mapping.get("actual"), f"{field_name}.actual"),
            maximum_absolute_error=_finite_non_negative(
                mapping.get("maximum_absolute_error", 0.0),
                f"{field_name}.maximum_absolute_error",
            ),
            mean_absolute_error=_finite_non_negative(
                mapping.get("mean_absolute_error", 0.0),
                f"{field_name}.mean_absolute_error",
            ),
            alpha_maximum_absolute_error=_finite_non_negative(
                mapping.get("alpha_maximum_absolute_error", 0.0),
                f"{field_name}.alpha_maximum_absolute_error",
            ),
        )


@dataclass(frozen=True, slots=True)
class PrivateFixtureSpec:
    fixture_id: str
    source_blend: Path
    legacy_json: Path
    actual_json: Path
    operator: str
    active_object: str
    selected_objects: Tuple[str, ...]
    capabilities: Tuple[str, ...]
    image_pairs: Tuple[PrivateImagePairSpec, ...]
    scene_attributes: Mapping[str, Any]
    scene_custom_properties: Mapping[str, Any]
    object_attributes: Mapping[str, Mapping[str, Any]]
    object_custom_properties: Mapping[str, Mapping[str, Any]]
    operator_kwargs: Mapping[str, Any]
    ignored_paths: Tuple[str, ...]
    accepted_warning_codes: Tuple[str, ...]
    strict_edges: bool
    compare_animations: bool
    absolute_tolerance: float
    relative_tolerance: float
    animated: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "PrivateFixtureSpec":
        field = f"fixtures[{index}]"
        mapping = _mapping(value, field)
        selected = _string_tuple(
            mapping.get("selected_objects", ()),
            f"{field}.selected_objects",
            allow_empty=False,
        )
        active = _non_empty_string(mapping.get("active_object"), f"{field}.active_object")
        if active not in selected:
            raise PrivateReleaseGateError(
                f"{field}.active_object must also appear in selected_objects"
            )
        operator = _non_empty_string(mapping.get("operator"), f"{field}.operator")
        if operator.count(".") != 1:
            raise PrivateReleaseGateError(
                f"{field}.operator must have the form module.operator"
            )
        image_values = mapping.get("image_pairs", ())
        if not isinstance(image_values, (list, tuple)):
            raise PrivateReleaseGateError(f"{field}.image_pairs must be an array")
        image_pairs = tuple(
            PrivateImagePairSpec.from_mapping(item, f"{field}.image_pairs[{pair_index}]")
            for pair_index, item in enumerate(image_values)
        )
        object_attributes = _mapping(
            mapping.get("object_attributes", {}),
            f"{field}.object_attributes",
        )
        object_custom = _mapping(
            mapping.get("object_custom_properties", {}),
            f"{field}.object_custom_properties",
        )
        return cls(
            fixture_id=_non_empty_string(mapping.get("id"), f"{field}.id"),
            source_blend=_relative_path(mapping.get("source_blend"), f"{field}.source_blend"),
            legacy_json=_relative_path(mapping.get("legacy_json"), f"{field}.legacy_json"),
            actual_json=_relative_path(mapping.get("actual_json"), f"{field}.actual_json"),
            operator=operator,
            active_object=active,
            selected_objects=selected,
            capabilities=_string_tuple(
                mapping.get("capabilities", ()),
                f"{field}.capabilities",
                allow_empty=False,
            ),
            image_pairs=image_pairs,
            scene_attributes=_mapping(
                mapping.get("scene_attributes", {}),
                f"{field}.scene_attributes",
            ),
            scene_custom_properties=_mapping(
                mapping.get("scene_custom_properties", {}),
                f"{field}.scene_custom_properties",
            ),
            object_attributes={
                _non_empty_string(name, f"{field}.object_attributes key"): _mapping(
                    attributes,
                    f"{field}.object_attributes[{name!r}]",
                )
                for name, attributes in object_attributes.items()
            },
            object_custom_properties={
                _non_empty_string(name, f"{field}.object_custom_properties key"): _mapping(
                    attributes,
                    f"{field}.object_custom_properties[{name!r}]",
                )
                for name, attributes in object_custom.items()
            },
            operator_kwargs=_mapping(
                mapping.get("operator_kwargs", {}),
                f"{field}.operator_kwargs",
            ),
            ignored_paths=_string_tuple(
                mapping.get("ignored_paths", ()),
                f"{field}.ignored_paths",
            ),
            accepted_warning_codes=_string_tuple(
                mapping.get("accepted_warning_codes", ()),
                f"{field}.accepted_warning_codes",
            ),
            strict_edges=bool(mapping.get("strict_edges", False)),
            compare_animations=bool(mapping.get("compare_animations", False)),
            absolute_tolerance=_finite_non_negative(
                mapping.get("absolute_tolerance", 1e-4),
                f"{field}.absolute_tolerance",
            ),
            relative_tolerance=_finite_non_negative(
                mapping.get("relative_tolerance", 1e-6),
                f"{field}.relative_tolerance",
            ),
            animated=bool(mapping.get("animated", False)),
        )


@dataclass(frozen=True, slots=True)
class PrivateReleaseGatePolicy:
    minimum_fixture_count: int
    required_capabilities: Tuple[str, ...]
    require_strict_edges: bool = True
    require_animation_parity_for_animated: bool = True
    allow_unaccepted_warnings: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivateReleaseGatePolicy":
        mapping = _mapping(value, "release_gate")
        minimum = mapping.get("minimum_fixture_count", 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            raise PrivateReleaseGateError(
                "release_gate.minimum_fixture_count must be a positive integer"
            )
        return cls(
            minimum_fixture_count=minimum,
            required_capabilities=_string_tuple(
                mapping.get("required_capabilities", ()),
                "release_gate.required_capabilities",
                allow_empty=False,
            ),
            require_strict_edges=bool(mapping.get("require_strict_edges", True)),
            require_animation_parity_for_animated=bool(
                mapping.get("require_animation_parity_for_animated", True)
            ),
            allow_unaccepted_warnings=bool(
                mapping.get("allow_unaccepted_warnings", False)
            ),
        )


@dataclass(frozen=True, slots=True)
class PrivateReleaseManifest:
    schema_version: int
    suite_id: str
    blender_version: str
    fixtures: Tuple[PrivateFixtureSpec, ...]
    release_gate: PrivateReleaseGatePolicy

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrivateReleaseManifest":
        mapping = _mapping(value, "manifest")
        version = mapping.get("schema_version")
        if version != 1:
            raise PrivateReleaseGateError(
                f"schema_version must be 1, got {version!r}"
            )
        fixture_values = mapping.get("fixtures")
        if not isinstance(fixture_values, (list, tuple)) or not fixture_values:
            raise PrivateReleaseGateError("fixtures must be a non-empty array")
        fixtures = tuple(
            PrivateFixtureSpec.from_mapping(item, index)
            for index, item in enumerate(fixture_values)
        )
        fixture_ids = tuple(item.fixture_id for item in fixtures)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise PrivateReleaseGateError("fixture ids must be unique")
        manifest = cls(
            schema_version=version,
            suite_id=_non_empty_string(mapping.get("suite_id"), "suite_id"),
            blender_version=_non_empty_string(
                mapping.get("blender_version"),
                "blender_version",
            ),
            fixtures=fixtures,
            release_gate=PrivateReleaseGatePolicy.from_mapping(
                mapping.get("release_gate", {})
            ),
        )
        manifest.validate_gate_coverage()
        return manifest

    def validate_gate_coverage(self) -> None:
        if len(self.fixtures) < self.release_gate.minimum_fixture_count:
            raise PrivateReleaseGateError(
                "fixture count is below release_gate.minimum_fixture_count; "
                f"actual={len(self.fixtures)}, required={self.release_gate.minimum_fixture_count}"
            )
        actual_capabilities = {
            capability
            for fixture in self.fixtures
            for capability in fixture.capabilities
        }
        missing = tuple(
            capability
            for capability in self.release_gate.required_capabilities
            if capability not in actual_capabilities
        )
        if missing:
            raise PrivateReleaseGateError(
                f"private fixture matrix is missing required capabilities: {missing}"
            )
        if self.release_gate.require_strict_edges:
            not_strict = tuple(
                fixture.fixture_id for fixture in self.fixtures if not fixture.strict_edges
            )
            if not_strict:
                raise PrivateReleaseGateError(
                    f"release gate requires strict edges for every fixture: {not_strict}"
                )
        if self.release_gate.require_animation_parity_for_animated:
            incomplete = tuple(
                fixture.fixture_id
                for fixture in self.fixtures
                if fixture.animated and not fixture.compare_animations
            )
            if incomplete:
                raise PrivateReleaseGateError(
                    "animated fixtures must compare animations: " + str(incomplete)
                )


def parse_private_release_manifest(value: Mapping[str, Any]) -> PrivateReleaseManifest:
    return PrivateReleaseManifest.from_mapping(value)


__all__ = [
    "PrivateFixtureSpec",
    "PrivateImagePairSpec",
    "PrivateReleaseGateError",
    "PrivateReleaseGatePolicy",
    "PrivateReleaseManifest",
    "parse_private_release_manifest",
]
