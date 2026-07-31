"""Canonical target-version registry for Spine JSON export."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Tuple


class SpineJsonTargetUnavailableError(ValueError):
    """Raised before geometry work when a selected target has no production codec."""


@dataclass(frozen=True, slots=True)
class SpineJsonVersionDescriptor:
    family: str
    exact_version: str
    label: str
    description: str
    uses_legacy_bone_transform_field: bool
    uses_legacy_constraint_mix_fields: bool
    uses_unified_constraints: bool
    supports_attachment_sequences: bool
    serializer_ready: bool
    supports_preview_animation: bool = False

    def __post_init__(self) -> None:
        for field_name in ("family", "exact_version", "label", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not self.exact_version.startswith(f"{self.family}."):
            raise ValueError(
                f"exact_version {self.exact_version!r} does not belong to {self.family!r}"
            )
        for field_name in (
            "uses_legacy_bone_transform_field",
            "uses_legacy_constraint_mix_fields",
            "uses_unified_constraints",
            "supports_attachment_sequences",
            "serializer_ready",
            "supports_preview_animation",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")


class SpineJsonTarget(str, Enum):
    SPINE_3_8 = "SPINE_3_8"
    SPINE_4_0 = "SPINE_4_0"
    SPINE_4_1 = "SPINE_4_1"
    SPINE_4_2 = "SPINE_4_2"
    SPINE_4_3 = "SPINE_4_3"

    @property
    def descriptor(self) -> SpineJsonVersionDescriptor:
        return _DESCRIPTORS[self]

    @property
    def family(self) -> str:
        return self.descriptor.family

    @property
    def exact_version(self) -> str:
        return self.descriptor.exact_version

    @property
    def label(self) -> str:
        return self.descriptor.label

    @property
    def description(self) -> str:
        return self.descriptor.description


_DESCRIPTORS: Mapping[SpineJsonTarget, SpineJsonVersionDescriptor] = MappingProxyType(
    {
        SpineJsonTarget.SPINE_3_8: SpineJsonVersionDescriptor(
            family="3.8",
            exact_version="3.8.99",
            label="Spine 3.8",
            description=(
                "Limited Spine 3.8.99 export: single-object and standalone "
                "multi-object for 2-Axis and 3-Axis rigs without texture sequences"
            ),
            uses_legacy_bone_transform_field=True,
            uses_legacy_constraint_mix_fields=True,
            uses_unified_constraints=False,
            supports_attachment_sequences=False,
            serializer_ready=True,
        ),
        SpineJsonTarget.SPINE_4_0: SpineJsonVersionDescriptor(
            family="4.0", exact_version="4.0.64", label="Spine 4.0",
            description=(
                "Limited Spine 4.0.64 export: 2-Axis single-object and standalone "
                "multi-object without texture sequences"
            ),
            uses_legacy_bone_transform_field=True,
            uses_legacy_constraint_mix_fields=False,
            uses_unified_constraints=False,
            supports_attachment_sequences=False,
            serializer_ready=True,
        ),
        SpineJsonTarget.SPINE_4_1: SpineJsonVersionDescriptor(
            family="4.1", exact_version="4.1.24", label="Spine 4.1",
            description=(
                "Limited Spine 4.1.24 export: 2-Axis single-object and standalone "
                "multi-object only; connected, mixed, and 3-Axis remain blocked"
            ),
            uses_legacy_bone_transform_field=True,
            uses_legacy_constraint_mix_fields=False,
            uses_unified_constraints=False,
            supports_attachment_sequences=True,
            serializer_ready=True,
        ),
        SpineJsonTarget.SPINE_4_2: SpineJsonVersionDescriptor(
            family="4.2", exact_version="4.2.43", label="Spine 4.2",
            description="Export setup-pose JSON for Spine 4.2.43",
            uses_legacy_bone_transform_field=False,
            uses_legacy_constraint_mix_fields=False,
            uses_unified_constraints=False,
            supports_attachment_sequences=True,
            serializer_ready=True,
        ),
        SpineJsonTarget.SPINE_4_3: SpineJsonVersionDescriptor(
            family="4.3", exact_version="4.3.23", label="Spine 4.3",
            description=(
                "Limited Spine 4.3.23 unified-constraint export: single-object and "
                "standalone multi-object for 2-Axis and 3-Axis rigs"
            ),
            uses_legacy_bone_transform_field=False,
            uses_legacy_constraint_mix_fields=False,
            uses_unified_constraints=True,
            supports_attachment_sequences=True,
            serializer_ready=True,
        ),
    }
)

DEFAULT_SPINE_JSON_TARGET = SpineJsonTarget.SPINE_4_2
DEFAULT_SPINE_JSON_VERSION = DEFAULT_SPINE_JSON_TARGET.exact_version
_TARGET_BY_EXACT_VERSION = MappingProxyType(
    {target.exact_version: target for target in SpineJsonTarget}
)
_TARGET_BY_FAMILY = MappingProxyType(
    {target.family: target for target in SpineJsonTarget}
)

if len(_TARGET_BY_EXACT_VERSION) != len(SpineJsonTarget):
    raise RuntimeError("Spine JSON exact versions must be unique")
if len(_TARGET_BY_FAMILY) != len(SpineJsonTarget):
    raise RuntimeError("Spine JSON families must be unique")


def spine_json_target_enum_items() -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        (target.value, target.label, target.description)
        for target in SpineJsonTarget
    )


def resolve_spine_json_target(value: object) -> SpineJsonTarget:
    if isinstance(value, SpineJsonTarget):
        return value
    if not isinstance(value, str):
        raise TypeError("Spine JSON target must be SpineJsonTarget or str")
    normalized = value.strip()
    if not normalized:
        raise ValueError("Spine JSON target cannot be empty")
    try:
        return SpineJsonTarget(normalized)
    except ValueError:
        pass
    target = _TARGET_BY_EXACT_VERSION.get(normalized)
    if target is None:
        target = _TARGET_BY_FAMILY.get(normalized)
    if target is not None:
        return target
    supported = tuple(target.value for target in SpineJsonTarget)
    raise ValueError(
        f"Unsupported Spine JSON target {value!r}; supported identifiers={supported}"
    )


def resolve_spine_json_exact_version(value: object) -> SpineJsonTarget:
    if not isinstance(value, str):
        raise TypeError("Spine JSON exact version must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError("Spine JSON exact version cannot be empty")
    target = _TARGET_BY_EXACT_VERSION.get(normalized)
    if target is None:
        supported = tuple(item.exact_version for item in SpineJsonTarget)
        raise ValueError(
            f"Unsupported Spine JSON exact version {value!r}; supported={supported}"
        )
    return target


def spine_json_version_filename_token(value: object) -> str:
    target = resolve_spine_json_target(value)
    return f"spine_{target.exact_version}"


def require_spine_json_target_serializable(value: object) -> SpineJsonTarget:
    target = resolve_spine_json_target(value)
    if not target.descriptor.serializer_ready:
        raise SpineJsonTargetUnavailableError(
            f"Spine JSON target {target.label} ({target.exact_version}) is selectable "
            "for implementation testing but its production serializer is not ready"
        )
    return target


__all__ = [
    "DEFAULT_SPINE_JSON_TARGET",
    "DEFAULT_SPINE_JSON_VERSION",
    "SpineJsonTarget",
    "SpineJsonTargetUnavailableError",
    "SpineJsonVersionDescriptor",
    "require_spine_json_target_serializable",
    "resolve_spine_json_exact_version",
    "resolve_spine_json_target",
    "spine_json_target_enum_items",
    "spine_json_version_filename_token",
]
