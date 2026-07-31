"""Immutable contracts for the legacy-compatible A1 control rigs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Tuple

from .legacy_profile import LegacyRigProfile
from .model import Bone, IKConstraint, TransformConstraint
from .rig_profiles import A1RigSetupPoseMode


def _require_canonical_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain leading or trailing whitespace")
    return value


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _require_non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


class UniformScaleMode(str, Enum):
    """Legacy texture-dimension strategies used to size the control rig."""

    AVERAGE = "AVERAGE"
    MAXIMUM = "MAXIMUM"
    MINIMUM = "MINIMUM"


class LegacyZGroupOriginMode(str, Enum):
    """Choose the source Z reference used for generated depth-bone offsets.

    ``MINIMUM_Z`` preserves the historical compatibility behavior where the lowest
    source layer becomes offset zero. ``OBJECT_ORIGIN`` keeps Blender local Z=0 as the
    zero depth plane, so negative and positive groups remain on their authored sides of
    the Object Origin.
    """

    MINIMUM_Z = "MINIMUM_Z"
    OBJECT_ORIGIN = "OBJECT_ORIGIN"


@dataclass(frozen=True, slots=True)
class LegacyZGroup:
    """One ordered depth group used to create a scale/rotation bone pair."""

    z_value: float
    height_real_pixels: float | None = None

    def __post_init__(self) -> None:
        _require_finite_number(self.z_value, "z_value")
        if self.height_real_pixels is not None:
            _require_finite_number(self.height_real_pixels, "height_real_pixels")


@dataclass(frozen=True, slots=True)
class LegacyRigBuildRequest:
    """All data required to build one selectable rig in Spine pixel coordinates."""

    prefix: str
    texture_width: int
    texture_height: int
    z_groups: Tuple[LegacyZGroup, ...]
    average_y_pixels: float = 0.0
    main_position_pixels: Tuple[float, float] | None = None
    scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE
    # Appended so historical positional construction remains stable.
    setup_pose_mode: A1RigSetupPoseMode = A1RigSetupPoseMode.PRESERVE_COMPOSITION
    # Appended so historical positional construction remains stable. Public Normal /
    # UV Segments two-axis export opts into OBJECT_ORIGIN explicitly; every legacy and
    # camera path keeps MINIMUM_Z unless its route owner selects otherwise.
    z_group_origin_mode: LegacyZGroupOriginMode = LegacyZGroupOriginMode.MINIMUM_Z

    def __post_init__(self) -> None:
        _require_canonical_string(self.prefix, "prefix")
        for field_name in ("texture_width", "texture_height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.z_groups, tuple) or not self.z_groups:
            raise ValueError("z_groups must be a non-empty tuple")
        if not all(isinstance(group, LegacyZGroup) for group in self.z_groups):
            raise TypeError("z_groups must contain LegacyZGroup values")
        normalized_z = tuple(float(group.z_value) for group in self.z_groups)
        if len(normalized_z) != len(set(normalized_z)):
            raise ValueError("z_groups cannot contain duplicate z_value entries")
        _require_finite_number(self.average_y_pixels, "average_y_pixels")
        if self.main_position_pixels is not None:
            if (
                not isinstance(self.main_position_pixels, tuple)
                or len(self.main_position_pixels) != 2
            ):
                raise ValueError(
                    "main_position_pixels must contain two finite numeric values"
                )
            for index, value in enumerate(self.main_position_pixels):
                _require_finite_number(value, f"main_position_pixels[{index}]")
        if not isinstance(self.scale_mode, UniformScaleMode):
            raise TypeError("scale_mode must be UniformScaleMode")
        if not isinstance(self.setup_pose_mode, A1RigSetupPoseMode):
            raise TypeError("setup_pose_mode must be A1RigSetupPoseMode")
        if not isinstance(self.z_group_origin_mode, LegacyZGroupOriginMode):
            raise TypeError(
                "z_group_origin_mode must be LegacyZGroupOriginMode"
            )


@dataclass(frozen=True, slots=True)
class LegacyZGroupBuildInfo:
    z_value: float
    index: int
    y_offset_pixels: float
    calculation_method: str
    scale_bone_name: str
    bone_name: str

    def __post_init__(self) -> None:
        _require_finite_number(self.z_value, "z_value")
        _require_non_negative_integer(self.index, "index")
        _require_finite_number(self.y_offset_pixels, "y_offset_pixels")
        if self.calculation_method not in {
            "height_real_pixels",
            "direct_3d_scaling",
        }:
            raise ValueError(
                "calculation_method must be height_real_pixels or direct_3d_scaling"
            )
        _require_canonical_string(self.scale_bone_name, "scale_bone_name")
        _require_canonical_string(self.bone_name, "bone_name")


@dataclass(frozen=True, slots=True)
class LegacyRigInfo:
    profile_id: str
    prefix: str
    uniform_scale: float
    half_scale: float
    root_bone_name: str
    main_bone_name: str
    base_bone_name: str
    scale_bone_name: str
    main_rotation_bone_name: str
    control_bone_names: Tuple[str, str, str]
    ik_chain_bone_names: Tuple[str, str, str, str]
    z_groups: Tuple[LegacyZGroupBuildInfo, ...]
    sub_bone_scale_names: Tuple[str, ...]
    sub_bone_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "profile_id",
            "prefix",
            "root_bone_name",
            "main_bone_name",
            "base_bone_name",
            "scale_bone_name",
            "main_rotation_bone_name",
        ):
            _require_canonical_string(getattr(self, field_name), field_name)
        uniform_scale = _require_finite_number(self.uniform_scale, "uniform_scale")
        half_scale = _require_finite_number(self.half_scale, "half_scale")
        if uniform_scale <= 0.0 or half_scale <= 0.0:
            raise ValueError("uniform_scale and half_scale must be positive")
        if (
            not isinstance(self.control_bone_names, tuple)
            or len(self.control_bone_names) != 3
        ):
            raise ValueError("control_bone_names must contain exactly three names")
        if (
            not isinstance(self.ik_chain_bone_names, tuple)
            or len(self.ik_chain_bone_names) != 4
        ):
            raise ValueError("ik_chain_bone_names must contain exactly four names")
        for field_name in (
            "control_bone_names",
            "ik_chain_bone_names",
            "sub_bone_scale_names",
            "sub_bone_names",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be tuple")
            for index, value in enumerate(values):
                _require_canonical_string(value, f"{field_name}[{index}]")
        if not isinstance(self.z_groups, tuple) or not self.z_groups:
            raise ValueError("z_groups must be a non-empty tuple")
        if not all(isinstance(item, LegacyZGroupBuildInfo) for item in self.z_groups):
            raise TypeError("z_groups must contain LegacyZGroupBuildInfo values")
        if len(self.sub_bone_scale_names) != len(self.z_groups):
            raise ValueError("sub_bone_scale_names must match z_groups")
        if len(self.sub_bone_names) != len(self.z_groups):
            raise ValueError("sub_bone_names must match z_groups")

    def bone_for_z(self, z_value: float, *, tolerance: float = 1e-9) -> str:
        resolved_z = _require_finite_number(z_value, "z_value")
        resolved_tolerance = _require_finite_number(tolerance, "tolerance")
        if resolved_tolerance < 0.0:
            raise ValueError("tolerance cannot be negative")
        matches = tuple(
            group.bone_name
            for group in self.z_groups
            if abs(float(group.z_value) - resolved_z) <= resolved_tolerance
        )
        if len(matches) != 1:
            raise KeyError(
                f"Expected one z-group bone for {z_value}, found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class LegacyRigBuildResult:
    request: LegacyRigBuildRequest
    profile: LegacyRigProfile
    bones: Tuple[Bone, ...]
    ik: Tuple[IKConstraint, ...]
    transform: Tuple[TransformConstraint, ...]
    info: LegacyRigInfo

    def __post_init__(self) -> None:
        if not isinstance(self.request, LegacyRigBuildRequest):
            raise TypeError("request must be LegacyRigBuildRequest")
        if not isinstance(self.profile, LegacyRigProfile):
            raise TypeError("profile must be LegacyRigProfile")
        if not isinstance(self.bones, tuple) or not self.bones:
            raise ValueError("bones must be a non-empty tuple")
        if not all(isinstance(item, Bone) for item in self.bones):
            raise TypeError("bones must contain Bone values")
        if not isinstance(self.ik, tuple) or not all(
            isinstance(item, IKConstraint) for item in self.ik
        ):
            raise TypeError("ik must be a tuple of IKConstraint values")
        if not isinstance(self.transform, tuple) or not all(
            isinstance(item, TransformConstraint) for item in self.transform
        ):
            raise TypeError("transform must be a tuple of TransformConstraint values")
        if not isinstance(self.info, LegacyRigInfo):
            raise TypeError("info must be LegacyRigInfo")

    def validate(self) -> None:
        """Validate exact profile semantics and generic Spine cross-references."""

        from .rig_profiles import A1RigProfile, resolve_a1_rig_profile

        profile = resolve_a1_rig_profile(self.profile.profile_id)
        if profile is A1RigProfile.THREE_AXIS_ROTATION:
            from .legacy_rig_validation import validate_legacy_rig_result

            validate_legacy_rig_result(self)
            return
        if profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
            from .two_axis_scale_rig import validate_two_axis_scale_rig_result

            validate_two_axis_scale_rig_result(self)
            return
        raise AssertionError(f"Unhandled rig profile: {profile}")


__all__ = [
    "LegacyRigBuildRequest",
    "LegacyRigBuildResult",
    "LegacyRigInfo",
    "LegacyZGroup",
    "LegacyZGroupBuildInfo",
    "LegacyZGroupOriginMode",
    "UniformScaleMode",
]
