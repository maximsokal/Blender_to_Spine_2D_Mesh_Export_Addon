"""Build the A1 rotatable-mesh hierarchy without Blender or JSON dictionaries.

The implementation is a typed translation of the stable parts of the current
``create_bones`` and ``build_constraints`` pipeline.  It preserves public names,
parent order, constraint order, colors, and transform parameters while removing
scene access, introspection, and mutable ``bones_info`` dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Tuple

from .legacy_profile import LegacyRigProfile
from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .validator import SpineValidator


class UniformScaleMode(str, Enum):
    """Legacy texture-dimension strategies used to size the control rig."""

    AVERAGE = "AVERAGE"
    MAXIMUM = "MAXIMUM"
    MINIMUM = "MINIMUM"


@dataclass(frozen=True, slots=True)
class LegacyZGroup:
    """One ordered depth group used to create a scale/rotation bone pair."""

    z_value: float
    height_real_pixels: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.z_value, (int, float)) or not isfinite(
            float(self.z_value)
        ):
            raise ValueError("z_value must be finite")
        if self.height_real_pixels is not None and (
            not isinstance(self.height_real_pixels, (int, float))
            or not isfinite(float(self.height_real_pixels))
        ):
            raise ValueError("height_real_pixels must be finite or None")


@dataclass(frozen=True, slots=True)
class LegacyRigBuildRequest:
    """All data required to build one legacy rig in Spine pixel coordinates."""

    prefix: str
    texture_width: int
    texture_height: int
    z_groups: Tuple[LegacyZGroup, ...]
    average_y_pixels: float = 0.0
    main_position_pixels: Tuple[float, float] | None = None
    scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or not self.prefix.strip():
            raise ValueError("prefix must be a non-empty string")
        for field_name in ("texture_width", "texture_height"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.z_groups, tuple) or not self.z_groups:
            raise ValueError("z_groups must be a non-empty tuple")
        if not all(isinstance(group, LegacyZGroup) for group in self.z_groups):
            raise TypeError("z_groups must contain LegacyZGroup values")
        normalized_z = tuple(float(group.z_value) for group in self.z_groups)
        if len(normalized_z) != len(set(normalized_z)):
            raise ValueError("z_groups cannot contain duplicate z_value entries")
        if not isinstance(self.average_y_pixels, (int, float)) or not isfinite(
            float(self.average_y_pixels)
        ):
            raise ValueError("average_y_pixels must be finite")
        if self.main_position_pixels is not None:
            if (
                not isinstance(self.main_position_pixels, tuple)
                or len(self.main_position_pixels) != 2
                or not all(
                    isinstance(value, (int, float)) and isfinite(float(value))
                    for value in self.main_position_pixels
                )
            ):
                raise ValueError(
                    "main_position_pixels must contain two finite numeric values"
                )
        if not isinstance(self.scale_mode, UniformScaleMode):
            raise TypeError("scale_mode must be UniformScaleMode")


@dataclass(frozen=True, slots=True)
class LegacyZGroupBuildInfo:
    z_value: float
    index: int
    y_offset_pixels: float
    calculation_method: str
    scale_bone_name: str
    bone_name: str


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

    def bone_for_z(self, z_value: float, *, tolerance: float = 1e-9) -> str:
        if not isinstance(z_value, (int, float)) or not isfinite(float(z_value)):
            raise ValueError("z_value must be finite")
        if not isinstance(tolerance, (int, float)) or not isfinite(float(tolerance)):
            raise ValueError("tolerance must be finite")
        if tolerance < 0.0:
            raise ValueError("tolerance cannot be negative")
        matches = tuple(
            group.bone_name
            for group in self.z_groups
            if abs(group.z_value - float(z_value)) <= tolerance
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

    def validate(self) -> None:
        """Validate parent order and all constraint references through Spine rules."""

        document = SpineDocument(
            skeleton={"spine": self.profile.spine_version},
            bones=self.bones,
            slots=(),
            skins=(),
            ik=self.ik,
            transform=self.transform,
        )
        SpineValidator().validate_or_raise(document)


class LegacyRigBuildError(ValueError):
    """Raised when the A1 hierarchy cannot be constructed consistently."""


def calculate_uniform_scale(
    texture_width: int,
    texture_height: int,
    mode: UniformScaleMode = UniformScaleMode.AVERAGE,
) -> float:
    """Return the exact legacy texture-size scale used by the current exporter."""

    for field_name, value in (
        ("texture_width", texture_width),
        ("texture_height", texture_height),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    if not isinstance(mode, UniformScaleMode):
        raise TypeError("mode must be UniformScaleMode")
    if mode is UniformScaleMode.AVERAGE:
        return (float(texture_width) + float(texture_height)) / 2.0
    if mode is UniformScaleMode.MAXIMUM:
        return float(max(texture_width, texture_height))
    return float(min(texture_width, texture_height))


def _main_position(request: LegacyRigBuildRequest) -> tuple[float, float]:
    if request.main_position_pixels is not None:
        x_value, y_value = request.main_position_pixels
        return round(float(x_value), 2), round(float(y_value), 2)
    return 0.0, round(float(request.average_y_pixels), 2)


def _build_z_group_bones(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile,
    *,
    parent_bone_name: str,
    uniform_scale: float,
    half_scale: float,
) -> tuple[Tuple[Bone, ...], Tuple[LegacyZGroupBuildInfo, ...]]:
    ordered_groups = tuple(sorted(request.z_groups, key=lambda group: group.z_value))
    minimum_z = float(ordered_groups[0].z_value)
    bones: list[Bone] = []
    group_info: list[LegacyZGroupBuildInfo] = []

    for offset, group in enumerate(ordered_groups):
        index = profile.z_index_base + offset
        scale_bone_name = profile.z_scale_bone(request.prefix, index)
        bone_name = profile.z_bone(request.prefix, index)
        if group.height_real_pixels is not None:
            y_offset = float(group.height_real_pixels)
            calculation_method = "height_real_pixels"
        else:
            y_offset = (float(group.z_value) - minimum_z) * uniform_scale
            calculation_method = "direct_3d_scaling"
        rounded_y = round(y_offset, 2)

        bones.extend(
            (
                Bone(
                    name=scale_bone_name,
                    parent=parent_bone_name,
                    length=half_scale,
                    rotation=90.0,
                    y=rounded_y,
                    color="abe323ff",
                    extras={"inherit": "onlyTranslation"},
                ),
                Bone(
                    name=bone_name,
                    parent=scale_bone_name,
                    rotation=-90.0,
                ),
            )
        )
        group_info.append(
            LegacyZGroupBuildInfo(
                z_value=float(group.z_value),
                index=index,
                y_offset_pixels=rounded_y,
                calculation_method=calculation_method,
                scale_bone_name=scale_bone_name,
                bone_name=bone_name,
            )
        )

    return tuple(bones), tuple(group_info)


def _build_constraints(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile,
    info: LegacyRigInfo,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    prefix = request.prefix
    control_x, control_y, control_z = info.control_bone_names
    constraint_bone, _, constraint_rotate_ik, constraint_ik = (
        info.ik_chain_bone_names
    )

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(prefix),
            order=3,
            bones=(constraint_bone,),
            target=constraint_ik,
            extras={"compress": True, "stretch": True},
        ),
    )

    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(prefix),
            order=1,
            bones=info.sub_bone_scale_names + (info.base_bone_name,),
            target=control_x,
            extras={
                "rotation": 90,
                "local": True,
                "relative": True,
                "x": -(info.uniform_scale * 2.0),
                "y": -info.half_scale,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(prefix),
            order=2,
            bones=(info.main_rotation_bone_name, constraint_rotate_ik),
            target=control_y,
            extras={
                "local": True,
                "relative": True,
                "x": info.uniform_scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_z_constraint(prefix),
            order=5,
            bones=info.sub_bone_names,
            target=control_z,
            extras={
                "local": True,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_constraint(prefix),
            order=4,
            bones=info.sub_bone_scale_names,
            target=constraint_bone,
            extras={
                "scaleX": -1,
                "mixRotate": 0,
                "mixX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_compensator_constraint(prefix),
            order=6,
            bones=tuple(reversed(info.sub_bone_scale_names)),
            target=info.base_bone_name,
            extras={
                "mixRotate": 0,
                "mixX": 0,
                "mixScaleX": 0,
                "mixScaleY": 0,
                "mixShearY": 0,
            },
        ),
    )
    return ik, transform


def build_legacy_rig(
    request: LegacyRigBuildRequest,
    profile: LegacyRigProfile | None = None,
) -> LegacyRigBuildResult:
    """Build and validate the complete ordered A1 control hierarchy."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    resolved_profile = profile or LegacyRigProfile()
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    prefix = request.prefix.strip()
    uniform_scale = calculate_uniform_scale(
        request.texture_width,
        request.texture_height,
        request.scale_mode,
    )
    half_scale = uniform_scale / 2.0
    main_x, main_y = _main_position(request)

    root_name = resolved_profile.root_bone()
    main_name = resolved_profile.main_bone(prefix)
    base_name = resolved_profile.base_bone(prefix)
    scale_name = resolved_profile.scale_rotate_x_bone(prefix)
    rotate_name = resolved_profile.rotate_x_bone(prefix)
    control_names = resolved_profile.control_bones(prefix)
    ik_chain_names = resolved_profile.ik_chain_bones(prefix)

    bones: list[Bone] = [
        Bone(name=root_name),
        Bone(
            name=main_name,
            parent=root_name,
            x=main_x,
            y=main_y,
        ),
        Bone(name=base_name, parent=main_name),
        Bone(
            name=scale_name,
            parent=base_name,
            length=half_scale,
            y=-0.5,
            scale_x=0.0,
        ),
        Bone(
            name=rotate_name,
            parent=scale_name,
            color="ff0000ff",
        ),
    ]

    z_bones, z_info = _build_z_group_bones(
        request,
        resolved_profile,
        parent_bone_name=rotate_name,
        uniform_scale=uniform_scale,
        half_scale=half_scale,
    )
    bones.extend(z_bones)

    control_x, control_y, control_z = control_names
    bones.extend(
        (
            Bone(
                name=control_x,
                parent=main_name,
                length=half_scale,
                x=uniform_scale,
                y=half_scale,
                color="ff0000ff",
            ),
            Bone(
                name=control_y,
                parent=main_name,
                length=half_scale,
                x=uniform_scale,
                color="00ff18ff",
            ),
            Bone(
                name=control_z,
                parent=main_name,
                length=half_scale,
                x=uniform_scale,
                y=-half_scale,
                color="002cffff",
            ),
        )
    )

    constraint_bone, constraint_scale_ik, constraint_rotate_ik, constraint_ik = (
        ik_chain_names
    )
    bones.extend(
        (
            Bone(
                name=constraint_bone,
                parent=base_name,
                length=half_scale,
                rotation=90.0,
                y=-0.5,
                color="abe323ff",
            ),
            Bone(
                name=constraint_scale_ik,
                parent=base_name,
                y=half_scale - 0.5,
                scale_x=0.0,
            ),
            Bone(
                name=constraint_rotate_ik,
                parent=constraint_scale_ik,
                x=-half_scale,
            ),
            Bone(
                name=constraint_ik,
                parent=constraint_rotate_ik,
                rotation=90.0,
                x=half_scale,
                color="ff3f00ff",
                icon="ik",
            ),
        )
    )

    info = LegacyRigInfo(
        profile_id=resolved_profile.profile_id,
        prefix=prefix,
        uniform_scale=uniform_scale,
        half_scale=half_scale,
        root_bone_name=root_name,
        main_bone_name=main_name,
        base_bone_name=base_name,
        scale_bone_name=scale_name,
        main_rotation_bone_name=rotate_name,
        control_bone_names=control_names,
        ik_chain_bone_names=ik_chain_names,
        z_groups=z_info,
        sub_bone_scale_names=tuple(group.scale_bone_name for group in z_info),
        sub_bone_names=tuple(group.bone_name for group in z_info),
    )
    ik, transform = _build_constraints(request, resolved_profile, info)

    result = LegacyRigBuildResult(
        request=request,
        profile=resolved_profile,
        bones=tuple(bones),
        ik=ik,
        transform=transform,
        info=info,
    )
    try:
        result.validate()
    except Exception as exc:
        raise LegacyRigBuildError(
            f"Generated A1 rig for '{prefix}' failed validation: {exc}"
        ) from exc
    return result
