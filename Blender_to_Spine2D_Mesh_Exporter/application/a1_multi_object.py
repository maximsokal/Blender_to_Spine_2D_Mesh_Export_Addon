"""Blender-independent settings for one A1 multi-object output transaction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from ..domain.baking import A1TextureExportMode, sanitize_filename_stem
from ..domain.spine import UniformScaleMode
from ..domain.spine.export_capabilities import (
    SpineJsonExportScope,
    require_spine_json_export_capability,
)
from ..domain.spine.rig_profiles import A1RigSetupPoseMode
from .a1_numeric_contracts import (
    require_finite_number,
    require_identity,
    require_integer,
    require_non_empty_string,
)
from .a1_single_object import A1SingleObjectExportSettings


class A1MultiObjectMode(str, Enum):
    STANDALONE = "STANDALONE"
    CONNECTED = "CONNECTED"
    MIXED = "MIXED"


class ConnectedCameraRenderPolicy(str, Enum):
    """Choose interactive rig layers or an explicit static camera flattening."""

    # Preserve one weighted mesh per object so global and object-specific controls
    # continue to deform every visible layer through the generated vertex-bone rig.
    INDIVIDUAL_LAYERS = "INDIVIDUAL_LAYERS"
    # Static flattening is depth-correct for the captured camera render, but the
    # resulting root-bound overlay cannot reproduce independent object controls.
    AUTO_GROUPED_CAMERA = "AUTO_GROUPED_CAMERA"
    GROUPED_CAMERA_REQUIRED = "GROUPED_CAMERA_REQUIRED"


class A1MultiObjectStage(str, Enum):
    VALIDATE_REQUEST = "VALIDATE_REQUEST"
    PREPARE_OBJECTS = "PREPARE_OBJECTS"
    VALIDATE_OUTPUTS = "VALIDATE_OUTPUTS"
    COMPOSE_DOCUMENT = "COMPOSE_DOCUMENT"
    SERIALIZE_DOCUMENT = "SERIALIZE_DOCUMENT"
    STAGE_OUTPUTS = "STAGE_OUTPUTS"
    COMMIT_OUTPUTS = "COMMIT_OUTPUTS"

    @property
    def error_code(self) -> str:
        return f"A1_MULTI_{self.value}_FAILED"


@dataclass(frozen=True, slots=True)
class A1MultiObjectExportSettings:
    output_directory: Path
    output_stem: str
    mode: A1MultiObjectMode = A1MultiObjectMode.STANDALONE
    json_indent: int = 2
    namespace_animations: bool = True
    animation_separator: str = "/"
    connected_group_prefix: str = "all_objects"
    anchor_component_id: str | None = None
    z_tolerance: float = 1e-4
    connected_scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE
    connected_camera_render_policy: ConnectedCameraRenderPolicy = (
        ConnectedCameraRenderPolicy.INDIVIDUAL_LAYERS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        require_non_empty_string(self.output_stem, "output_stem")
        sanitize_filename_stem(self.output_stem)
        if not isinstance(self.mode, A1MultiObjectMode):
            raise TypeError("mode must be A1MultiObjectMode")
        require_integer(self.json_indent, "json_indent", minimum=0, maximum=16)
        if not isinstance(self.namespace_animations, bool):
            raise TypeError("namespace_animations must be bool")
        require_non_empty_string(self.animation_separator, "animation_separator")
        require_identity(self.connected_group_prefix, "connected_group_prefix")
        if self.anchor_component_id is not None:
            require_identity(self.anchor_component_id, "anchor_component_id")
        require_finite_number(
            self.z_tolerance,
            "z_tolerance",
            minimum=0.0,
        )
        if not isinstance(self.connected_scale_mode, UniformScaleMode):
            raise TypeError("connected_scale_mode must be UniformScaleMode")
        if not isinstance(
            self.connected_camera_render_policy,
            ConnectedCameraRenderPolicy,
        ):
            raise TypeError(
                "connected_camera_render_policy must be ConnectedCameraRenderPolicy"
            )

    @property
    def resolved_output_stem(self) -> str:
        return sanitize_filename_stem(self.output_stem)

    @property
    def json_path(self) -> Path:
        root = self.output_directory.expanduser().resolve(strict=False)
        return root / f"{self.resolved_output_stem}.json"


def _export_scope_for_multi_object_mode(
    mode: A1MultiObjectMode,
) -> SpineJsonExportScope:
    """Map one preparation mode to the matching target capability scope."""

    if not isinstance(mode, A1MultiObjectMode):
        raise TypeError("mode must be A1MultiObjectMode")
    if mode is A1MultiObjectMode.STANDALONE:
        return SpineJsonExportScope.STANDALONE_MULTI_OBJECT
    if mode is A1MultiObjectMode.CONNECTED:
        return SpineJsonExportScope.CONNECTED_MULTI_OBJECT
    raise ValueError(
        "MIXED mode must be resolved to CONNECTED or STANDALONE before preparation"
    )


def _standalone_projected_object_root_settings(
    settings: A1SingleObjectExportSettings,
) -> A1SingleObjectExportSettings:
    """Neutralize legacy setup calibration for already-projected standalone geometry.

    Signed-axis Normal / UV source preparation has already transformed local geometry and
    Blender Object Origin into canonical U/V/depth space. Applying the historical
    ``PRESERVE_COMPOSITION`` setup rotations after that projection rotates the object a
    second time. Different depth layouts then acquire different setup translations, which
    makes separate object blocks appear to drift apart after standalone composition.

    ``CAMERA_VIEW_NORMAL`` is the established projected Object-Root setup contract: it
    keeps the projected ``main`` position, neutralizes setup-only X/Y/depth calibration,
    and parents vertex bones through inverse depth-setup children while retaining live
    deformation controls. Its stable identifier predates signed-axis projection support,
    but the mechanics are projection-generic.

    Explicit non-default setup modes are respected. Camera Projection and Depth Camera
    Projection do not enter this policy because they are not Normal / UV object-bake
    textures, and Active Camera keeps its dedicated routing in document preparation.
    """

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    if (
        settings.bake_execution.texture_export_mode
        is not A1TextureExportMode.NORMAL_UV_SEGMENTS
    ):
        return settings
    if not settings.projection_direction.axis_aligned:
        return settings
    if settings.rig_setup_pose_mode is not A1RigSetupPoseMode.PRESERVE_COMPOSITION:
        return settings
    return replace(
        settings,
        rig_setup_pose_mode=A1RigSetupPoseMode.CAMERA_VIEW_NORMAL,
    )


def resolve_a1_multi_object_preparation_settings(
    settings: A1SingleObjectExportSettings,
    mode: A1MultiObjectMode,
) -> A1SingleObjectExportSettings:
    """Return the exact per-object settings used by multi-object preparation.

    The target capability is checked before geometry work. Connected documents omit each
    object's absolute projected translation; connected composition adds anchor-relative
    projected translation later. Standalone Normal / UV signed-axis documents preserve
    absolute projected Object Origins but use the neutral projected Object-Root setup so
    already-projected geometry is not transformed a second time in Spine Setup Pose.
    MIXED must be resolved into explicit connected and standalone subgroups first.

    Active Camera Object Root continues to select its neutral setup in document
    preparation because camera projection kind is resolved there. Rendered Camera
    Projection and Depth Camera Projection retain their independent setup contracts.
    """

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    scope = _export_scope_for_multi_object_mode(mode)
    require_spine_json_export_capability(
        settings.export.spine_target,
        settings.export.rig_profile,
        scope,
    )
    if mode is A1MultiObjectMode.STANDALONE:
        return _standalone_projected_object_root_settings(settings)
    if mode is A1MultiObjectMode.CONNECTED:
        if not settings.use_world_location_for_main_bone:
            return settings
        return replace(settings, use_world_location_for_main_bone=False)
    raise ValueError(f"Unsupported multi-object preparation mode: {mode!r}")


__all__ = [
    "A1MultiObjectExportSettings",
    "A1MultiObjectMode",
    "A1MultiObjectStage",
    "ConnectedCameraRenderPolicy",
    "resolve_a1_multi_object_preparation_settings",
]
