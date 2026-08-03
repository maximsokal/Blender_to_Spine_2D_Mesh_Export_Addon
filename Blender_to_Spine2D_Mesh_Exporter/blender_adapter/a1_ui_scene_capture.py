"""Capture one immutable Rewrite export profile from Blender 5.2 Scene RNA."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import isfinite
from pathlib import Path
from typing import Any

from ..application import A1GeometryPreparationSettings
from ..domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
    CameraProjectionInfluencePolicy,
    DepthCameraProjectionSettings,
    DepthProjectionBaseMode,
    TextureSequenceTiming,
)
from ..domain.baking.generated_materials import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    ColorRGBA,
)
from ..domain.geometry import A1AngularMode
from ..domain.projection import (
    A1ProjectionDirection,
    resolve_a1_projection_direction,
)
from ..domain.spine.rig_profiles import A1RigProfile, resolve_a1_rig_profile
from ..domain.spine.version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    SpineJsonTarget,
    resolve_spine_json_target,
)


logger = logging.getLogger(__name__)
_DEFAULT_PROJECTION_ALPHA_THRESHOLD = 1.0 / 255.0
_DEFAULT_GENERATED_GRAY: ColorRGBA = (0.5, 0.5, 0.5, 1.0)
_PREVIEW_ANIMATION_EXPORT_ENABLED = False
_PUBLIC_RIG_PROFILE = A1RigProfile.TWO_AXIS_ROTATION_SCALE


@dataclass(frozen=True, slots=True)
class _SceneExportProfile:
    """One immutable snapshot of all Scene-level Rewrite export settings."""

    output_directory: Path
    images_relative_path: str
    texture_size: int
    seam_mode: str
    angle_limit_degrees: float
    geometry: A1GeometryPreparationSettings
    bake_execution: BakeExecutionSettings
    include_control_icons: bool
    include_preview_animation: bool
    spine_target: SpineJsonTarget
    rig_profile: A1RigProfile
    material_source_policy: A1MaterialSourcePolicy = (
        A1MaterialSourcePolicy.REQUIRE_SOURCE
    )
    generated_material_pattern: A1GeneratedMaterialPattern = (
        A1GeneratedMaterialPattern.SOLID_GRAY
    )
    generated_gray_color: ColorRGBA = _DEFAULT_GENERATED_GRAY
    sequence_timing: TextureSequenceTiming = TextureSequenceTiming()
    texture_export_mode: A1TextureExportMode = (
        A1TextureExportMode.NORMAL_UV_SEGMENTS
    )
    projection_direction: A1ProjectionDirection = (
        A1ProjectionDirection.POSITIVE_Z
    )

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        if (
            not isinstance(self.images_relative_path, str)
            or not self.images_relative_path
        ):
            raise ValueError("images_relative_path must be a non-empty string")
        if (
            not isinstance(self.texture_size, int)
            or isinstance(self.texture_size, bool)
            or self.texture_size < 64
            or self.texture_size > 4096
            or self.texture_size % 2
        ):
            raise ValueError(
                "texture_size must be an even integer between 64 and 4096"
            )
        if self.seam_mode not in {"AUTO", "CUSTOM"}:
            raise ValueError("seam_mode must be AUTO or CUSTOM")
        if isinstance(self.angle_limit_degrees, bool) or not isinstance(
            self.angle_limit_degrees,
            (int, float),
        ):
            raise TypeError("angle_limit_degrees must be numeric")
        if not isfinite(float(self.angle_limit_degrees)):
            raise ValueError("angle_limit_degrees must be finite")
        if not isinstance(self.geometry, A1GeometryPreparationSettings):
            raise TypeError("geometry must be A1GeometryPreparationSettings")
        if not isinstance(self.bake_execution, BakeExecutionSettings):
            raise TypeError("bake_execution must be BakeExecutionSettings")
        if not isinstance(self.include_control_icons, bool):
            raise TypeError("include_control_icons must be bool")
        if not isinstance(self.include_preview_animation, bool):
            raise TypeError("include_preview_animation must be bool")
        if not isinstance(self.spine_target, SpineJsonTarget):
            raise TypeError("spine_target must be SpineJsonTarget")
        if not isinstance(self.rig_profile, A1RigProfile):
            raise TypeError("rig_profile must be A1RigProfile")
        if not isinstance(
            self.material_source_policy,
            A1MaterialSourcePolicy,
        ):
            raise TypeError(
                "material_source_policy must be A1MaterialSourcePolicy"
            )
        if not isinstance(
            self.generated_material_pattern,
            A1GeneratedMaterialPattern,
        ):
            raise TypeError(
                "generated_material_pattern must be A1GeneratedMaterialPattern"
            )
        if (
            not isinstance(self.generated_gray_color, tuple)
            or len(self.generated_gray_color) != 4
        ):
            raise ValueError("generated_gray_color must contain four values")
        for index, value in enumerate(self.generated_gray_color):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"generated_gray_color[{index}] must be numeric"
                )
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
                raise ValueError(
                    f"generated_gray_color[{index}] must be finite in [0, 1]"
                )
        if float(self.generated_gray_color[3]) != 1.0:
            raise ValueError(
                "generated_gray_color[3] must be 1.0 for opaque generated textures"
            )
        if not isinstance(self.sequence_timing, TextureSequenceTiming):
            raise TypeError("sequence_timing must be TextureSequenceTiming")
        if not isinstance(self.texture_export_mode, A1TextureExportMode):
            raise TypeError("texture_export_mode must be A1TextureExportMode")
        if self.bake_execution.texture_export_mode is not self.texture_export_mode:
            raise ValueError(
                "bake_execution.texture_export_mode must match texture_export_mode"
            )
        if not isinstance(self.projection_direction, A1ProjectionDirection):
            raise TypeError(
                "projection_direction must be A1ProjectionDirection"
            )


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise RuntimeError("Blender bpy module is unavailable") from exc
    return bpy


def _resolve_output_directory(scene: Any) -> Path:
    bpy = _load_bpy()
    raw = str(getattr(scene, "spine2d_json_path", "") or "").strip()
    resolved = str(bpy.path.abspath(raw) if raw else "").strip()
    if not resolved:
        from ..config import get_default_output_dir

        resolved = str(get_default_output_dir()).strip()
    if not resolved:
        raise ValueError("Output directory is empty; save the .blend file first")
    return Path(resolved).expanduser().resolve(strict=False)


def _resolve_images_relative_path(scene: Any) -> str:
    value = str(getattr(scene, "spine2d_images_path", "images") or "images")
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    return normalized or "images"


def _texture_size(scene: Any) -> int:
    raw = getattr(scene, "spine2d_texture_size", 1024)
    if isinstance(raw, bool):
        raise ValueError("spine2d_texture_size must be an integer, not bool")
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spine2d_texture_size must be an integer") from exc
    if value < 64 or value > 4096 or value % 2:
        raise ValueError(
            f"Texture size must be an even integer in [64, 4096], got {value}"
        )
    return value


def _projection_alpha_threshold(scene: Any) -> float:
    raw_value = getattr(
        scene,
        "spine2d_projection_alpha_threshold",
        _DEFAULT_PROJECTION_ALPHA_THRESHOLD,
    )
    if isinstance(raw_value, bool):
        raise ValueError(
            "spine2d_projection_alpha_threshold must be numeric, not bool"
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "spine2d_projection_alpha_threshold must be numeric"
        ) from exc
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            "spine2d_projection_alpha_threshold must be finite in [0, 1]"
        )
    return value


def _finite_scene_float(
    scene: Any,
    property_name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = getattr(scene, property_name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{property_name} must be numeric, not bool")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{property_name} must be numeric") from exc
    if not isfinite(value) or value < minimum or value > maximum:
        raise ValueError(
            f"{property_name} must be finite in [{minimum}, {maximum}]"
        )
    return value


def _scene_integer(
    scene: Any,
    property_name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = getattr(scene, property_name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{property_name} must be int, not bool")
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{property_name} must be int") from exc
    if value < minimum or value > maximum:
        raise ValueError(
            f"{property_name} must be in [{minimum}, {maximum}]"
        )
    return value


def _resolve_depth_projection_settings(scene: Any) -> DepthCameraProjectionSettings:
    """Capture depth controls; only farthest-visible is public in 0.81.0."""

    raw_base = str(
        getattr(
            scene,
            "spine2d_depth_base_mode",
            DepthProjectionBaseMode.FARTHEST_VISIBLE.value,
        )
        or DepthProjectionBaseMode.FARTHEST_VISIBLE.value
    ).strip().upper()
    try:
        base_mode = DepthProjectionBaseMode(raw_base)
    except ValueError as exc:
        raise ValueError(f"Unsupported depth base mode: {raw_base!r}") from exc
    return DepthCameraProjectionSettings(
        smoothing=_finite_scene_float(
            scene,
            "spine2d_depth_smoothing",
            0.35,
            minimum=0.0,
            maximum=1.0,
        ),
        edge_threshold_fraction=_finite_scene_float(
            scene,
            "spine2d_depth_edge_threshold",
            0.08,
            minimum=0.0,
            maximum=1.0,
        ),
        mesh_error_pixels=_finite_scene_float(
            scene,
            "spine2d_depth_mesh_error_pixels",
            4.0,
            minimum=0.25,
            maximum=128.0,
        ),
        max_points=_scene_integer(
            scene,
            "spine2d_depth_max_points",
            128,
            minimum=4,
            maximum=4096,
        ),
        base_mode=base_mode,
    )


def _resolve_geometry_settings(scene: Any) -> A1GeometryPreparationSettings:
    raw_mode = str(
        getattr(
            scene,
            "spine2d_angular_mode",
            A1AngularMode.SEED_CONE.value,
        )
        or A1AngularMode.SEED_CONE.value
    ).strip().upper()
    try:
        angular_mode = A1AngularMode(raw_mode)
    except ValueError as exc:
        supported = tuple(mode.value for mode in A1AngularMode)
        raise ValueError(
            f"Unsupported Spine2D angular mode {raw_mode!r}; supported={supported}"
        ) from exc

    if angular_mode is A1AngularMode.SEED_CONE:
        local_angle_limit = None
    else:
        raw_local_limit = getattr(scene, "spine2d_local_angle_limit", None)
        local_angle_limit = (
            None
            if raw_local_limit is None or raw_local_limit == ""
            else float(raw_local_limit)
        )
    return A1GeometryPreparationSettings(
        angular_mode=angular_mode,
        local_angle_limit_degrees=local_angle_limit,
    )


def _resolve_material_source_policy(scene: Any) -> A1MaterialSourcePolicy:
    raw = str(
        getattr(
            scene,
            "spine2d_material_source_policy",
            A1MaterialSourcePolicy.REQUIRE_SOURCE.value,
        )
        or A1MaterialSourcePolicy.REQUIRE_SOURCE.value
    ).strip().upper()
    try:
        return A1MaterialSourcePolicy(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported generated material source policy: {raw!r}"
        ) from exc


def _resolve_generated_material_pattern(
    scene: Any,
) -> A1GeneratedMaterialPattern:
    raw = str(
        getattr(
            scene,
            "spine2d_generated_material_pattern",
            A1GeneratedMaterialPattern.SOLID_GRAY.value,
        )
        or A1GeneratedMaterialPattern.SOLID_GRAY.value
    ).strip().upper()
    try:
        return A1GeneratedMaterialPattern(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported generated material pattern: {raw!r}"
        ) from exc


def _resolve_generated_gray_color(scene: Any) -> ColorRGBA:
    raw = getattr(
        scene,
        "spine2d_generated_gray_color",
        _DEFAULT_GENERATED_GRAY[:3],
    )
    try:
        values = tuple(raw)
    except Exception as exc:
        raise ValueError(
            "spine2d_generated_gray_color must contain three numeric RGB values"
        ) from exc
    if len(values) != 3:
        raise ValueError(
            "spine2d_generated_gray_color must contain exactly three RGB values"
        )

    resolved: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"spine2d_generated_gray_color[{index}] must be numeric"
            )
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(
                f"spine2d_generated_gray_color[{index}] must be finite"
            )
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(
                f"spine2d_generated_gray_color[{index}] must be in [0, 1]"
            )
        resolved.append(numeric)
    return resolved[0], resolved[1], resolved[2], 1.0


def _resolve_texture_export_mode(scene: Any) -> A1TextureExportMode:
    raw = str(
        getattr(
            scene,
            "spine2d_texture_export_mode",
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
        )
        or A1TextureExportMode.NORMAL_UV_SEGMENTS.value
    ).strip().upper()
    try:
        return A1TextureExportMode(raw)
    except ValueError as exc:
        supported = tuple(mode.value for mode in A1TextureExportMode)
        raise ValueError(
            f"Unsupported texture export mode {raw!r}; supported={supported}"
        ) from exc


def _resolve_projection_direction(scene: Any) -> A1ProjectionDirection:
    raw = getattr(
        scene,
        "spine2d_projection_direction",
        A1ProjectionDirection.POSITIVE_Z.value,
    )
    return resolve_a1_projection_direction(raw)


def _resolve_spine_target(scene: Any) -> SpineJsonTarget:
    raw = getattr(
        scene,
        "spine2d_target_spine_version",
        DEFAULT_SPINE_JSON_TARGET.value,
    )
    return resolve_spine_json_target(raw)


def _resolve_rig_profile(scene: Any) -> A1RigProfile:
    raw = getattr(
        scene,
        "spine2d_rig_profile",
        _PUBLIC_RIG_PROFILE.value,
    )
    resolved = resolve_a1_rig_profile(raw)
    if resolved is not _PUBLIC_RIG_PROFILE:
        logger.warning(
            "Scene requested hidden rig profile %s; public Rewrite UI uses %s",
            resolved.value,
            _PUBLIC_RIG_PROFILE.value,
        )
        return _PUBLIC_RIG_PROFILE
    return resolved


def _resolve_sequence_timing(scene: Any) -> TextureSequenceTiming:
    render = getattr(scene, "render", None)
    try:
        scene_fps = max(0, int(getattr(render, "fps", 30)))
    except (TypeError, ValueError, OverflowError):
        scene_fps = 0
    try:
        scene_fps_base = max(0.0, float(getattr(render, "fps_base", 1.0)))
    except (TypeError, ValueError, OverflowError):
        scene_fps_base = 0.0
    try:
        override_fps = max(
            0.0,
            float(getattr(scene, "spine2d_sequence_fps_override", 0.0)),
        )
    except (TypeError, ValueError, OverflowError):
        override_fps = 0.0
    return TextureSequenceTiming(
        scene_fps=scene_fps,
        scene_fps_base=scene_fps_base,
        override_fps=override_fps,
    )


def _resolve_camera_influence_policy(
    scene: Any,
) -> CameraProjectionInfluencePolicy:
    return CameraProjectionInfluencePolicy(
        include_scene_shadows=bool(
            getattr(scene, "spine2d_include_scene_shadows", True)
        ),
        include_scene_reflection_transmission=bool(
            getattr(
                scene,
                "spine2d_include_scene_reflection_transmission",
                True,
            )
        ),
        world_affects_lighting_reflections=bool(
            getattr(scene, "spine2d_world_affects_lighting_reflections", True)
        ),
    )


def _capture_scene_profile(
    scene: Any,
    *,
    output_directory: Path | None = None,
    texture_size: int | None = None,
    images_relative_path: str | None = None,
) -> _SceneExportProfile:
    """Capture every mutable Scene setting exactly once for one export request."""

    seam_mode = str(
        getattr(scene, "spine2d_seam_maker_mode", "AUTO") or "AUTO"
    ).strip().upper()
    angle_limit = float(getattr(scene, "spine2d_angle_limit", 30.0))
    render_engine = str(
        getattr(getattr(scene, "render", None), "engine", "CYCLES")
        or "CYCLES"
    )
    texture_export_mode = _resolve_texture_export_mode(scene)
    sequence_timing = _resolve_sequence_timing(scene)
    bake_execution = BakeExecutionSettings(
        render_engine=render_engine,
        projection_alpha_threshold=_projection_alpha_threshold(scene),
        texture_export_mode=texture_export_mode,
        camera_influence_policy=_resolve_camera_influence_policy(scene),
        depth_projection=_resolve_depth_projection_settings(scene),
    )
    return _SceneExportProfile(
        output_directory=(
            output_directory
            if output_directory is not None
            else _resolve_output_directory(scene)
        ),
        images_relative_path=(
            images_relative_path
            if images_relative_path is not None
            else _resolve_images_relative_path(scene)
        ),
        texture_size=(
            texture_size if texture_size is not None else _texture_size(scene)
        ),
        seam_mode=seam_mode,
        angle_limit_degrees=angle_limit,
        geometry=_resolve_geometry_settings(scene),
        bake_execution=bake_execution,
        include_control_icons=bool(
            getattr(scene, "spine2d_control_icons", False)
        ),
        include_preview_animation=_PREVIEW_ANIMATION_EXPORT_ENABLED,
        spine_target=_resolve_spine_target(scene),
        rig_profile=_resolve_rig_profile(scene),
        material_source_policy=_resolve_material_source_policy(scene),
        generated_material_pattern=_resolve_generated_material_pattern(scene),
        generated_gray_color=_resolve_generated_gray_color(scene),
        sequence_timing=sequence_timing,
        texture_export_mode=texture_export_mode,
        projection_direction=_resolve_projection_direction(scene),
    )


__all__ = [
    "_SceneExportProfile",
    "_capture_scene_profile",
    "_projection_alpha_threshold",
    "_resolve_camera_influence_policy",
    "_resolve_depth_projection_settings",
    "_resolve_generated_gray_color",
    "_resolve_generated_material_pattern",
    "_resolve_geometry_settings",
    "_resolve_images_relative_path",
    "_resolve_material_source_policy",
    "_resolve_output_directory",
    "_resolve_projection_direction",
    "_resolve_rig_profile",
    "_resolve_sequence_timing",
    "_resolve_spine_target",
    "_resolve_texture_export_mode",
    "_texture_size",
]
