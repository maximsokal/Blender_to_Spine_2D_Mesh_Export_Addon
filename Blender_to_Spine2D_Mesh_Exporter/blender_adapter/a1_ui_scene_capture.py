"""Capture one immutable Rewrite export profile from mutable Blender Scene RNA."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from ..application import A1GeometryPreparationSettings
from ..domain.baking import BakeExecutionSettings
from ..domain.baking.generated_materials import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    ColorRGBA,
)
from ..domain.geometry import A1AngularMode


_DEFAULT_PROJECTION_ALPHA_THRESHOLD = 1.0 / 255.0
_DEFAULT_GENERATED_GRAY: ColorRGBA = (0.5, 0.5, 0.5, 1.0)


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
    material_source_policy: A1MaterialSourcePolicy = (
        A1MaterialSourcePolicy.REQUIRE_SOURCE
    )
    generated_material_pattern: A1GeneratedMaterialPattern = (
        A1GeneratedMaterialPattern.SOLID_GRAY
    )
    generated_gray_color: ColorRGBA = _DEFAULT_GENERATED_GRAY

    def __post_init__(self) -> None:
        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be pathlib.Path")
        if (
            not isinstance(self.images_relative_path, str)
            or not self.images_relative_path
        ):
            raise ValueError("images_relative_path must be a non-empty string")
        if not isinstance(self.texture_size, int) or self.texture_size <= 0:
            raise ValueError("texture_size must be a positive integer")
        if self.seam_mode not in {"AUTO", "CUSTOM"}:
            raise ValueError("seam_mode must be AUTO or CUSTOM")
        if not isinstance(self.angle_limit_degrees, (int, float)):
            raise TypeError("angle_limit_degrees must be numeric")
        if not isinstance(self.geometry, A1GeometryPreparationSettings):
            raise TypeError("geometry must be A1GeometryPreparationSettings")
        if not isinstance(self.bake_execution, BakeExecutionSettings):
            raise TypeError("bake_execution must be BakeExecutionSettings")
        if not isinstance(self.include_control_icons, bool):
            raise TypeError("include_control_icons must be bool")
        if not isinstance(self.include_preview_animation, bool):
            raise TypeError("include_preview_animation must be bool")
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
            if not isfinite(float(value)) or value < 0.0 or value > 1.0:
                raise ValueError(
                    f"generated_gray_color[{index}] must be finite in [0, 1]"
                )
        if float(self.generated_gray_color[3]) != 1.0:
            raise ValueError(
                "generated_gray_color[3] must be 1.0 for opaque generated textures"
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
    value = int(getattr(scene, "spine2d_texture_size", 1024))
    if value <= 0:
        raise ValueError(f"Texture size must be positive, got {value}")
    return value


def _projection_alpha_threshold(scene: Any) -> float:
    property_name = "spine2d_projection_alpha_threshold"
    raw_value = getattr(scene, property_name, None)
    if raw_value is None:
        getter = getattr(scene, "get", None)
        if callable(getter):
            raw_value = getter(
                property_name,
                _DEFAULT_PROJECTION_ALPHA_THRESHOLD,
            )
    if raw_value is None:
        raw_value = _DEFAULT_PROJECTION_ALPHA_THRESHOLD
    if isinstance(raw_value, bool):
        raise ValueError(
            "spine2d_projection_alpha_threshold must be numeric, not bool"
        )
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "spine2d_projection_alpha_threshold must be numeric"
        ) from exc


def _resolve_geometry_settings(scene: Any) -> A1GeometryPreparationSettings:
    raw_mode = str(
        getattr(
            scene,
            "spine2d_angular_mode",
            A1AngularMode.LEGACY_SEED_CONE.value,
        )
        or A1AngularMode.LEGACY_SEED_CONE.value
    ).strip().upper()
    try:
        angular_mode = A1AngularMode(raw_mode)
    except ValueError as exc:
        supported = tuple(mode.value for mode in A1AngularMode)
        raise ValueError(
            f"Unsupported Spine2D angular mode {raw_mode!r}; supported={supported}"
        ) from exc

    if angular_mode is A1AngularMode.LEGACY_SEED_CONE:
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
    """Read RGB Scene RNA and normalize legacy RGBA values to opaque RGBA."""

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

    if len(values) not in {3, 4}:
        raise ValueError(
            "spine2d_generated_gray_color must contain three RGB values "
            "(legacy four-component values are also accepted)"
        )

    resolved: list[float] = []
    for index, value in enumerate(values[:3]):
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
        bake_execution=BakeExecutionSettings(
            render_engine=render_engine,
            projection_alpha_threshold=_projection_alpha_threshold(scene),
        ),
        include_control_icons=bool(
            getattr(scene, "spine2d_control_icons", True)
        ),
        include_preview_animation=bool(
            getattr(scene, "spine2d_export_preview_animation", True)
        ),
        material_source_policy=_resolve_material_source_policy(scene),
        generated_material_pattern=_resolve_generated_material_pattern(scene),
        generated_gray_color=_resolve_generated_gray_color(scene),
    )


__all__ = [
    "_SceneExportProfile",
    "_capture_scene_profile",
    "_projection_alpha_threshold",
    "_resolve_generated_gray_color",
    "_resolve_generated_material_pattern",
    "_resolve_geometry_settings",
    "_resolve_images_relative_path",
    "_resolve_material_source_policy",
    "_resolve_output_directory",
    "_texture_size",
]
