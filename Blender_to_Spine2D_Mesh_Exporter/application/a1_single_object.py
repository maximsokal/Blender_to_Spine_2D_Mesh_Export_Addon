"""Typed contracts and pure helpers for one complete A1 object export.

The live Blender object is intentionally absent from this module. Blender adapters
consume this immutable configuration and return the generic :class:`ExportResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Tuple

from ..domain.baking import (
    BakeExecutionSettings,
    BakeMaterialPolicy,
    BakeMode,
    BakePlan,
    BakeSettings,
    TextureFormat,
    sanitize_filename_stem,
)
from ..domain.baking.generated_materials import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
)
from ..domain.geometry import MeshSnapshot, ModifierLineagePolicy
from ..domain.spine import LegacyAttachmentSequence
from ..domain.spine.legacy_rig_contracts import UniformScaleMode
from ..domain.spine.legacy_rig_scale import calculate_uniform_scale
from ..domain.uv import UvUnwrapSettings
from .a1_geometry_preparation import A1GeometryPreparationSettings
from .a1_numeric_contracts import (
    require_finite_number,
    require_integer,
    require_non_empty_string,
)
from .contracts import ExportSettings


class A1SourceGeometryMode(str, Enum):
    ORIGINAL = "ORIGINAL"
    EVALUATED = "EVALUATED"


class A1SourceUvBoundaryMode(str, Enum):
    """Select which pre-unwrap UV layout may influence A1 segmentation.

    ``DISABLED`` is the Rewrite default because A1 generates a new shared
    ``SpineBakeUV`` layout after segmentation and decomposition. The source
    Blender active UV layer therefore must not silently change export topology.

    ``EXPLICIT_LAYER`` opts into source-UV discontinuity cuts using the exact
    layer named by ``source_uv_boundary_layer_name``.

    ``ACTIVE_LAYER_LEGACY`` retains the historical active-layer behavior only
    for callers that explicitly request compatibility.
    """

    DISABLED = "DISABLED"
    EXPLICIT_LAYER = "EXPLICIT_LAYER"
    ACTIVE_LAYER_LEGACY = "ACTIVE_LAYER_LEGACY"


class A1SingleObjectStage(str, Enum):
    VALIDATE_REQUEST = "VALIDATE_REQUEST"
    READ_GEOMETRY = "READ_GEOMETRY"
    ASSIGN_Z_GROUPS = "ASSIGN_Z_GROUPS"
    PREPARE_GEOMETRY = "PREPARE_GEOMETRY"
    BUILD_TEXTURING_TOPOLOGY = "BUILD_TEXTURING_TOPOLOGY"
    UNWRAP_TEXTURE_UV = "UNWRAP_TEXTURE_UV"
    PROPAGATE_REGION_UV = "PROPAGATE_REGION_UV"
    ANALYZE_MATERIALS = "ANALYZE_MATERIALS"
    PLAN_BAKE = "PLAN_BAKE"
    BUILD_RIG = "BUILD_RIG"
    ASSEMBLE_DOCUMENT = "ASSEMBLE_DOCUMENT"
    STAGE_OUTPUTS = "STAGE_OUTPUTS"
    COMMIT_OUTPUTS = "COMMIT_OUTPUTS"

    @property
    def error_code(self) -> str:
        return f"A1_{self.value}_FAILED"


@dataclass(frozen=True, slots=True)
class A1SingleObjectExportSettings:
    export: ExportSettings
    prefix: str | None = None
    output_stem: str | None = None
    json_output_stem: str | None = None
    source_geometry_mode: A1SourceGeometryMode = A1SourceGeometryMode.EVALUATED
    modifier_lineage_policy: ModifierLineagePolicy = (
        ModifierLineagePolicy.STRICT_PRESERVE
    )
    geometry: A1GeometryPreparationSettings = A1GeometryPreparationSettings()
    uv: UvUnwrapSettings = UvUnwrapSettings()
    texture_format: TextureFormat = TextureFormat.PNG
    material_policy: BakeMaterialPolicy = (
        BakeMaterialPolicy.IMAGE_DEPENDENCY_DIFFUSE
    )
    diffuse_mode: BakeMode = BakeMode.DIFFUSE
    # A1 exports albedo-like Spine textures. COMBINED depends on scene lighting and
    # can return an opaque black image in headless/empty scenes even though Blender
    # reports FINISHED. DIFFUSE with color-only passes is lighting-independent and
    # still evaluates Image Texture and procedural Base Color node graphs.
    procedural_mode: BakeMode = BakeMode.DIFFUSE
    selected_to_active: bool = False
    cage_extrusion: float = 0.1
    bake_execution: BakeExecutionSettings = BakeExecutionSettings()
    rig_scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE
    use_world_location_for_main_bone: bool = True
    include_control_icons: bool = False
    include_preview_animation: bool = False
    json_indent: int = 2
    # Appended to preserve the positional layout of the pre-hardening dataclass.
    source_uv_boundary_mode: A1SourceUvBoundaryMode = (
        A1SourceUvBoundaryMode.DISABLED
    )
    source_uv_boundary_layer_name: str | None = None
    material_source_policy: A1MaterialSourcePolicy = (
        A1MaterialSourcePolicy.REQUIRE_SOURCE
    )
    generated_material_pattern: A1GeneratedMaterialPattern = (
        A1GeneratedMaterialPattern.SOLID_GRAY
    )
    generated_gray_color: Tuple[float, float, float, float] = (
        0.5,
        0.5,
        0.5,
        1.0,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.export, ExportSettings):
            raise TypeError("export must be ExportSettings")
        if self.export.spine_version != "4.2.43":
            raise ValueError("A1 currently supports Spine 4.2.43 only")
        if self.export.rig_profile != "LEGACY_ROTATABLE_MESH":
            raise ValueError("A1 requires rig_profile LEGACY_ROTATABLE_MESH")
        for field_name in ("prefix", "output_stem", "json_output_stem"):
            value = getattr(self, field_name)
            if value is not None:
                require_non_empty_string(value, field_name)
        if not isinstance(self.source_geometry_mode, A1SourceGeometryMode):
            raise TypeError("source_geometry_mode must be A1SourceGeometryMode")
        if not isinstance(self.modifier_lineage_policy, ModifierLineagePolicy):
            raise TypeError(
                "modifier_lineage_policy must be ModifierLineagePolicy"
            )
        if not isinstance(self.geometry, A1GeometryPreparationSettings):
            raise TypeError("geometry must be A1GeometryPreparationSettings")
        if not isinstance(self.source_uv_boundary_mode, A1SourceUvBoundaryMode):
            raise TypeError(
                "source_uv_boundary_mode must be A1SourceUvBoundaryMode"
            )
        if self.source_uv_boundary_mode is A1SourceUvBoundaryMode.EXPLICIT_LAYER:
            if self.source_uv_boundary_layer_name is None:
                raise ValueError(
                    "source_uv_boundary_layer_name is required for EXPLICIT_LAYER"
                )
            require_non_empty_string(
                self.source_uv_boundary_layer_name,
                "source_uv_boundary_layer_name",
            )
        elif self.source_uv_boundary_layer_name is not None:
            raise ValueError(
                "source_uv_boundary_layer_name is only valid for EXPLICIT_LAYER"
            )
        if not isinstance(self.uv, UvUnwrapSettings):
            raise TypeError("uv must be UvUnwrapSettings")
        if not isinstance(self.texture_format, TextureFormat):
            raise TypeError("texture_format must be TextureFormat")
        if not isinstance(self.material_policy, BakeMaterialPolicy):
            raise TypeError("material_policy must be BakeMaterialPolicy")
        for field_name in ("diffuse_mode", "procedural_mode"):
            if not isinstance(getattr(self, field_name), BakeMode):
                raise TypeError(f"{field_name} must be BakeMode")
        if not isinstance(self.selected_to_active, bool):
            raise TypeError("selected_to_active must be bool")
        require_finite_number(
            self.cage_extrusion,
            "cage_extrusion",
            minimum=0.0,
        )
        if not isinstance(self.bake_execution, BakeExecutionSettings):
            raise TypeError("bake_execution must be BakeExecutionSettings")
        if not isinstance(self.rig_scale_mode, UniformScaleMode):
            raise TypeError("rig_scale_mode must be UniformScaleMode")
        for field_name in (
            "use_world_location_for_main_bone",
            "include_control_icons",
            "include_preview_animation",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        require_integer(self.json_indent, "json_indent", minimum=0, maximum=16)
        if not isinstance(self.material_source_policy, A1MaterialSourcePolicy):
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
            require_finite_number(
                value,
                f"generated_gray_color[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
        if float(self.generated_gray_color[3]) != 1.0:
            raise ValueError(
                "generated_gray_color[3] must be 1.0 for opaque generated textures"
            )

    def resolved_geometry_settings(self) -> A1GeometryPreparationSettings:
        segmentation = self.geometry.segmentation

        # A1 unwraps one shared full-object texturing snapshot after segmentation.
        # Consequently, source UV discontinuities are not part of the generated
        # SpineBakeUV topology unless the caller opts in explicitly.
        if self.source_uv_boundary_mode is A1SourceUvBoundaryMode.DISABLED:
            segmentation = replace(
                segmentation,
                split_uv_boundaries=False,
                uv_layer_name=None,
            )
        elif self.source_uv_boundary_mode is A1SourceUvBoundaryMode.EXPLICIT_LAYER:
            segmentation = replace(
                segmentation,
                split_uv_boundaries=True,
                uv_layer_name=self.source_uv_boundary_layer_name,
            )
        else:
            segmentation = replace(
                segmentation,
                split_uv_boundaries=True,
                uv_layer_name=None,
            )

        if self.export.seam_mode == "CUSTOM":
            segmentation = replace(
                segmentation,
                split_by_angle=False,
                respect_seams=True,
            )
        else:
            segmentation = replace(
                segmentation,
                split_by_angle=True,
                angle_limit_degrees=self.export.angle_limit_degrees,
                respect_seams=True,
            )
        return replace(self.geometry, segmentation=segmentation)


@dataclass(frozen=True, slots=True)
class A1ResolvedOutputPaths:
    output_stem: str
    json_output_stem: str
    json_path: Path
    image_directory: Path
    image_relative_directory: str


@dataclass(frozen=True, slots=True)
class A1MeshBounds:
    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float
    center_x: float
    center_y: float


def resolve_a1_names(
    object_name: str,
    settings: A1SingleObjectExportSettings,
) -> Tuple[str, str]:
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("object_name must be a non-empty string")
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    prefix = (settings.prefix or object_name).strip()
    output_stem = sanitize_filename_stem(settings.output_stem or prefix)
    return prefix, output_stem


def resolve_a1_output_paths(
    object_name: str,
    settings: A1SingleObjectExportSettings,
) -> A1ResolvedOutputPaths:
    _, output_stem = resolve_a1_names(object_name, settings)
    json_output_stem = sanitize_filename_stem(
        settings.json_output_stem or output_stem
    )
    output_directory = settings.export.output_directory.expanduser().resolve(
        strict=False
    )
    raw_relative = settings.export.images_relative_path.replace("\\", "/").strip("/")
    if raw_relative in {"", "."}:
        image_relative_directory = ""
        image_directory = output_directory
    else:
        relative_path = PurePosixPath(raw_relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                "images_relative_path must be a safe relative directory"
            )
        image_relative_directory = relative_path.as_posix()
        image_directory = output_directory.joinpath(*relative_path.parts)
    return A1ResolvedOutputPaths(
        output_stem=output_stem,
        json_output_stem=json_output_stem,
        json_path=output_directory / f"{json_output_stem}.json",
        image_directory=image_directory,
        image_relative_directory=image_relative_directory,
    )


def calculate_a1_mesh_bounds(snapshot: MeshSnapshot) -> A1MeshBounds:
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not snapshot.vertices:
        raise ValueError("snapshot contains no vertices")
    x_values = tuple(float(vertex.position[0]) for vertex in snapshot.vertices)
    y_values = tuple(float(vertex.position[1]) for vertex in snapshot.vertices)
    minimum_x = min(x_values)
    maximum_x = max(x_values)
    minimum_y = min(y_values)
    maximum_y = max(y_values)
    return A1MeshBounds(
        minimum_x=minimum_x,
        maximum_x=maximum_x,
        minimum_y=minimum_y,
        maximum_y=maximum_y,
        center_x=(minimum_x + maximum_x) / 2.0,
        center_y=(minimum_y + maximum_y) / 2.0,
    )


def build_a1_bake_settings(
    object_name: str,
    settings: A1SingleObjectExportSettings,
) -> BakeSettings:
    paths = resolve_a1_output_paths(object_name, settings)
    return BakeSettings(
        width=settings.export.texture_width,
        height=settings.export.texture_height,
        output_directory=paths.image_directory,
        output_stem=paths.output_stem,
        uv_layer_name=settings.uv.layer_name,
        texture_format=settings.texture_format,
        margin_pixels=settings.export.bake_margin,
        selected_to_active=settings.selected_to_active,
        cage_extrusion=settings.cage_extrusion,
        diffuse_mode=settings.diffuse_mode,
        procedural_mode=settings.procedural_mode,
        material_policy=settings.material_policy,
        sequence_start_frame=settings.export.sequence_start_frame,
        sequence_frame_count=settings.export.sequence_frame_count,
    )


def build_a1_attachment_path(
    bake_plan: BakePlan,
    paths: A1ResolvedOutputPaths,
) -> str:
    if not isinstance(bake_plan, BakePlan):
        raise TypeError("bake_plan must be BakePlan")
    if not isinstance(paths, A1ResolvedOutputPaths):
        raise TypeError("paths must be A1ResolvedOutputPaths")
    base_name = (
        f"{paths.output_stem}_Baked_"
        if bake_plan.sequence
        else bake_plan.representative_task.image_name
    )
    if paths.image_relative_directory:
        return PurePosixPath(
            paths.image_relative_directory,
            base_name,
        ).as_posix()
    return base_name


def build_a1_attachment_sequence(
    bake_plan: BakePlan,
) -> LegacyAttachmentSequence | None:
    if not isinstance(bake_plan, BakePlan):
        raise TypeError("bake_plan must be BakePlan")
    if not bake_plan.sequence:
        return None
    settings = bake_plan.settings
    return LegacyAttachmentSequence(
        count=settings.sequence_frame_count,
        start=settings.sequence_start_frame,
        digits=settings.sequence_frame_digits,
    )


def calculate_a1_main_position_pixels(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
) -> tuple[float, float] | None:
    if not settings.use_world_location_for_main_bone:
        return None
    if len(snapshot.world_matrix) != 16:
        raise ValueError("snapshot.world_matrix must contain 16 values")
    uniform_scale = calculate_uniform_scale(
        settings.export.texture_width,
        settings.export.texture_height,
        settings.rig_scale_mode,
    )
    return (
        float(snapshot.world_matrix[3]) * uniform_scale,
        float(snapshot.world_matrix[7]) * uniform_scale,
    )
