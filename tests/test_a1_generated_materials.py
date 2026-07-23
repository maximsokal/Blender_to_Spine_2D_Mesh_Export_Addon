from dataclasses import replace
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1GeometryPreparationSettings,
    A1SingleObjectExportSettings,
    build_a1_texturing_topology,
    prepare_a1_geometry_regions,
    propagate_texturing_uv_to_regions,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_generated_materials import (
    build_generated_material_plan,
    generated_palette_color,
)
from Blender_to_Spine2D_Mesh_Exporter.application.contracts import ExportSettings
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.generated_materials import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    GeneratedMaterialPlan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    LoopUV,
    SegmentationSettings,
)

from test_a1_segmentation_decomposition import build_quad_ring
from test_geometry_domain import build_square_snapshot


def _with_test_uv(snapshot):
    vertex_by_id = snapshot.vertex_by_id()
    x_values = tuple(vertex.position[0] for vertex in snapshot.vertices)
    y_values = tuple(vertex.position[1] for vertex in snapshot.vertices)
    minimum_x, maximum_x = min(x_values), max(x_values)
    minimum_y, maximum_y = min(y_values), max(y_values)
    span_x = maximum_x - minimum_x or 1.0
    span_y = maximum_y - minimum_y or 1.0
    loops = tuple(
        replace(
            loop,
            uvs=(
                LoopUV(
                    "UVMap",
                    (
                        (vertex_by_id[loop.vertex_id].position[0] - minimum_x) / span_x,
                        (vertex_by_id[loop.vertex_id].position[1] - minimum_y) / span_y,
                    ),
                ),
            ),
        )
        for loop in snapshot.loops
    )
    return replace(
        snapshot,
        loops=loops,
        uv_layer_names=("UVMap",),
        active_uv_layer="UVMap",
        render_uv_layer="UVMap",
    )


def _final_regions(snapshot, settings=None):
    source = _with_test_uv(snapshot)
    geometry = prepare_a1_geometry_regions(source, settings)
    topology = build_a1_texturing_topology(source, geometry)
    return propagate_texturing_uv_to_regions(
        topology.snapshot,
        geometry,
        source_layer_name="UVMap",
        target_layer_name="SpineBakeUV",
    )


def test_generated_material_defaults_preserve_strict_source_policy(tmp_path: Path):
    settings = A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=tmp_path,
        )
    )

    assert settings.material_source_policy is A1MaterialSourcePolicy.REQUIRE_SOURCE
    assert settings.generated_material_pattern is A1GeneratedMaterialPattern.SOLID_GRAY
    assert settings.generated_gray_color == (0.5, 0.5, 0.5, 1.0)


def test_generated_material_settings_reject_invalid_gray_components(tmp_path: Path):
    export = ExportSettings(
        texture_width=128,
        texture_height=128,
        output_directory=tmp_path,
    )

    with pytest.raises(TypeError, match=r"generated_gray_color\[0\]"):
        A1SingleObjectExportSettings(
            export=export,
            generated_gray_color=(True, 0.5, 0.5, 1.0),
        )
    with pytest.raises(ValueError, match=r"generated_gray_color\[1\].*finite"):
        A1SingleObjectExportSettings(
            export=export,
            generated_gray_color=(0.5, float("nan"), 0.5, 1.0),
        )
    with pytest.raises(ValueError, match=r"generated_gray_color\[2\].*<= 1.0"):
        A1SingleObjectExportSettings(
            export=export,
            generated_gray_color=(0.5, 0.5, 1.1, 1.0),
        )


def test_generated_palette_is_deterministic_opaque_and_distinct():
    first = tuple(generated_palette_color(index) for index in range(12))
    second = tuple(generated_palette_color(index) for index in range(12))

    assert first == second
    assert len(set(first)) == len(first)
    assert all(color[3] == 1.0 for color in first)
    assert all(0.0 <= component <= 1.0 for color in first for component in color)


def test_solid_gray_colors_every_final_export_triangle():
    regions = _final_regions(build_square_snapshot())
    gray = (0.25, 0.25, 0.25, 1.0)

    plan = build_generated_material_plan(
        regions,
        source_policy=A1MaterialSourcePolicy.FORCE_GENERATED,
        pattern=A1GeneratedMaterialPattern.SOLID_GRAY,
        gray_color=gray,
    )

    assert len(plan.target_snapshot.faces) == sum(
        len(region.snapshot.faces) for region in regions.regions
    )
    assert plan.face_colors == (gray,) * len(plan.target_snapshot.faces)
    assert {face.material_index for face in plan.target_snapshot.faces} == {0}
    assert plan.target_snapshot.active_uv_layer == "SpineBakeUV"


def test_region_pattern_uses_one_color_per_final_decomposition_region():
    regions = _final_regions(
        build_quad_ring(),
        A1GeometryPreparationSettings(
            segmentation=SegmentationSettings(
                split_by_angle=False,
                split_materials=False,
                split_uv_boundaries=False,
            )
        ),
    )
    assert len(regions.regions) > 1

    plan = build_generated_material_plan(
        regions,
        source_policy=A1MaterialSourcePolicy.FORCE_GENERATED,
        pattern=A1GeneratedMaterialPattern.REGION_COLORS,
    )

    offset = 0
    region_colors = []
    for region in regions.regions:
        face_count = len(region.snapshot.faces)
        colors = plan.face_colors[offset : offset + face_count]
        assert colors == (colors[0],) * face_count
        region_colors.append(colors[0])
        offset += face_count
    assert len(set(region_colors)) == len(region_colors)
    assert offset == len(plan.face_colors)


def test_exported_face_pattern_uses_one_color_per_final_triangle():
    regions = _final_regions(build_square_snapshot())

    plan = build_generated_material_plan(
        regions,
        source_policy=A1MaterialSourcePolicy.GENERATE_IF_MISSING,
        pattern=A1GeneratedMaterialPattern.EXPORTED_FACE_COLORS,
    )

    assert len(plan.target_snapshot.faces) >= 2
    assert len(set(plan.face_colors)) == len(plan.target_snapshot.faces)
    assert tuple(face.id.index for face in plan.target_snapshot.faces) == tuple(
        range(len(plan.target_snapshot.faces))
    )
    assert tuple(loop.id.index for loop in plan.target_snapshot.loops) == tuple(
        range(len(plan.target_snapshot.loops))
    )


def test_generated_material_plan_rejects_face_color_count_mismatch():
    regions = _final_regions(build_square_snapshot())
    valid = build_generated_material_plan(
        regions,
        source_policy=A1MaterialSourcePolicy.FORCE_GENERATED,
        pattern=A1GeneratedMaterialPattern.SOLID_GRAY,
    )

    with pytest.raises(ValueError, match="one color for every"):
        GeneratedMaterialPlan(
            source_policy=valid.source_policy,
            pattern=valid.pattern,
            target_snapshot=valid.target_snapshot,
            face_colors=valid.face_colors[:-1],
        )


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _source(relative_path: str) -> str:
    return (PACKAGE / relative_path).read_text(encoding="utf-8")


def test_generated_material_pipeline_never_imports_legacy_implementation_modules():
    generated_sources = "\n".join(
        _source(path)
        for path in (
            "domain/baking/generated_materials.py",
            "application/a1_generated_materials.py",
            "blender_adapter/generated_material_ui.py",
            "blender_adapter/a1_texture_planning.py",
        )
    )
    for forbidden in (
        "from .main",
        "from ..main",
        "texture_baker",
        "multi_object_export",
        "legacy_loader",
    ):
        assert forbidden not in generated_sources


def test_generated_bake_uses_one_temporary_material_and_corner_color_attribute():
    source = _source("blender_adapter/bake_materials.py")

    assert 'type="FLOAT_COLOR"' in source
    assert 'domain="CORNER"' in source
    assert "build_mesh_topology_correspondence" in source
    assert "ShaderNodeEmission" in source
    assert "ShaderNodeTexImage" in source
    assert "target_mesh.materials.append(generated_copy)" in source
    assert "for face, color in zip(" in source
    assert "bpy_module.data.materials.new" in source
    assert "source_obj.data.materials" not in source


def test_generated_runtime_bypasses_source_slots_but_keeps_source_path_strict():
    validation = _source("blender_adapter/semantic_bake_validation.py")
    planning = _source("blender_adapter/a1_texture_planning.py")

    assert "isinstance(plan, GeneratedBakePlan)" in validation
    assert "Generated target snapshot must reference only synthetic slot zero" in validation
    assert "len(source_slots) != len(plan.material_analysis.slots)" in validation
    assert "A1MaterialSourcePolicy.REQUIRE_SOURCE" in planning
    assert "A1MaterialSourcePolicy.FORCE_GENERATED" in planning
    assert "GENERATED_MATERIAL_ACTIVE" in planning


def test_generated_material_resources_are_removed_in_finally():
    source = _source("blender_adapter/bake_materials.py")
    finally_position = source.index("    finally:", source.index("def temporary_bake_materials"))
    cleanup = source[finally_position:]

    assert "_remove_color_attribute" in cleanup
    assert "_clear_material_slots" in cleanup
    assert "_remove_materials" in cleanup


def test_generated_controls_are_registered_as_rewrite_only_child_panel():
    source = _source("blender_adapter/generated_material_ui.py")
    root = _source("__init__.py")

    assert 'bl_parent_id = "OBJECT_PT_spine2d_mesh"' in source
    assert "These controls are ignored by Legacy export" in source
    assert "spine2d_material_source_policy" in source
    assert "spine2d_generated_material_pattern" in source
    assert "spine2d_generated_gray_color" in source
    assert "generated_material_ui.register" in root
    assert "generated_material_ui.unregister" in root
