from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _projection_alpha_threshold,
    _resolve_generated_gray_color,
    _texture_size,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    GeneratedMaterialPlan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)


IDENTITY = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
GENERATED_UI = PACKAGE / "blender_adapter" / "generated_material_ui.py"
SCENE_CAPTURE = PACKAGE / "blender_adapter" / "a1_ui_scene_capture.py"
SCENE_PROPERTIES = PACKAGE / "blender_adapter" / "scene_properties.py"


def _triangle_snapshot() -> MeshSnapshot:
    source = "Hero"
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(source, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(((0, 1), (1, 2), (0, 2)))
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(source, 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
            uvs=(),
        )
        for index in range(3)
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(source, 0),
        loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
        material_index=0,
        normal=(0.0, 0.0, 1.0),
    )
    return MeshSnapshot(
        snapshot_id="Hero:generated",
        source_object_id=source,
        object_name="Hero",
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
        world_matrix=IDENTITY,
    )


def test_generated_material_plan_rejects_translucent_face_colors():
    with pytest.raises(ValueError, match="opaque generated textures"):
        GeneratedMaterialPlan(
            source_policy=A1MaterialSourcePolicy.FORCE_GENERATED,
            pattern=A1GeneratedMaterialPattern.SOLID_GRAY,
            target_snapshot=_triangle_snapshot(),
            face_colors=((0.5, 0.5, 0.5, 0.5),),
        )


def test_a1_settings_reject_translucent_generated_gray(tmp_path: Path):
    with pytest.raises(ValueError, match="opaque generated textures"):
        A1SingleObjectExportSettings(
            export=ExportSettings(
                texture_width=64,
                texture_height=64,
                output_directory=tmp_path,
            ),
            generated_gray_color=(0.5, 0.5, 0.5, 0.5),
        )


def test_scene_capture_appends_opaque_alpha_to_rgb():
    assert _resolve_generated_gray_color(
        SimpleNamespace(spine2d_generated_gray_color=(0.2, 0.3, 0.4))
    ) == (0.2, 0.3, 0.4, 1.0)


def test_scene_capture_rejects_legacy_rgba_vector():
    with pytest.raises(ValueError, match="exactly three RGB values"):
        _resolve_generated_gray_color(
            SimpleNamespace(spine2d_generated_gray_color=(0.2, 0.3, 0.4, 0.25))
        )


def test_scene_capture_rejects_invalid_rgb_length():
    with pytest.raises(ValueError, match="exactly three RGB values"):
        _resolve_generated_gray_color(
            SimpleNamespace(spine2d_generated_gray_color=(0.2, 0.3))
        )


def test_scene_capture_rejects_non_finite_rgb():
    with pytest.raises(ValueError, match=r"generated_gray_color\[1\].*finite"):
        _resolve_generated_gray_color(
            SimpleNamespace(spine2d_generated_gray_color=(0.2, float("nan"), 0.4))
        )


def test_projection_alpha_reads_registered_rna_value():
    assert _projection_alpha_threshold(
        SimpleNamespace(spine2d_projection_alpha_threshold=0.125)
    ) == 0.125


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_projection_alpha_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="projection_alpha_threshold"):
        _projection_alpha_threshold(
            SimpleNamespace(spine2d_projection_alpha_threshold=value)
        )


@pytest.mark.parametrize("value", [63, 65, 4097, True, "invalid"])
def test_texture_size_rejects_values_outside_blender_52_rna_contract(value):
    with pytest.raises(ValueError, match="spine2d_texture_size|Texture size"):
        _texture_size(SimpleNamespace(spine2d_texture_size=value))


def test_generated_gray_scene_rna_exposes_display_rgb_without_alpha():
    source = GENERATED_UI.read_text(encoding="utf-8")

    assert 'subtype="COLOR_GAMMA"' in source
    assert "size=3" in source
    assert "default=(0.5, 0.5, 0.5)" in source
    assert "setattr(scene, GENERATED_GRAY_COLOR_PROPERTY, (0.5, 0.5, 0.5))" in source
    assert "size=4" not in source


def test_scene_capture_contains_no_id_property_or_legacy_rgba_fallback():
    source = SCENE_CAPTURE.read_text(encoding="utf-8")

    assert 'getattr(scene, "get"' not in source
    assert "legacy four-component" not in source
    assert "len(values) != 3" in source


def test_projection_alpha_is_a_registered_blender_52_float_property():
    source = SCENE_PROPERTIES.read_text(encoding="utf-8")
    section = source.split('"spine2d_projection_alpha_threshold"', 1)[1]

    assert "bpy.props.FloatProperty(" in section
    assert "default=1.0 / 255.0" in section
    assert "min=0.0" in section
    assert "max=1.0" in section
