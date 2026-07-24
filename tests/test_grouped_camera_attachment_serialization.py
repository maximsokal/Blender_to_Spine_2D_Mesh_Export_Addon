from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.application.a1_grouped_camera_projection import (
    _build_grouped_attachment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeSettings,
    CameraBakeSnapshot,
    CameraProjectionLayout,
    ColorManagementSnapshot,
    MaterialAnalysis,
    MaterialKind,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    ProjectionContourMode,
    ProjectionCropBounds,
    ProjectionPixelPoint,
    SceneBakeContext,
    build_camera_projection_plan,
    build_grouped_camera_projection_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import SpineSerializer


IDENTITY_MATRIX = (
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


def _scene_context():
    return SceneBakeContext(
        scene_name="Scene",
        render_engine="BLENDER_EEVEE",
        analysis_frame=0,
        world=None,
        camera=CameraBakeSnapshot(
            object_id="Camera",
            camera_type="ORTHO",
            world_matrix=IDENTITY_MATRIX,
            lens=50.0,
            ortho_scale=4.0,
            clip_start=0.1,
            clip_end=100.0,
        ),
        lights=(),
        visible_object_ids=("Object_A", "Object_B"),
        shadow_caster_ids=(),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="None",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def _source_plan(object_id, output_stem, tmp_path):
    analysis = ObjectMaterialAnalysis(
        object_id,
        (
            MaterialAnalysis(
                slot_index=0,
                material_name=f"{object_id}_Material",
                kind=MaterialKind.SOLID_COLOR,
                node_types=("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"),
            ),
        ),
    )
    settings = BakeSettings(
        width=4,
        height=4,
        output_directory=tmp_path,
        output_stem=output_stem,
    )
    object_context = ObjectBakeContext(
        source_object_id=object_id,
        object_type="MESH",
        world_matrix=IDENTITY_MATRIX,
    )
    return build_camera_projection_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=_scene_context(),
    )


def _concave_layout():
    return CameraProjectionLayout(
        full_width=4,
        full_height=4,
        crop=ProjectionCropBounds(0, 0, 4, 4),
        hull=(
            ProjectionPixelPoint(0, 0),
            ProjectionPixelPoint(4, 0),
            ProjectionPixelPoint(4, 4),
            ProjectionPixelPoint(2, 2),
            ProjectionPixelPoint(0, 4),
        ),
        alpha_threshold=0.01,
        padding_pixels=0,
        frame_count=1,
        visible_pixel_count=8,
        contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    )


def test_grouped_concave_attachment_serializes_one_convex_hull_and_even_edges(tmp_path):
    grouped = build_grouped_camera_projection_plan(
        (
            _source_plan("Object_A", "Object_A", tmp_path),
            _source_plan("Object_B", "Object_B", tmp_path),
        ),
        group_id="all_objects",
        output_stem="Grouped",
    )
    attachment = _build_grouped_attachment(
        grouped,
        _concave_layout(),
        attachment_name="all_objects_grouped_camera_attachment",
        image_relative_directory="images",
    )

    assert attachment.hull == 4
    assert len(attachment.uvs) == 10
    assert len(attachment.vertices) == 10
    assert max(attachment.triangles) == 4
    assert any(index == 4 for index in attachment.triangles)
    assert max(attachment.edges) == 4

    serialized = SpineSerializer().attachment_to_dict(attachment)

    assert serialized["hull"] == 4
    assert serialized["edges"] == [index * 2 for index in attachment.edges]
    assert all(index % 2 == 0 for index in serialized["edges"])
    assert max(serialized["edges"]) == 8
    assert serialized["spine2dStaticFlattening"] is True
    assert serialized["spine2dSourceContourVertexCount"] == 5
    assert serialized["spine2dConvexHullVertexCount"] == 4
