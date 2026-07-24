from Blender_to_Spine2D_Mesh_Exporter.application.a1_grouped_camera_projection import (
    _build_grouped_attachment,
    apply_grouped_camera_overlay,
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
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
)


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


def _source_plan(object_id, output_stem, tmp_path, *, sequence_count=0):
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
        sequence_start_frame=0,
        sequence_frame_count=sequence_count,
        sequence_frame_digits=4,
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


def _grouped_plan(tmp_path, *, sequence_count=0):
    return build_grouped_camera_projection_plan(
        (
            _source_plan(
                "Object_A",
                "Object_A",
                tmp_path,
                sequence_count=sequence_count,
            ),
            _source_plan(
                "Object_B",
                "Object_B",
                tmp_path,
                sequence_count=sequence_count,
            ),
        ),
        group_id="all_objects",
        output_stem="Grouped",
    )


def _concave_layout(*, frame_count=1):
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
        frame_count=frame_count,
        visible_pixel_count=8,
        contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    )


def _source_sequence_attachment(name):
    return MeshAttachment(
        name=name,
        path=f"images/{name}_",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(-50.0, 50.0, 50.0, 50.0, -50.0, -50.0),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
        width=100.0,
        height=100.0,
        sequence={"count": 3, "start": 0, "digits": 4, "setup": 1},
    )


def _connected_source_document():
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(
            Slot("Object_A_slot", "root", attachment="Object_A"),
            Slot("Object_B_slot", "root", attachment="Object_B"),
        ),
        skins=(
            Skin(
                name="default",
                attachments={
                    "Object_A_slot": {
                        "Object_A": _source_sequence_attachment("Object_A")
                    },
                    "Object_B_slot": {
                        "Object_B": _source_sequence_attachment("Object_B")
                    },
                },
            ),
        ),
        animations={
            "animation": {
                "attachments": {
                    "default": {
                        "Object_A_slot": {
                            "Object_A": {
                                "sequence": [
                                    {"mode": "loop", "delay": 0.0333},
                                ]
                            }
                        },
                        "Object_B_slot": {
                            "Object_B": {
                                "sequence": [
                                    {"mode": "loop", "delay": 0.0333},
                                ]
                            }
                        },
                    }
                }
            }
        },
    )


def test_grouped_concave_attachment_serializes_one_convex_hull_and_even_edges(tmp_path):
    grouped = _grouped_plan(tmp_path)
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


def test_grouped_sequence_overlay_replaces_hidden_source_timelines(tmp_path):
    result = apply_grouped_camera_overlay(
        _connected_source_document(),
        _grouped_plan(tmp_path, sequence_count=3),
        _concave_layout(frame_count=3),
        visual_slot_names=("Object_A_slot", "Object_B_slot"),
        image_relative_directory="images",
        slot_name="all_objects_grouped_camera_slot",
        attachment_name="all_objects_grouped_camera_attachment",
    )

    animation = result.document.animations["animation"]
    skin_timelines = animation["attachments"]["default"]
    assert set(skin_timelines) == {"all_objects_grouped_camera_slot"}
    timeline = skin_timelines["all_objects_grouped_camera_slot"][
        "all_objects_grouped_camera_attachment"
    ]["sequence"]
    assert timeline == [
        {"mode": "loop", "delay": 0.0333},
        {"time": 0.0333, "mode": "loop", "index": 1},
        {"time": 0.0666, "mode": "loop", "index": 2},
    ]
    hidden_slots = {
        slot.name: slot.color
        for slot in result.document.slots
        if slot.name in {"Object_A_slot", "Object_B_slot"}
    }
    assert hidden_slots == {
        "Object_A_slot": "ffffff00",
        "Object_B_slot": "ffffff00",
    }

    serialized = SpineSerializer().to_dict(result.document)
    serialized_animation = serialized["animations"]["animation"]
    assert set(serialized_animation["attachments"]["default"]) == {
        "all_objects_grouped_camera_slot"
    }
