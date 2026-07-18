from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    ConnectedB4RenderPolicy,
    apply_grouped_camera_overlay,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_multi_object import (
    A1MultiObjectExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.grouped_camera_projection_executor import (
    _configure_group_camera_visibility,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMode,
    BakeSettings,
    CameraBakeSnapshot,
    CameraProjectionLayout,
    CameraProjectionPlan,
    ColorManagementSnapshot,
    GroupedCameraProjectionPlanError,
    MaterialAnalysis,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    ProjectionContourMode,
    ProjectionCropBounds,
    ProjectionPixelPoint,
    SceneBakeContext,
    ShaderNodeSnapshot,
    TextureFormat,
    build_grouped_camera_projection_plan,
    build_texture_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineValidator,
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


def _analysis(source_id):
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    graph = MaterialGraphSnapshot(
        material_name=f"Material_{source_id}",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=(MaterialSemanticChannel.VOLUME,),
        dependencies=(),
    )
    return ObjectMaterialAnalysis(
        source_object_id=source_id,
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name=graph.material_name,
                kind=MaterialKind.PROCEDURAL,
                graph=graph,
            ),
        ),
    )


def _scene_context():
    return SceneBakeContext(
        scene_name="Scene",
        render_engine="CYCLES",
        analysis_frame=1,
        world=None,
        camera=CameraBakeSnapshot(
            object_id="Camera",
            camera_type="PERSP",
            world_matrix=IDENTITY,
            lens=50.0,
            ortho_scale=6.0,
            clip_start=0.1,
            clip_end=1000.0,
        ),
        lights=(),
        visible_object_ids=("Camera", "ObjectA", "ObjectB"),
        shadow_caster_ids=("ObjectA", "ObjectB"),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def _plan(tmp_path: Path, source_id: str, *, frame_count=0):
    plan = build_texture_plan(
        _analysis(source_id),
        BakeSettings(
            width=80,
            height=48,
            output_directory=tmp_path / "images",
            output_stem=source_id,
            texture_format=TextureFormat.PNG,
            diffuse_mode=BakeMode.DIFFUSE,
            procedural_mode=BakeMode.DIFFUSE,
            sequence_start_frame=3,
            sequence_frame_count=frame_count,
            sequence_frame_digits=4,
        ),
        object_context=ObjectBakeContext(
            source_object_id=source_id,
            object_type="MESH",
            world_matrix=IDENTITY,
            collection_names=("Collection",),
        ),
        scene_context=_scene_context(),
    )
    assert isinstance(plan, CameraProjectionPlan)
    return plan


def _source_attachment(name):
    return MeshAttachment(
        name=name,
        uvs=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
        triangles=(0, 1, 2, 0, 2, 3),
        vertices=(-10.0, -10.0, 10.0, -10.0, 10.0, 10.0, -10.0, 10.0),
        hull=4,
        path=f"images/{name}",
        width=20.0,
        height=20.0,
    )


def _document():
    attachment_a = _source_attachment("A")
    attachment_b = _source_attachment("B")
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"), Bone("A_bone", parent="root"), Bone("B_bone", parent="root")),
        slots=(
            Slot("A_slot", "A_bone", attachment="A"),
            Slot("B_slot", "B_bone", attachment="B"),
        ),
        skins=(
            Skin(
                "default",
                {
                    "A_slot": {"A": attachment_a},
                    "B_slot": {"B": attachment_b},
                },
            ),
        ),
        animations={"preview": {"bones": {"A_bone": {"rotate": []}}}},
    )


def _layout(frame_count=1):
    return CameraProjectionLayout(
        full_width=80,
        full_height=48,
        crop=ProjectionCropBounds(8, 6, 62, 43),
        hull=(
            ProjectionPixelPoint(10, 8),
            ProjectionPixelPoint(58, 8),
            ProjectionPixelPoint(58, 20),
            ProjectionPixelPoint(34, 20),
            ProjectionPixelPoint(34, 40),
            ProjectionPixelPoint(10, 40),
        ),
        alpha_threshold=1.0 / 255.0,
        padding_pixels=2,
        frame_count=frame_count,
        visible_pixel_count=900,
        contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
        source_contour_vertex_count=6,
        simplify_tolerance_pixels=1.0,
    )


def test_grouped_plan_requires_matching_camera_frame_and_output_contract(tmp_path):
    first = _plan(tmp_path, "ObjectA", frame_count=3)
    second = _plan(tmp_path, "ObjectB", frame_count=3)

    grouped = build_grouped_camera_projection_plan(
        (first, second),
        group_id="all_objects",
        output_stem="Combined_grouped_camera",
    )

    assert grouped.source_object_ids == ("ObjectA", "ObjectB")
    assert grouped.camera_object_id == "Camera"
    assert tuple(task.timeline_frame for task in grouped.frame_tasks) == (3, 4, 5)
    assert all("Combined_grouped_camera" in task.image_name for task in grouped.frame_tasks)
    assert not any(
        grouped_task.output_path == source_task.output_path
        for grouped_task in grouped.frame_tasks
        for plan in (first, second)
        for source_task in plan.frame_tasks
    )


def test_grouped_plan_rejects_different_frame_ranges(tmp_path):
    first = _plan(tmp_path, "ObjectA", frame_count=3)
    second = _plan(tmp_path, "ObjectB", frame_count=2)

    with pytest.raises(GroupedCameraProjectionPlanError, match="render contract"):
        build_grouped_camera_projection_plan(
            (first, second),
            group_id="all_objects",
            output_stem="Combined_grouped_camera",
        )


def test_grouped_overlay_hides_source_slots_and_adds_root_bound_mesh(tmp_path):
    grouped = build_grouped_camera_projection_plan(
        (_plan(tmp_path, "ObjectA"), _plan(tmp_path, "ObjectB")),
        group_id="all_objects",
        output_stem="Combined_grouped_camera",
    )

    result = apply_grouped_camera_overlay(
        _document(),
        grouped,
        _layout(),
        visual_slot_names=("A_slot", "B_slot"),
        image_relative_directory="images",
        slot_name="all_objects_grouped_camera_slot",
        attachment_name="all_objects_grouped_camera_attachment",
    )

    SpineValidator().validate_or_raise(result.document)
    slots = {slot.name: slot for slot in result.document.slots}
    assert slots["A_slot"].color == "ffffff00"
    assert slots["B_slot"].color == "ffffff00"
    assert slots[result.slot_name].bone == "root"
    assert slots[result.slot_name].attachment == result.attachment_name
    assert result.document.animations == _document().animations

    default_skin = next(skin for skin in result.document.skins if skin.name == "default")
    attachment = default_skin.attachments[result.slot_name][result.attachment_name]
    assert isinstance(attachment, MeshAttachment)
    assert attachment.hull == 6
    assert len(attachment.vertices) == 12
    assert len(attachment.uvs) == 12
    assert len(attachment.triangles) == 12
    assert attachment.path == "images/Combined_grouped_camera_Baked"
    assert attachment.extras["spine2dGroupedCamera"] is True


def test_grouped_sequence_overlay_uses_sequence_base_path_and_metadata(tmp_path):
    grouped = build_grouped_camera_projection_plan(
        (
            _plan(tmp_path, "ObjectA", frame_count=3),
            _plan(tmp_path, "ObjectB", frame_count=3),
        ),
        group_id="all_objects",
        output_stem="Combined_grouped_camera",
    )
    result = apply_grouped_camera_overlay(
        _document(),
        grouped,
        _layout(frame_count=3),
        visual_slot_names=("A_slot", "B_slot"),
        image_relative_directory="images",
        slot_name="group_slot",
        attachment_name="group_attachment",
    )
    skin = next(skin for skin in result.document.skins if skin.name == "default")
    attachment = skin.attachments["group_slot"]["group_attachment"]

    assert attachment.path == "images/Combined_grouped_camera_Baked_"
    assert attachment.sequence == {"count": 3, "start": 3, "digits": 4}


def test_group_visibility_keeps_sources_visible_and_only_hides_other_camera_rays():
    first = SimpleNamespace(name="ObjectA", type="MESH", hide_render=True, visible_camera=False)
    second = SimpleNamespace(name="ObjectB", type="MESH", hide_render=True, visible_camera=False)
    dependency = SimpleNamespace(
        name="Reflector",
        type="MESH",
        hide_render=False,
        visible_camera=True,
    )
    light = SimpleNamespace(name="Light", type="LIGHT", hide_render=False)
    scene = SimpleNamespace(objects=(first, second, dependency, light))

    _configure_group_camera_visibility((first, second), scene)

    assert not first.hide_render and first.visible_camera
    assert not second.hide_render and second.visible_camera
    assert not dependency.visible_camera
    assert not dependency.hide_render
    assert not hasattr(light, "visible_camera")


def test_multi_settings_grouped_policy_is_typed(tmp_path):
    settings = A1MultiObjectExportSettings(
        output_directory=tmp_path,
        output_stem="Combined",
    )
    assert (
        settings.connected_b4_render_policy
        is ConnectedB4RenderPolicy.AUTO_GROUPED_CAMERA
    )
    with pytest.raises(TypeError, match="connected_b4_render_policy"):
        replace(settings, connected_b4_render_policy="AUTO_GROUPED_CAMERA")
