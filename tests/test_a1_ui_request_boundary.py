from types import SimpleNamespace

import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_bridge as bridge
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import A1TextureExportMode
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    A1RigSetupPoseMode,
)


class _RnaObject:
    def __init__(self, name: str, pointer: int):
        self.name = name
        self.name_full = name
        self.type = "MESH"
        self.data = object()
        self._pointer = pointer

    def as_pointer(self) -> int:
        return self._pointer


def _scene(
    rig_profile: A1RigProfile = A1RigProfile.TWO_AXIS_ROTATION_SCALE,
):
    return SimpleNamespace(
        spine2d_texture_export_mode=(
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value
        ),
        spine2d_rig_profile=rig_profile.value,
        spine2d_seam_maker_mode="AUTO",
        spine2d_angle_limit=30.0,
        spine2d_control_icons=True,
        # Legacy .blend files may still persist this value. The target-version
        # setup-pose pipeline must ignore it until animation versioning is explicit.
        spine2d_export_preview_animation=True,
        spine2d_bake_frame_start=0,
        spine2d_frames_for_render=0,
        render=SimpleNamespace(engine="CYCLES"),
    )


def _object(name: str, *, frame_start: int = 0, frame_count: int = 0):
    return SimpleNamespace(
        name=name,
        name_full=name,
        spine2d_bake_settings=SimpleNamespace(
            bake_frame_start=frame_start,
            frames_for_render=frame_count,
        ),
    )


def test_bridge_routes_multi_and_mixed_to_post_render_output_services():
    assert bridge.export_a1_multi_object.__module__.endswith(
        ".a1_multi_object_output"
    )
    assert bridge.export_a1_mixed_object.__module__.endswith(
        ".a1_mixed_object_output"
    )
    assert bridge.export_a1_single_object.__module__.endswith(
        ".a1_single_object_export"
    )


def test_single_settings_request_a_neutral_authoring_setup_pose(tmp_path):
    settings = bridge._build_single_object_settings(
        _object("Hero"),
        _scene(),
        output_directory=tmp_path,
        texture_size=128,
        images_relative_path="images",
    )

    assert settings.export.rig_profile == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    assert settings.rig_setup_pose_mode is A1RigSetupPoseMode.NORMALIZED_SINGLE
    assert settings.include_preview_animation is False


def test_multi_sources_share_one_immutable_scene_snapshot(tmp_path):
    sources = bridge._build_sources(
        (
            _object("A"),
            _object("B", frame_start=4, frame_count=3),
        ),
        _scene(A1RigProfile.TWO_AXIS_ROTATION_SCALE),
        output_directory=tmp_path,
        texture_size=128,
        images_relative_path="images",
    )

    assert len(sources) == 2
    assert sources[0].settings.geometry is sources[1].settings.geometry
    assert (
        sources[0].settings.bake_execution
        is sources[1].settings.bake_execution
    )
    assert (
        sources[0].settings.bake_execution.texture_export_mode
        is A1TextureExportMode.NORMAL_UV_SEGMENTS
    )
    assert (
        sources[0].settings.export.rig_profile
        == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    )
    assert (
        sources[1].settings.export.rig_profile
        == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    )
    assert all(
        source.settings.rig_setup_pose_mode
        is A1RigSetupPoseMode.PRESERVE_COMPOSITION
        for source in sources
    )
    assert all(
        source.settings.include_preview_animation is False
        for source in sources
    )
    assert sources[1].settings.export.sequence_start_frame == 4
    assert sources[1].settings.export.sequence_frame_count == 3


def test_selected_order_matches_active_object_by_stable_rna_identity():
    selected_active_wrapper = _RnaObject("Active", 101)
    context_active_wrapper = _RnaObject("Active", 101)
    other = _RnaObject("Alpha", 202)
    duplicate_active_wrapper = _RnaObject("Active", 101)
    context = SimpleNamespace(
        active_object=context_active_wrapper,
        selected_objects=(other, selected_active_wrapper, duplicate_active_wrapper),
    )

    ordered = bridge._ordered_selected_meshes(context)

    assert tuple(item.name for item in ordered) == ("Active", "Alpha")
    assert ordered[0] is selected_active_wrapper


def test_component_ids_and_animation_namespaces_remain_deterministic(tmp_path):
    sources = bridge._build_sources(
        (_object("Hero"), _object("Weapon")),
        _scene(),
        output_directory=tmp_path,
        texture_size=64,
        images_relative_path="images",
    )

    assert tuple(source.component_id for source in sources) == (
        "object_1:Hero",
        "object_2:Weapon",
    )
    assert tuple(source.animation_namespace for source in sources) == (
        "object_1",
        "object_2",
    )
