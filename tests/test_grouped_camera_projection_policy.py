from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.grouped_camera_projection_policy as policy_module
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    ConnectedB4RenderPolicy,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.grouped_camera_projection_policy import (
    GroupedCameraProjectionPolicyError,
    resolve_grouped_camera_projection_request,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    CameraBakeSnapshot,
    CameraProjectionPlan,
    ColorManagementSnapshot,
    MaterialAnalysis,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderNodeSnapshot,
    TextureFormat,
    build_texture_plan,
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


class FakePrepared:
    def __init__(
        self,
        object_id,
        bake_plan,
        *,
        execution=None,
        image_relative_directory="images",
        slot_name=None,
    ):
        self.object_id = object_id
        self.bake_plan = bake_plan
        self.settings = SimpleNamespace(
            bake_execution=execution or BakeExecutionSettings()
        )
        self.output_paths = SimpleNamespace(
            image_relative_directory=image_relative_directory
        )
        self.source_object = SimpleNamespace(name=object_id, name_full=object_id)
        self.document_assembly = SimpleNamespace(
            projections=(
                SimpleNamespace(
                    request=SimpleNamespace(
                        slot_name=slot_name or f"{object_id}_slot"
                    )
                ),
            )
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
        visible_object_ids=("Camera", "A", "B"),
        shadow_caster_ids=("A", "B"),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def _camera_plan(tmp_path: Path, object_id):
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    analysis = ObjectMaterialAnalysis(
        object_id,
        (
            MaterialAnalysis(
                0,
                f"Material_{object_id}",
                MaterialKind.PROCEDURAL,
                graph=MaterialGraphSnapshot(
                    material_name=f"Material_{object_id}",
                    active_output_node_id=output.node_id,
                    reachable_nodes=(output,),
                    reachable_links=(),
                    semantic_channels=(MaterialSemanticChannel.VOLUME,),
                    dependencies=(),
                ),
            ),
        ),
    )
    plan = build_texture_plan(
        analysis,
        BakeSettings(
            width=64,
            height=64,
            output_directory=tmp_path / "images",
            output_stem=object_id,
            texture_format=TextureFormat.PNG,
            diffuse_mode=BakeMode.DIFFUSE,
            procedural_mode=BakeMode.DIFFUSE,
        ),
        object_context=ObjectBakeContext(
            source_object_id=object_id,
            object_type="MESH",
            world_matrix=IDENTITY,
            collection_names=("Collection",),
        ),
        scene_context=_scene_context(),
    )
    assert isinstance(plan, CameraProjectionPlan)
    return plan


def _settings(tmp_path, policy=ConnectedB4RenderPolicy.AUTO_GROUPED_CAMERA):
    return A1MultiObjectExportSettings(
        output_directory=tmp_path,
        output_stem="Combined",
        mode=A1MultiObjectMode.CONNECTED,
        connected_b4_render_policy=policy,
    )


def test_auto_groups_only_complete_compatible_b4_set(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_module, "PreparedA1Object", FakePrepared)
    prepared = (
        FakePrepared("A", _camera_plan(tmp_path, "A")),
        FakePrepared("B", _camera_plan(tmp_path, "B")),
    )

    request = resolve_grouped_camera_projection_request(
        prepared,
        _settings(tmp_path),
    )

    assert request is not None
    assert request.plan.source_object_ids == ("A", "B")
    assert request.visual_slot_names == ("A_slot", "B_slot")
    assert request.image_relative_directory == "images"


def test_individual_policy_never_groups(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_module, "PreparedA1Object", FakePrepared)
    prepared = (
        FakePrepared("A", _camera_plan(tmp_path, "A")),
        FakePrepared("B", _camera_plan(tmp_path, "B")),
    )

    assert (
        resolve_grouped_camera_projection_request(
            prepared,
            _settings(tmp_path, ConnectedB4RenderPolicy.INDIVIDUAL_LAYERS),
        )
        is None
    )


def test_auto_falls_back_when_one_connected_component_is_not_b4(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_module, "PreparedA1Object", FakePrepared)
    camera = FakePrepared("A", _camera_plan(tmp_path, "A"))
    local = FakePrepared("B", SimpleNamespace(source_object_id="B"))

    assert (
        resolve_grouped_camera_projection_request(
            (camera, local),
            _settings(tmp_path),
        )
        is None
    )


def test_required_rejects_mixed_b4_and_local_components(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_module, "PreparedA1Object", FakePrepared)
    camera = FakePrepared("A", _camera_plan(tmp_path, "A"))
    local = FakePrepared("B", SimpleNamespace(source_object_id="B"))

    with pytest.raises(GroupedCameraProjectionPolicyError, match="every connected"):
        resolve_grouped_camera_projection_request(
            (camera, local),
            _settings(
                tmp_path,
                ConnectedB4RenderPolicy.GROUPED_CAMERA_REQUIRED,
            ),
        )


def test_auto_falls_back_on_different_execution_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_module, "PreparedA1Object", FakePrepared)
    prepared = (
        FakePrepared("A", _camera_plan(tmp_path, "A")),
        FakePrepared(
            "B",
            _camera_plan(tmp_path, "B"),
            execution=BakeExecutionSettings(samples=32),
        ),
    )

    assert (
        resolve_grouped_camera_projection_request(
            prepared,
            _settings(tmp_path),
        )
        is None
    )


def test_required_rejects_different_image_directories(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_module, "PreparedA1Object", FakePrepared)
    prepared = (
        FakePrepared("A", _camera_plan(tmp_path, "A")),
        FakePrepared(
            "B",
            _camera_plan(tmp_path, "B"),
            image_relative_directory="other_images",
        ),
    )

    with pytest.raises(GroupedCameraProjectionPolicyError, match="image-relative"):
        resolve_grouped_camera_projection_request(
            prepared,
            _settings(
                tmp_path,
                ConnectedB4RenderPolicy.GROUPED_CAMERA_REQUIRED,
            ),
        )
