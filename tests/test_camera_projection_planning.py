from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1DocumentAssemblySettings,
    assemble_a1_camera_projection_document,
    build_camera_projection_quad_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeExecutionResult,
    BakeMode,
    BakePlan,
    BakePlanError,
    BakeSettings,
    CameraBakeSnapshot,
    CameraProjectionPlan,
    ColorManagementSnapshot,
    MaterialAnalysis,
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    SceneBakeContext,
    ShaderNodeSnapshot,
    TextureFormat,
    build_texture_plan,
    requires_camera_projection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    build_legacy_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_z_groups import (
    A1SourceVertexZBinding,
    A1ZGroupAssignmentPlan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import SourceVertexId


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


def _graph(*channels, dependencies=()):
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=tuple(channels),
        dependencies=tuple(dependencies),
    )


def _analysis(*channels, dependencies=()):
    return ObjectMaterialAnalysis(
        source_object_id="Object",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.PROCEDURAL,
                graph=_graph(*channels, dependencies=dependencies),
            ),
        ),
    )


def _settings(tmp_path: Path, *, texture_format=TextureFormat.PNG, sequence=False):
    return BakeSettings(
        width=80,
        height=48,
        output_directory=tmp_path,
        output_stem="Projection",
        texture_format=texture_format,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
        sequence_start_frame=3 if sequence else 0,
        sequence_frame_count=2 if sequence else 0,
        sequence_frame_digits=4,
    )


def _object_context():
    return ObjectBakeContext(
        source_object_id="Object",
        object_type="MESH",
        world_matrix=IDENTITY,
        collection_names=("Collection",),
    )


def _scene_context(*, camera=True):
    return SceneBakeContext(
        scene_name="Scene",
        render_engine="CYCLES",
        analysis_frame=1,
        world=None,
        camera=(
            CameraBakeSnapshot(
                object_id="Camera",
                camera_type="PERSP",
                world_matrix=IDENTITY,
                lens=50.0,
                ortho_scale=6.0,
                clip_start=0.1,
                clip_end=1000.0,
            )
            if camera
            else None
        ),
        lights=(),
        visible_object_ids=("Camera", "Object") if camera else ("Object",),
        shadow_caster_ids=("Object",),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def test_reflection_dependency_selects_camera_projection(tmp_path: Path):
    analysis = _analysis(
        MaterialSemanticChannel.SURFACE_COLOR,
        dependencies=(MaterialDependencyKind.REFLECTION,),
    )

    plan = build_texture_plan(
        analysis,
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert requires_camera_projection(analysis)
    assert isinstance(plan, CameraProjectionPlan)
    assert isinstance(plan, BakePlan)
    assert plan.camera_object_id == "Camera"
    assert plan.frame_tasks[0].output_path == tmp_path / "Projection_Baked.png"


@pytest.mark.parametrize(
    "channel",
    (MaterialSemanticChannel.VOLUME, MaterialSemanticChannel.DISPLACEMENT),
)
def test_volume_and_displacement_select_camera_projection(
    tmp_path: Path,
    channel: MaterialSemanticChannel,
):
    analysis = _analysis(channel)

    plan = build_texture_plan(
        analysis,
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert isinstance(plan, CameraProjectionPlan)


def test_local_surface_keeps_object_bake_plan(tmp_path: Path):
    plan = build_texture_plan(
        _analysis(MaterialSemanticChannel.SURFACE_COLOR),
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert type(plan) is BakePlan


def test_camera_projection_requires_active_camera(tmp_path: Path):
    analysis = _analysis(
        MaterialSemanticChannel.SURFACE_COLOR,
        dependencies=(MaterialDependencyKind.VIEW,),
    )

    with pytest.raises(BakePlanError, match="active scene camera"):
        build_texture_plan(
            analysis,
            _settings(tmp_path),
            object_context=_object_context(),
            scene_context=_scene_context(camera=False),
        )


def test_camera_projection_rejects_jpeg(tmp_path: Path):
    analysis = _analysis(MaterialSemanticChannel.VOLUME)

    with pytest.raises(BakePlanError, match="alpha-capable|transparent"):
        build_texture_plan(
            analysis,
            _settings(tmp_path, texture_format=TextureFormat.JPEG),
            object_context=_object_context(),
            scene_context=_scene_context(),
        )


def test_camera_projection_sequence_uses_existing_output_contract(tmp_path: Path):
    plan = build_texture_plan(
        _analysis(MaterialSemanticChannel.VOLUME),
        _settings(tmp_path, sequence=True),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )

    assert isinstance(plan, CameraProjectionPlan)
    assert tuple(task.timeline_frame for task in plan.frame_tasks) == (3, 4)
    assert tuple(task.image_name for task in plan.frame_tasks) == (
        "Projection_Baked_0003",
        "Projection_Baked_0004",
    )


def _rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="ProjectionRig",
            texture_width=80,
            texture_height=48,
            z_groups=(LegacyZGroup(0.0),),
            main_position_pixels=None,
        )
    )


def _z_groups():
    source = SourceVertexId("Object", 0)
    return A1ZGroupAssignmentPlan(
        source_snapshot_id="Object:source",
        z_index_base=1,
        groups=(LegacyZGroup(0.0),),
        source_bindings=(A1SourceVertexZBinding(source, 1),),
    )


def test_full_frame_projection_quad_has_deterministic_topology(tmp_path: Path):
    plan = build_texture_plan(
        _analysis(MaterialSemanticChannel.VOLUME),
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )
    assert isinstance(plan, CameraProjectionPlan)
    rig = _rig()

    snapshot = build_camera_projection_quad_snapshot(
        plan,
        rig,
        uv_layer_name="SpineBakeUV",
    )

    assert len(snapshot.vertices) == 4
    assert len(snapshot.edges) == 5
    assert len(snapshot.loops) == 6
    assert len(snapshot.faces) == 2
    assert {loop.uv("SpineBakeUV") for loop in snapshot.loops} == {
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    }


def test_camera_projection_document_contains_one_full_frame_mesh(tmp_path: Path):
    plan = build_texture_plan(
        _analysis(MaterialSemanticChannel.VOLUME),
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )
    assert isinstance(plan, CameraProjectionPlan)
    rig = _rig()

    result = assemble_a1_camera_projection_document(
        rig,
        _z_groups(),
        plan,
        A1DocumentAssemblySettings(
            prefix="ProjectionRig",
            uv_layer_name="SpineBakeUV",
            image_path="Projection_Baked",
            attachment_width=80,
            attachment_height=48,
            center_x=0.0,
            center_y=0.0,
        ),
    )

    request = result.projections[0].request
    assert len(request.vertices) == 4
    assert len(request.triangles) == 6
    assert request.hull == 4
    assert request.width == 80
    assert request.height == 48
    assert {vertex.uv for vertex in request.vertices} == {
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    }
    assert {
        (round(vertex.bone_position_pixels[0], 6), round(vertex.bone_position_pixels[1], 6))
        for vertex in request.vertices
    } == {(-40.0, -24.0), (-40.0, 24.0), (40.0, -24.0), (40.0, 24.0)}


def test_execution_result_accepts_camera_projection_plan(tmp_path: Path):
    from Blender_to_Spine2D_Mesh_Exporter.domain.baking import BakeArtifact

    plan = build_texture_plan(
        _analysis(MaterialSemanticChannel.VOLUME),
        _settings(tmp_path),
        object_context=_object_context(),
        scene_context=_scene_context(),
    )
    assert isinstance(plan, CameraProjectionPlan)
    task = plan.frame_tasks[0]

    result = BakeExecutionResult(
        plan=plan,
        artifacts=(
            BakeArtifact(
                task_index=0,
                timeline_frame=None,
                image_name=task.image_name,
                output_path=task.output_path.resolve(strict=False),
                width=80,
                height=48,
            ),
        ),
    )

    assert result.representative_artifact.image_name == "Projection_Baked"
