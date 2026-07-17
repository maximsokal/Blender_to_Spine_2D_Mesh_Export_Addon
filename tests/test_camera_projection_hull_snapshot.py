from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.application import (
    build_camera_projection_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMode,
    BakeSettings,
    CameraBakeSnapshot,
    CameraProjectionLayout,
    CameraProjectionPlan,
    ColorManagementSnapshot,
    MaterialAnalysis,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectBakeContext,
    ObjectMaterialAnalysis,
    ProjectionCropBounds,
    ProjectionPixelPoint,
    SceneBakeContext,
    ShaderNodeSnapshot,
    TextureFormat,
    build_texture_plan,
    convex_hull,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    LegacyZGroup,
    build_legacy_rig,
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


def _projection_plan(tmp_path: Path) -> CameraProjectionPlan:
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    graph = MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=(MaterialSemanticChannel.VOLUME,),
        dependencies=(),
    )
    analysis = ObjectMaterialAnalysis(
        source_object_id="Object",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.PROCEDURAL,
                graph=graph,
            ),
        ),
    )
    settings = BakeSettings(
        width=80,
        height=48,
        output_directory=tmp_path,
        output_stem="Projection",
        texture_format=TextureFormat.PNG,
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.DIFFUSE,
    )
    object_context = ObjectBakeContext(
        source_object_id="Object",
        object_type="MESH",
        world_matrix=IDENTITY,
        collection_names=("Collection",),
    )
    scene_context = SceneBakeContext(
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
        visible_object_ids=("Camera", "Object"),
        shadow_caster_ids=("Object",),
        color_management=ColorManagementSnapshot(
            view_transform="Standard",
            look="",
            exposure=0.0,
            gamma=1.0,
        ),
    )
    plan = build_texture_plan(
        analysis,
        settings,
        object_context=object_context,
        scene_context=scene_context,
    )
    assert isinstance(plan, CameraProjectionPlan)
    return plan


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


def test_irregular_projection_snapshot_uses_layout_triangle_contract(tmp_path: Path):
    plan = _projection_plan(tmp_path)
    rig = _rig()
    hull = convex_hull(
        (
            ProjectionPixelPoint(10, 10),
            ProjectionPixelPoint(28, 6),
            ProjectionPixelPoint(52, 11),
            ProjectionPixelPoint(62, 28),
            ProjectionPixelPoint(43, 41),
            ProjectionPixelPoint(16, 36),
        )
    )
    layout = CameraProjectionLayout(
        full_width=80,
        full_height=48,
        crop=ProjectionCropBounds(8, 4, 65, 44),
        hull=hull,
        alpha_threshold=1.0 / 255.0,
        padding_pixels=2,
        frame_count=1,
        visible_pixel_count=700,
    )

    snapshot = build_camera_projection_mesh_snapshot(
        plan,
        rig,
        uv_layer_name="SpineBakeUV",
        layout=layout,
    )
    loop_map = snapshot.loop_by_id()
    face_vertex_indices = tuple(
        tuple(loop_map[loop_id].vertex_id.index for loop_id in face.loop_ids)
        for face in snapshot.faces
    )

    assert face_vertex_indices == layout.triangle_indices
    assert len(snapshot.vertices) == len(layout.hull)
    assert len(snapshot.faces) == len(layout.hull) - 2
    assert len(snapshot.loops) == (len(layout.hull) - 2) * 3
    assert len(snapshot.edges) == len(layout.hull) + len(layout.hull) - 3
    assert tuple(vertex.position[:2] for vertex in snapshot.vertices) == tuple(
        (
            layout.spine_position_pixels(point)[0] / rig.info.uniform_scale,
            layout.spine_position_pixels(point)[1] / rig.info.uniform_scale,
        )
        for point in layout.hull
    )
    assert {
        loop.uv("SpineBakeUV") for loop in snapshot.loops
    } == {layout.spine_uv(point) for point in layout.hull}
