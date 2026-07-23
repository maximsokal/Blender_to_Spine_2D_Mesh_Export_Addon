from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_scene_capture import (
    _resolve_generated_gray_color,
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


def test_scene_capture_accepts_opaque_rgba():
    assert _resolve_generated_gray_color(
        SimpleNamespace(spine2d_generated_gray_color=(0.2, 0.3, 0.4, 1.0))
    ) == (0.2, 0.3, 0.4, 1.0)


def test_scene_capture_rejects_translucent_rgba():
    with pytest.raises(ValueError, match="opaque generated textures"):
        _resolve_generated_gray_color(
            SimpleNamespace(spine2d_generated_gray_color=(0.2, 0.3, 0.4, 0.5))
        )
