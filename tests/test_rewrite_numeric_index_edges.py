from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionResult,
    A1AttachmentVertexKey,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeCompositePlan,
    BakeFrameTask,
    BakeMode,
    BakePlan,
    BakeSettings,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
)


def _analysis() -> ObjectMaterialAnalysis:
    return ObjectMaterialAnalysis(
        source_object_id="Cube",
        slots=(MaterialAnalysis(0, "Material", MaterialKind.SOLID_COLOR),),
    )


def _bake_settings(tmp_path: Path) -> BakeSettings:
    return BakeSettings(
        width=64,
        height=64,
        output_directory=tmp_path,
        output_stem="Cube",
    )


def test_bake_frame_task_rejects_bool_timeline_frame(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        BakeFrameTask(
            task_index=0,
            timeline_frame=True,
            image_name="Cube_Baked",
            output_path=tmp_path / "Cube_Baked.png",
        )


def test_bake_plan_rejects_bool_representative_task_index(tmp_path):
    settings = _bake_settings(tmp_path)
    task = BakeFrameTask(
        task_index=0,
        timeline_frame=None,
        image_name="Cube_Baked",
        output_path=tmp_path / "Cube_Baked.png",
    )

    with pytest.raises((TypeError, ValueError)):
        BakePlan(
            source_object_id="Cube",
            settings=settings,
            material_analysis=_analysis(),
            bake_mode=BakeMode.DIFFUSE,
            frame_tasks=(task,),
            representative_task_index=True,
        )


def test_bake_composite_rejects_bool_alpha_pass_index():
    with pytest.raises((TypeError, ValueError)):
        BakeCompositePlan(
            mode=BakeCompositeMode.ADD_RGB_REPLACE_ALPHA,
            color_pass_indices=(0,),
            alpha_pass_index=True,
        )


def test_attachment_projection_result_rejects_bool_attachment_index():
    request = LegacyMeshAttachmentRequest(
        slot_name="slot",
        attachment_name="attachment",
        vertex_prefix="vertex",
        image_path="images/Cube",
        width=64.0,
        height=64.0,
        vertices=(
            LegacyAttachmentVertex(
                index=0,
                uv=(0.0, 0.0),
                bone_position_pixels=(0.0, 0.0),
                z_group_index=0,
            ),
        ),
        triangles=(),
        hull=1,
    )
    key = A1AttachmentVertexKey(VertexId(0), (0.0, 0.0))

    with pytest.raises((TypeError, ValueError)):
        A1AttachmentProjectionResult(
            request=request,
            hull_vertex_keys=(key,),
            ordered_vertex_keys=(key,),
            loop_to_attachment_index=((LoopId(0), True),),
        )
