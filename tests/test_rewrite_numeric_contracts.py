from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionSettings,
    A1AttachmentVertexKey,
    A1MultiObjectExportSettings,
    A1SingleObjectExportSettings,
    A1SourceVertexZBinding,
    A1VertexZBinding,
    A1ZGroupAssignmentPlan,
    A1ZGroupHeightOverride,
    ExportSettings,
    build_a1_z_group_assignment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeArtifact,
    BakeCompositePlan,
    BakeEvaluationScope,
    BakeFrameTask,
    BakeMode,
    BakePassPlan,
    BakeSettings,
    BakeStrategyId,
    CameraBakeSnapshot,
    ColorManagementSnapshot,
    ImageDependency,
    LightBakeSnapshot,
    MaterialAnalysis,
    MaterialKind,
    MaterialPreparationMode,
    MaterialSemanticChannel,
    MaterialSlotPreparation,
    ObjectBakeContext,
    ProjectionCoverageMode,
    ProjectionCoverageResult,
    SceneBakeContext,
    TextureFormat,
    WorldBakeSnapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import SourceVertexId, VertexId
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import LegacyZGroup

from test_geometry_domain import build_square_snapshot


def _matrix(*, first=1.0):
    return (
        first,
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


def _export_settings(tmp_path: Path, **overrides) -> ExportSettings:
    values = {
        "texture_width": 256,
        "texture_height": 256,
        "output_directory": tmp_path,
    }
    values.update(overrides)
    return ExportSettings(**values)


def _bake_settings(tmp_path: Path, **overrides) -> BakeSettings:
    values = {
        "width": 256,
        "height": 256,
        "output_directory": tmp_path,
        "output_stem": "Cube",
    }
    values.update(overrides)
    return BakeSettings(**values)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("width", True),
        ("height", False),
        ("margin_pixels", True),
        ("sequence_start_frame", False),
        ("sequence_frame_count", True),
        ("sequence_frame_digits", False),
    ),
)
def test_bake_settings_reject_bool_for_integer_fields(tmp_path, field_name, value):
    with pytest.raises((TypeError, ValueError)):
        _bake_settings(tmp_path, **{field_name: value})


def test_bake_settings_reject_bool_for_numeric_cage(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        _bake_settings(tmp_path, cage_extrusion=True)


@pytest.mark.parametrize(
    "factory",
    (
        lambda path: ImageDependency("Image", "FILE", frame_duration=True),
        lambda path: MaterialAnalysis(True, None, MaterialKind.EMPTY),
        lambda path: BakeFrameTask(False, None, "Image", path / "image.png"),
        lambda path: MaterialSlotPreparation(True, MaterialPreparationMode.PRESERVE),
        lambda path: BakePassPlan(
            pass_index=True,
            strategy_id=BakeStrategyId.SURFACE_COLOR,
            bake_mode=BakeMode.DIFFUSE,
            material_slot_indices=(0,),
            semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
            evaluation_scope=BakeEvaluationScope.LOCAL,
        ),
        lambda path: BakePassPlan(
            pass_index=0,
            strategy_id=BakeStrategyId.SURFACE_COLOR,
            bake_mode=BakeMode.DIFFUSE,
            material_slot_indices=(False,),
            semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
        ),
        lambda path: BakeCompositePlan(color_pass_indices=(False,)),
        lambda path: BakeArtifact(False, None, "Image", path / "image.png", 1, 1),
    ),
)
def test_baking_contracts_reject_bool_as_integer(tmp_path, factory):
    with pytest.raises((TypeError, ValueError)):
        factory(tmp_path)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ObjectBakeContext("Cube", "MESH", _matrix(first=True)),
        lambda: WorldBakeSnapshot("World", (True, 0.0, 0.0), False),
        lambda: WorldBakeSnapshot(
            "World", (0.0, 0.0, 0.0), False, background_strength=True
        ),
        lambda: LightBakeSnapshot(
            "Light", "POINT", True, (1.0, 1.0, 1.0), _matrix()
        ),
        lambda: CameraBakeSnapshot(
            "Camera", "PERSP", _matrix(), True, 1.0, 0.1, 100.0
        ),
        lambda: ColorManagementSnapshot("Standard", "", True, 1.0),
    ),
)
def test_baking_context_rejects_bool_as_number(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_scene_context_rejects_bool_analysis_frame():
    with pytest.raises((TypeError, ValueError)):
        SceneBakeContext(
            scene_name="Scene",
            render_engine="CYCLES",
            analysis_frame=True,
            world=None,
            camera=None,
            lights=(),
            visible_object_ids=(),
            shadow_caster_ids=(),
            color_management=ColorManagementSnapshot("Standard", "", 0.0, 1.0),
        )


def test_projection_coverage_result_rejects_bool_counter():
    with pytest.raises((TypeError, ValueError)):
        ProjectionCoverageResult(
            mask=b"\x01",
            mode=ProjectionCoverageMode.BINARY_THRESHOLD,
            visible_pixel_count=True,
            raw_nonzero_pixel_count=1,
            strong_pixel_count=1,
            component_count_before_cleanup=1,
            component_count_after_cleanup=1,
            removed_component_pixel_count=0,
            filled_hole_pixel_count=0,
            used_weak_only_fallback=False,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("texture_width", True),
        ("texture_height", False),
        ("angle_limit_degrees", True),
        ("bake_margin", False),
        ("sequence_start_frame", True),
        ("sequence_frame_count", False),
    ),
)
def test_export_settings_reject_bool_numeric_fields(
    tmp_path, field_name, value
):
    with pytest.raises((TypeError, ValueError)):
        _export_settings(tmp_path, **{field_name: value})


def test_single_object_settings_reject_bool_numeric_fields(tmp_path):
    export = _export_settings(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        A1SingleObjectExportSettings(export=export, cage_extrusion=True)
    with pytest.raises((TypeError, ValueError)):
        A1SingleObjectExportSettings(export=export, json_indent=True)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("json_indent", True),
        ("z_tolerance", False),
    ),
)
def test_multi_object_settings_reject_bool_numeric_fields(
    tmp_path, field_name, value
):
    with pytest.raises((TypeError, ValueError)):
        A1MultiObjectExportSettings(
            output_directory=tmp_path,
            output_stem="Group",
            **{field_name: value},
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("connected_group_prefix", " all_objects"),
        ("connected_group_prefix", "all_objects "),
        ("anchor_component_id", " Cube"),
        ("anchor_component_id", "Cube "),
    ),
)
def test_multi_object_identity_fields_reject_boundary_whitespace(
    tmp_path, field_name, value
):
    with pytest.raises(ValueError, match="whitespace"):
        A1MultiObjectExportSettings(
            output_directory=tmp_path,
            output_stem="Group",
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: A1ZGroupHeightOverride(True, 1.0),
        lambda: A1ZGroupHeightOverride(0.0, False),
        lambda: A1SourceVertexZBinding(SourceVertexId("Cube", 0), True),
        lambda: A1ZGroupAssignmentPlan(
            source_snapshot_id="Cube",
            z_index_base=True,
            groups=(LegacyZGroup(0.0),),
            source_bindings=(
                A1SourceVertexZBinding(SourceVertexId("Cube", 0), 1),
            ),
        ),
    ),
)
def test_z_group_contracts_reject_bool_numeric_fields(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_z_group_builder_rejects_bool_index_base():
    with pytest.raises((TypeError, ValueError)):
        build_a1_z_group_assignment(build_square_snapshot(), z_index_base=True)


def test_z_group_snapshot_identity_is_canonical():
    with pytest.raises(ValueError, match="whitespace"):
        A1ZGroupAssignmentPlan(
            source_snapshot_id=" Cube",
            z_index_base=1,
            groups=(LegacyZGroup(0.0),),
            source_bindings=(
                A1SourceVertexZBinding(SourceVertexId("Cube", 0), 1),
            ),
        )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: A1VertexZBinding(VertexId(0), True),
        lambda: A1AttachmentVertexKey(VertexId(0), (True, 0.5)),
        lambda: A1AttachmentProjectionSettings(
            slot_name="slot",
            attachment_name="attachment",
            vertex_prefix="vertex",
            image_path="images/Cube",
            uv_layer_name="UVMap",
            attachment_width=True,
            attachment_height=64.0,
            center_x=0.0,
            center_y=0.0,
            z_bindings=(A1VertexZBinding(VertexId(0), 0),),
        ),
        lambda: A1AttachmentProjectionSettings(
            slot_name="slot",
            attachment_name="attachment",
            vertex_prefix="vertex",
            image_path="images/Cube",
            uv_layer_name="UVMap",
            attachment_width=64.0,
            attachment_height=64.0,
            center_x=False,
            center_y=0.0,
            z_bindings=(A1VertexZBinding(VertexId(0), 0),),
        ),
    ),
)
def test_attachment_projection_rejects_bool_numeric_fields(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_attachment_projection_identity_fields_are_canonical():
    with pytest.raises(ValueError, match="whitespace"):
        A1AttachmentProjectionSettings(
            slot_name=" slot",
            attachment_name="attachment",
            vertex_prefix="vertex",
            image_path="images/Cube",
            uv_layer_name="UVMap",
            attachment_width=64.0,
            attachment_height=64.0,
            center_x=0.0,
            center_y=0.0,
            z_bindings=(A1VertexZBinding(VertexId(0), 0),),
        )


def test_valid_contract_values_remain_accepted(tmp_path):
    bake = _bake_settings(
        tmp_path,
        width=128,
        height=64,
        margin_pixels=0,
        cage_extrusion=0.0,
        sequence_start_frame=0,
        sequence_frame_count=0,
        sequence_frame_digits=4,
        texture_format=TextureFormat.PNG,
    )
    multi = A1MultiObjectExportSettings(
        output_directory=tmp_path,
        output_stem="Group",
        connected_group_prefix="all_objects",
        anchor_component_id="Cube",
        json_indent=2,
        z_tolerance=1e-4,
    )

    assert bake.width == 128
    assert bake.cage_extrusion == 0.0
    assert multi.connected_group_prefix == "all_objects"
    assert multi.anchor_component_id == "Cube"
