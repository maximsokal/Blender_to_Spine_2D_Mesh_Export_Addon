from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1AttachmentProjectionSettings,
    A1ZGroupAssignmentError,
    A1ZGroupHeightOverride,
    build_a1_z_group_assignment,
    project_triangulated_disk_attachment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    FaceId,
    MeshVertex,
    SourceVertexId,
    VertexId,
    extract_face_subset,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigBuildRequest,
    build_legacy_mesh_attachment,
    build_legacy_rig,
)

from test_geometry_domain import build_square_snapshot
from test_geometry_triangulation import build_polygon_snapshot


def source_with_two_z_groups():
    snapshot = build_square_snapshot()
    vertices = tuple(
        replace(
            vertex,
            position=(
                vertex.position[0],
                vertex.position[1],
                -1.0 if vertex.id.index < 2 else 1.0,
            ),
        )
        for vertex in snapshot.vertices
    )
    return replace(snapshot, vertices=vertices)


def test_source_snapshot_builds_sorted_one_based_z_groups():
    plan = build_a1_z_group_assignment(source_with_two_z_groups())

    assert tuple(group.z_value for group in plan.groups) == (-1.0, 1.0)
    assert tuple(group.height_real_pixels for group in plan.groups) == (None, None)
    assert tuple(binding.z_group_index for binding in plan.source_bindings) == (
        1,
        1,
        2,
        2,
    )
    assert plan.group_index_for_source(SourceVertexId("Cube", 0)) == 1
    assert plan.group_index_for_source(SourceVertexId("Cube", 3)) == 2


def test_height_overrides_are_applied_by_exact_source_z_value():
    plan = build_a1_z_group_assignment(
        source_with_two_z_groups(),
        height_overrides=(
            A1ZGroupHeightOverride(-1.0, -128.0),
            A1ZGroupHeightOverride(1.0, 128.0),
        ),
    )

    assert tuple(group.height_real_pixels for group in plan.groups) == (
        -128.0,
        128.0,
    )


def test_local_reindexing_does_not_change_z_parent_assignment():
    source = source_with_two_z_groups()
    plan = build_a1_z_group_assignment(source)
    derived = extract_face_subset(
        source,
        (FaceId(1),),
        snapshot_id="Cube:face-1",
        object_name="Cube_Segment_1",
    )

    bindings = plan.projection_bindings(derived)

    assert tuple(binding.vertex_id for binding in bindings) == (
        VertexId(0),
        VertexId(1),
        VertexId(2),
    )
    # Face 1 contains original/source vertices 0, 2, 3.
    assert tuple(binding.z_group_index for binding in bindings) == (1, 2, 2)


def test_z_plan_drives_rig_and_attachment_without_coordinate_lookup():
    source = build_square_snapshot()
    z_plan = build_a1_z_group_assignment(
        source,
        height_overrides=(A1ZGroupHeightOverride(0.0, 0.0),),
    )
    rig = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Cube",
            texture_width=100,
            texture_height=100,
            z_groups=z_plan.groups,
        )
    )
    projection = project_triangulated_disk_attachment(
        source,
        rig,
        A1AttachmentProjectionSettings(
            slot_name="Cube_Segment_0",
            attachment_name="Cube_Segment_0",
            vertex_prefix="Cube_Segment_0",
            image_path="images/Cube_Baked",
            uv_layer_name="UVMap",
            attachment_width=100,
            attachment_height=100,
            center_x=0.5,
            center_y=0.5,
            z_bindings=z_plan.projection_bindings(source),
        ),
    )
    result = build_legacy_mesh_attachment(rig, projection.request)

    assert all(bone.parent == "Cube_1" for bone in result.vertex_bones)


def test_unknown_height_override_is_rejected():
    with pytest.raises(A1ZGroupAssignmentError, match="absent from source snapshot"):
        build_a1_z_group_assignment(
            source_with_two_z_groups(),
            height_overrides=(A1ZGroupHeightOverride(99.0, 100.0),),
        )


def test_derived_snapshot_from_another_source_is_rejected():
    plan = build_a1_z_group_assignment(build_square_snapshot())
    other = build_polygon_snapshot(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        name="Other",
    )

    with pytest.raises(A1ZGroupAssignmentError, match="without source Z assignment"):
        plan.projection_bindings(other)


def test_duplicate_height_override_is_rejected():
    with pytest.raises(ValueError, match="duplicate z_value"):
        build_a1_z_group_assignment(
            build_square_snapshot(),
            height_overrides=(
                A1ZGroupHeightOverride(0.0, 0.0),
                A1ZGroupHeightOverride(0.0, 10.0),
            ),
        )
