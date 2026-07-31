"""Regression coverage for connected TWO_AXIS_ROTATION_SCALE composition."""

from __future__ import annotations

from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigSetupPoseMode,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    MeshAttachment,
    SpineDocument,
    SpineValidator,
    TwoAxisScaleRigProfile,
    build_connected_group_document,
    build_legacy_mesh_document,
    build_two_axis_scale_rig,
    decode_weighted_vertices,
)


ROOT = Path(__file__).resolve().parents[1]
COMPOSITION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_multi_object_composition.py"
)
SETTINGS = ROOT / "docs" / "settings-reference.md"


def _two_axis_document(
    prefix: str,
    main_position: tuple[float, float],
) -> SpineDocument:
    rig = build_two_axis_scale_rig(
        LegacyRigBuildRequest(
            prefix=prefix,
            texture_width=100,
            texture_height=100,
            z_groups=(
                LegacyZGroup(-1.0, height_real_pixels=-50.0),
                LegacyZGroup(1.0, height_real_pixels=50.0),
            ),
            main_position_pixels=main_position,
            setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        )
    )
    slot_name = f"{prefix}_Segment_0"
    attachment = LegacyMeshAttachmentRequest(
        slot_name=slot_name,
        attachment_name=slot_name,
        vertex_prefix=slot_name,
        image_path=f"images/{prefix}_Baked",
        width=100,
        height=100,
        vertices=(
            LegacyAttachmentVertex(0, (0.0, 0.0), (-50.0, 50.0), 1),
            LegacyAttachmentVertex(1, (1.0, 0.0), (50.0, 50.0), 1),
            LegacyAttachmentVertex(2, (0.0, 1.0), (-50.0, -50.0), 2),
        ),
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
    )
    built = build_legacy_mesh_document(
        rig,
        (attachment,),
        skeleton_metadata={
            "spine": "4.2.43",
            "hash": f"{prefix}-connected-two-axis",
            "width": 100,
            "height": 100,
            "images": "",
            "audio": "./audio",
        },
    )
    SpineValidator().validate_or_raise(built.document)
    return built.document


def _objects() -> tuple[ConnectedObjectDocument, ...]:
    return (
        ConnectedObjectDocument(
            component_id="first",
            prefix="First",
            document=_two_axis_document("First", (10.0, 20.0)),
            world_position=(10.0, 20.0, 5.0),
            animation_namespace="object_1",
        ),
        ConnectedObjectDocument(
            component_id="second",
            prefix="Second",
            document=_two_axis_document("Second", (30.0, -10.0)),
            world_position=(12.0, 23.0, 7.0),
            animation_namespace="object_2",
        ),
    )


def _mesh_attachment(document: SpineDocument, slot_name: str) -> MeshAttachment:
    for skin in document.skins:
        slot_attachments = skin.attachments.get(slot_name)
        if slot_attachments is None:
            continue
        attachment = slot_attachments.get(slot_name)
        if isinstance(attachment, MeshAttachment):
            return attachment
    raise AssertionError(f"Mesh attachment {slot_name!r} was not found")


def _constraint_order(document: SpineDocument, name: str) -> int:
    matches = tuple(
        constraint.order
        for constraint in (*document.ik, *document.transform)
        if constraint.name == name
    )
    assert len(matches) == 1, f"Expected one constraint {name!r}, found {len(matches)}"
    return matches[0]


def test_connected_two_axis_builds_global_and_per_object_controls():
    result = build_connected_group_document(
        _objects(),
        ConnectedGroupSettings(
            texture_width=100,
            texture_height=100,
            anchor_component_id="first",
        ),
        profile=TwoAxisScaleRigProfile(),
    )

    SpineValidator().validate_or_raise(result.document)
    bone_names = {bone.name for bone in result.document.bones}
    assert {
        "all_objects_rotation_X",
        "all_objects_rotation_Y",
        "all_objects_scale",
        "First_rotation_X",
        "First_rotation_Y",
        "First_scale",
        "Second_rotation_X",
        "Second_rotation_Y",
        "Second_scale",
    } <= bone_names
    assert "all_objects_rotation_Z" not in bone_names
    assert "First_rotation_Z" not in bone_names
    assert "Second_rotation_Z" not in bone_names

    bones = {bone.name: bone for bone in result.document.bones}
    assert (
        bones["First_main"].parent,
        bones["First_main"].x,
        bones["First_main"].y,
    ) == ("all_objects_layer_1", 10.0, 20.0)
    # The two-axis wrapper owns +200 setup Y for relative Z=2, leaving +90 local Y.
    assert (
        bones["Second_main"].parent,
        bones["Second_main"].x,
        bones["Second_main"].y,
    ) == ("all_objects_layer_0", 230.0, 90.0)


def test_connected_two_axis_schedule_is_layer_grouped_and_semantically_ordered():
    result = build_connected_group_document(
        _objects(),
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=TwoAxisScaleRigProfile(),
    )
    schedule = result.constraint_schedule
    constraints = (*result.document.ik, *result.document.transform)
    orders = tuple(constraint.order for constraint in constraints)

    assert schedule.profile_id == "TWO_AXIS_ROTATION_SCALE"
    assert schedule.unique_orders == tuple(range(15))
    assert len(orders) == len(set(orders)) == 15
    assert set(orders) == set(schedule.unique_orders)
    assert (
        schedule.global_rotation_x,
        schedule.global_scale_ik,
        schedule.global_scale,
        schedule.global_scale_depth,
        schedule.global_rotation_y,
    ) == (0, 1, 2, 3, 4)
    # Assignment tuples preserve source-object order while order values come from layer.
    assert schedule.object_rotation_x == (("first", 6), ("second", 5))
    assert schedule.object_scale_depth == (("first", 12), ("second", 11))
    assert schedule.object_rotation_y == (("first", 14), ("second", 13))
    assert schedule.global_rotation_y < min(
        order for _, order in schedule.object_rotation_x
    )

    for prefix in ("First", "Second"):
        assert (
            _constraint_order(result.document, f"{prefix}_rotation_X_constraint")
            < _constraint_order(result.document, f"{prefix}_IK")
            < _constraint_order(result.document, f"{prefix}_scale")
            < _constraint_order(
                result.document,
                f"{prefix}_scale_rotate_X_constraint",
            )
            < _constraint_order(result.document, f"{prefix}_rotation_Y")
        )

    assert (
        _constraint_order(result.document, "all_objects_rotation_X_constraint")
        < _constraint_order(result.document, "all_objects_IK")
        < _constraint_order(result.document, "all_objects_scale")
        < _constraint_order(
            result.document,
            "all_objects_scale_rotate_X_constraint",
        )
        < _constraint_order(result.document, "all_objects_rotation_Y")
    )
    assert not any(
        constraint.name.endswith("_scale_compensator")
        or constraint.name.endswith("_rotation_Z")
        for constraint in constraints
    )


def test_connected_two_axis_remaps_weighted_indices_after_global_rig_insertion():
    objects = _objects()
    result = build_connected_group_document(
        objects,
        ConnectedGroupSettings(100, 100, anchor_component_id="first"),
        profile=TwoAxisScaleRigProfile(),
    )

    local = _mesh_attachment(objects[1].document, "Second_Segment_0")
    composed = _mesh_attachment(result.document, "Second_Segment_0")
    local_vertices = decode_weighted_vertices(
        local.vertices,
        expected_vertex_count=len(local.uvs) // 2,
    )
    composed_vertices = decode_weighted_vertices(
        composed.vertices,
        expected_vertex_count=len(composed.uvs) // 2,
    )
    bone_map = result.composition.bone_map_for("second")

    assert tuple(
        tuple(influence.bone_index for influence in vertex.influences)
        for vertex in composed_vertices
    ) == tuple(
        tuple(
            bone_map.global_index_for(influence.bone_index)
            for influence in vertex.influences
        )
        for vertex in local_vertices
    )
    assert tuple(
        tuple((influence.x, influence.y, influence.weight) for influence in vertex.influences)
        for vertex in composed_vertices
    ) == tuple(
        tuple((influence.x, influence.y, influence.weight) for influence in vertex.influences)
        for vertex in local_vertices
    )


def test_production_adapter_routes_connected_two_axis_to_profile_aware_builder():
    source = COMPOSITION.read_text(encoding="utf-8")

    assert "CONNECTED mode does not yet support TWO_AXIS_ROTATION_SCALE" not in source
    assert "build_connected_group_document(" in source
    assert "profile=prepared[0].rig.profile" in source


def test_public_settings_document_connected_two_axis_support():
    source = SETTINGS.read_text(encoding="utf-8")

    assert "TWO_AXIS_ROTATION_SCALE" in source
    assert "Connected composition remains blocked" not in source
    assert "five-phase connected" in source
