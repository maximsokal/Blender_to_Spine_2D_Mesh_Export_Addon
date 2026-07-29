from dataclasses import replace

import pytest
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    ConnectedGroupBuildError,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    LegacyRigProfile,
    SpineSerializer,
    SpineValidator,
    build_connected_group_document,
    decode_weighted_vertices,
)

from test_spine_composition import build_document, mesh_attachment


def connected_objects():
    return (
        ConnectedObjectDocument(
            component_id="first",
            prefix="First",
            document=build_document("First"),
            world_position=(10.0, 20.0, 5.0),
            animation_namespace="object_1",
        ),
        ConnectedObjectDocument(
            component_id="second",
            prefix="Second",
            document=build_document("Second"),
            world_position=(12.0, 23.0, 7.0),
            animation_namespace="object_2",
        ),
        ConnectedObjectDocument(
            component_id="third",
            prefix="Third",
            document=build_document("Third"),
            world_position=(9.0, 18.0, 5.00005),
            animation_namespace="object_3",
        ),
    )


def settings(**overrides):
    values = {
        "texture_width": 100,
        "texture_height": 100,
        "group_prefix": "all_objects",
        "z_tolerance": 1e-4,
    }
    values.update(overrides)
    return ConnectedGroupSettings(**values)


def bone_by_name(document, name):
    return next(bone for bone in document.bones if bone.name == name)


def constraint_by_name(document, name):
    return next(
        constraint
        for constraint in (*document.ik, *document.transform)
        if constraint.name == name
    )


def test_layers_are_clustered_top_down_and_offsets_use_first_anchor():
    result = build_connected_group_document(connected_objects(), settings())

    assert tuple(
        (
            layer.layer_index,
            layer.component_ids,
            layer.scale_bone_name,
            layer.layer_bone_name,
        )
        for layer in result.layers
    ) == (
        (0, ("second",), "all_objects_0_scale", "all_objects_layer_0"),
        (
            1,
            ("first", "third"),
            "all_objects_1_scale",
            "all_objects_layer_1",
        ),
    )
    placements = {item.component_id: item for item in result.placements}
    assert (
        placements["first"].relative_x,
        placements["first"].relative_y,
        placements["first"].relative_z,
        placements["first"].layer_index,
    ) == (0.0, 0.0, 0.0, 1)
    assert (
        placements["second"].relative_x,
        placements["second"].relative_y,
        placements["second"].relative_z,
        placements["second"].layer_index,
    ) == (2.0, 3.0, 2.0, 0)
    assert placements["third"].layer_index == 1


def test_main_bones_are_reparented_and_keep_full_legacy_xy_offsets():
    result = build_connected_group_document(connected_objects(), settings())

    first = bone_by_name(result.document, "First_main")
    second = bone_by_name(result.document, "Second_main")
    third = bone_by_name(result.document, "Third_main")
    assert (first.parent, first.x, first.y) == ("all_objects_layer_1", 0.0, 0.0)
    assert (second.parent, second.x, second.y) == (
        "all_objects_layer_0",
        200.0,
        300.0,
    )
    assert (third.parent, third.x, third.y) == (
        "all_objects_layer_1",
        -100.0,
        -200.0,
    )
    assert tuple(bone.name for bone in result.document.bones).count("root") == 1
    non_legacy_order_issues = tuple(
        issue
        for issue in SpineValidator().validate(result.document)
        if issue.code != "DUPLICATE_CONSTRAINT_ORDER"
    )
    assert non_legacy_order_issues == ()


def test_global_rig_contains_required_legacy_names_and_targets():
    result = build_connected_group_document(connected_objects(), settings())
    profile = LegacyRigProfile()
    bone_names = {bone.name for bone in result.document.bones}

    assert {
        profile.main_bone("all_objects"),
        profile.base_bone("all_objects"),
        profile.scale_rotate_x_bone("all_objects"),
        profile.rotate_x_bone("all_objects"),
        *profile.control_bones("all_objects"),
        *profile.ik_chain_bones("all_objects"),
        "all_objects_0_scale",
        "all_objects_layer_0",
        "all_objects_1_scale",
        "all_objects_layer_1",
    } <= bone_names

    rotation_x = constraint_by_name(
        result.document,
        profile.rotation_x_constraint("all_objects"),
    )
    assert rotation_x.bones == (
        "all_objects_0_scale",
        "all_objects_1_scale",
        "all_objects",
    )
    assert rotation_x.extras["rotation"] == 90
    assert rotation_x.extras["x"] == -200.0
    assert rotation_x.extras["y"] == -50.0
    assert rotation_x.extras["scaleX"] == -1
    assert rotation_x.extras["scaleY"] == -1

    rotation_z = constraint_by_name(
        result.document,
        profile.rotation_z_constraint("all_objects"),
    )
    assert rotation_z.bones == ("First", "Second", "Third")
    assert rotation_z.target == profile.control_z_bone("all_objects")
    assert rotation_z.extras == {
        "local": True,
        "mixX": 0,
        "mixScaleX": 0,
        "mixShearY": 0,
    }


def test_constraint_schedule_uses_shared_orders_for_same_layer_objects():
    objects = connected_objects()
    result = build_connected_group_document(objects, settings())
    schedule = result.constraint_schedule

    assert schedule.unique_orders == tuple(range(15))
    assert schedule.object_rotation_x == (
        ("first", 4),
        ("second", 3),
        ("third", 4),
    )
    all_constraints = (*result.document.ik, *result.document.transform)
    all_orders = tuple(constraint.order for constraint in all_constraints)
    assert set(all_orders) == set(schedule.unique_orders)
    assert len(all_orders) > len(set(all_orders))
    assert constraint_by_name(result.document, "First_rotation_X").order == (
        constraint_by_name(result.document, "Third_rotation_X").order
    )
    for item in objects:
        assert constraint_by_name(
            result.document,
            f"{item.prefix}_scale_compensator",
        ).order == 6


def test_weighted_attachments_remain_valid_after_global_bones_are_inserted():
    objects = connected_objects()
    result = build_connected_group_document(objects, settings())

    for item in objects:
        local = mesh_attachment(
            item.document,
            f"{item.prefix}_Segment_0",
        )
        global_attachment = mesh_attachment(
            result.document,
            f"{item.prefix}_Segment_0",
        )
        local_vertices = decode_weighted_vertices(
            local.vertices,
            expected_vertex_count=len(local.uvs) // 2,
        )
        global_vertices = decode_weighted_vertices(
            global_attachment.vertices,
            expected_vertex_count=len(global_attachment.uvs) // 2,
        )
        bone_map = result.composition.bone_map_for(item.component_id)
        assert tuple(
            tuple(influence.bone_index for influence in vertex.influences)
            for vertex in global_vertices
        ) == tuple(
            tuple(
                bone_map.global_index_for(influence.bone_index)
                for influence in vertex.influences
            )
            for vertex in local_vertices
        )


def test_animations_keep_explicit_legacy_style_namespaces():
    result = build_connected_group_document(connected_objects(), settings())

    assert tuple(result.document.animations) == (
        "object_1/animation",
        "object_2/animation",
        "object_3/animation",
    )


def test_explicit_anchor_changes_offsets_without_reordering_input_documents():
    result = build_connected_group_document(
        connected_objects(),
        settings(anchor_component_id="second"),
    )
    placements = {item.component_id: item for item in result.placements}

    assert (
        placements["second"].relative_x,
        placements["second"].relative_y,
        placements["second"].relative_z,
    ) == (0.0, 0.0, 0.0)
    assert (
        placements["first"].relative_x,
        placements["first"].relative_y,
        placements["first"].relative_z,
    ) == (-2.0, -3.0, -2.0)
    assert tuple(result.document.animations) == (
        "object_1/animation",
        "object_2/animation",
        "object_3/animation",
    )


def test_source_documents_are_not_mutated():
    objects = connected_objects()
    serializer = SpineSerializer()
    before = tuple(serializer.to_json(item.document, indent=2) for item in objects)

    build_connected_group_document(objects, settings())

    after = tuple(serializer.to_json(item.document, indent=2) for item in objects)
    assert after == before


def test_duplicate_prefix_and_group_prefix_collision_are_rejected():
    objects = connected_objects()
    duplicate = replace(objects[1], prefix="First")
    with pytest.raises(ValueError, match="prefixes must be unique"):
        build_connected_group_document(
            (objects[0], duplicate),
            settings(),
        )

    with pytest.raises(ValueError, match="group_prefix cannot equal"):
        build_connected_group_document(
            objects[:2],
            settings(group_prefix="First"),
        )


def test_missing_or_extra_a1_constraints_are_rejected_before_composition():
    objects = connected_objects()
    broken_document = replace(
        objects[0].document,
        transform=objects[0].document.transform[:-1],
    )
    SpineValidator().validate_or_raise(broken_document)
    broken = replace(objects[0], document=broken_document)

    with pytest.raises(ConnectedGroupBuildError, match="exactly the six A1 constraints"):
        build_connected_group_document(
            (broken, objects[1]),
            settings(),
        )


def test_at_least_two_objects_and_known_anchor_are_required():
    objects = connected_objects()
    with pytest.raises(ValueError, match="at least two"):
        build_connected_group_document((objects[0],), settings())
    with pytest.raises(ValueError, match="anchor_component_id"):
        build_connected_group_document(
            objects[:2],
            settings(anchor_component_id="missing"),
        )
