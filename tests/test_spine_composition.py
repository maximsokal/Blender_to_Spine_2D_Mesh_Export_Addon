from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    ConstraintOrderPolicy,
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    MeshAttachment,
    Skin,
    SpineCompositionError,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineSerializer,
    SpineValidator,
    build_legacy_mesh_document,
    build_legacy_rig,
    compose_spine_documents,
    decode_weighted_vertices,
)


def build_document(prefix, *, spine_version="4.2.43", event_value=None):
    rig = build_legacy_rig(
        LegacyRigBuildRequest(
            prefix=prefix,
            texture_width=100,
            texture_height=100,
            z_groups=(LegacyZGroup(0.0, height_real_pixels=0.0),),
        )
    )
    slot_name = f"{prefix}_Segment_0"
    request = LegacyMeshAttachmentRequest(
        slot_name=slot_name,
        attachment_name=slot_name,
        vertex_prefix=slot_name,
        image_path=f"images/{prefix}_Baked",
        width=100,
        height=100,
        vertices=(
            LegacyAttachmentVertex(0, (0.0, 0.0), (-50.0, 50.0), 1),
            LegacyAttachmentVertex(1, (1.0, 0.0), (50.0, 50.0), 1),
            LegacyAttachmentVertex(2, (0.0, 1.0), (-50.0, -50.0), 1),
        ),
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
    )
    built = build_legacy_mesh_document(
        rig,
        (request,),
        skeleton_metadata={
            "spine": spine_version,
            "hash": f"{prefix}-hash",
            "width": 100,
            "height": 100,
            "images": "",
            "audio": "./audio",
        },
    )
    events = {} if event_value is None else {"event": {"int": event_value}}
    document = replace(
        built.document,
        animations={
            "animation": {
                "bones": {
                    rig.info.main_bone_name: {
                        "rotate": [{"value": 5.0}],
                    }
                }
            }
        },
        events=events,
    )
    SpineValidator().validate_or_raise(document)
    return document


def mesh_attachment(document, slot_name):
    for skin in document.skins:
        slot_attachments = skin.attachments.get(slot_name)
        if slot_attachments is None:
            continue
        attachment = slot_attachments.get(slot_name)
        assert isinstance(attachment, MeshAttachment)
        return attachment
    raise AssertionError(f"Attachment {slot_name!r} not found")


def influence_indices(attachment):
    decoded = decode_weighted_vertices(
        attachment.vertices,
        expected_vertex_count=len(attachment.uvs) // 2,
    )
    return tuple(
        tuple(influence.bone_index for influence in vertex.influences)
        for vertex in decoded
    )


def components():
    first = build_document("First")
    second = build_document("Second")
    return (
        SpineDocumentComponent("first", first, animation_namespace="object_1"),
        SpineDocumentComponent("second", second, animation_namespace="object_2"),
    )


def test_composition_shares_root_and_remaps_every_weighted_index():
    first_component, second_component = components()
    result = compose_spine_documents((first_component, second_component))

    assert tuple(bone.name for bone in result.document.bones).count("root") == 1
    assert len(result.document.bones) == (
        len(first_component.document.bones)
        + len(second_component.document.bones)
        - 1
    )
    first_map = result.bone_map_for("first")
    second_map = result.bone_map_for("second")
    assert first_map.global_index_for(0) == 0
    assert second_map.global_index_for(0) == 0

    local_second = mesh_attachment(
        second_component.document,
        "Second_Segment_0",
    )
    global_second = mesh_attachment(
        result.document,
        "Second_Segment_0",
    )
    local_indices = influence_indices(local_second)
    global_indices = influence_indices(global_second)
    assert global_indices == tuple(
        tuple(second_map.global_index_for(index) for index in vertex_indices)
        for vertex_indices in local_indices
    )
    assert all(
        global_index != local_index
        for local_vertex, global_vertex in zip(local_indices, global_indices)
        for local_index, global_index in zip(local_vertex, global_vertex)
    )
    assert SpineValidator().validate(result.document) == ()


def test_constraint_orders_are_rebased_without_changing_relative_order():
    result = compose_spine_documents(components())
    assignments = result.constraint_orders

    assert tuple(item.global_order for item in assignments) == tuple(
        range(len(assignments))
    )
    all_orders = tuple(
        constraint.order
        for constraint in (*result.document.ik, *result.document.transform)
    )
    assert set(all_orders) == set(range(len(assignments)))
    for component_id in ("first", "second"):
        component_assignments = tuple(
            item for item in assignments if item.component_id == component_id
        )
        assert tuple(item.original_order for item in component_assignments) == tuple(
            sorted(item.original_order for item in component_assignments)
        )


def test_animations_are_namespaced_in_component_order():
    result = compose_spine_documents(components())

    assert tuple(result.document.animations) == (
        "object_1/animation",
        "object_2/animation",
    )
    assert tuple(
        (item.component_id, item.original_name, item.global_name)
        for item in result.animation_names
    ) == (
        ("first", "animation", "object_1/animation"),
        ("second", "animation", "object_2/animation"),
    )


def test_source_documents_are_not_mutated_by_composition():
    source_components = components()
    serializer = SpineSerializer()
    before = tuple(
        serializer.to_json(component.document, indent=2)
        for component in source_components
    )

    compose_spine_documents(source_components)

    after = tuple(
        serializer.to_json(component.document, indent=2)
        for component in source_components
    )
    assert after == before


def test_preserve_constraint_policy_rejects_cross_document_order_collisions():
    with pytest.raises(SpineCompositionError, match="order .* collides"):
        compose_spine_documents(
            components(),
            SpineCompositionSettings(
                constraint_order_policy=ConstraintOrderPolicy.PRESERVE,
            ),
        )


def test_non_shared_bone_name_collision_is_rejected():
    duplicated = build_document("Same")
    with pytest.raises(SpineCompositionError, match="non-shared bone"):
        compose_spine_documents(
            (
                SpineDocumentComponent("first", duplicated),
                SpineDocumentComponent("second", duplicated),
            )
        )


def test_shared_root_must_be_exactly_identical():
    first = build_document("First")
    second = build_document("Second")
    changed_root = replace(second.bones[0], length=1.0)
    second = replace(second, bones=(changed_root,) + second.bones[1:])
    SpineValidator().validate_or_raise(second)

    with pytest.raises(SpineCompositionError, match="Shared bone 'root' differs"):
        compose_spine_documents(
            (
                SpineDocumentComponent("first", first),
                SpineDocumentComponent("second", second),
            )
        )


def test_spine_version_mismatch_is_rejected_before_merge():
    with pytest.raises(SpineCompositionError, match="Spine version"):
        compose_spine_documents(
            (
                SpineDocumentComponent("first", build_document("First")),
                SpineDocumentComponent(
                    "second",
                    build_document("Second", spine_version="3.8.99"),
                ),
            )
        )


def test_conflicting_event_definitions_are_rejected():
    with pytest.raises(SpineCompositionError, match="events entry 'event' conflicts"):
        compose_spine_documents(
            (
                SpineDocumentComponent(
                    "first",
                    build_document("First", event_value=1),
                ),
                SpineDocumentComponent(
                    "second",
                    build_document("Second", event_value=2),
                ),
            )
        )


def test_unknown_local_weighted_bone_index_never_falls_back_to_root():
    document = build_document("Broken")
    attachment = mesh_attachment(document, "Broken_Segment_0")
    malformed_vertices = list(attachment.vertices)
    malformed_vertices[1] = 999
    malformed_attachment = replace(
        attachment,
        vertices=tuple(malformed_vertices),
    )
    skin = document.skins[0]
    changed_attachments = {
        slot_name: dict(slot_attachments)
        for slot_name, slot_attachments in skin.attachments.items()
    }
    changed_attachments["Broken_Segment_0"]["Broken_Segment_0"] = (
        malformed_attachment
    )
    invalid_document = replace(
        document,
        skins=(replace(skin, attachments=changed_attachments),),
    )

    with pytest.raises(SpineCompositionError, match="not a valid Spine document"):
        compose_spine_documents(
            (SpineDocumentComponent("broken", invalid_document),)
        )
