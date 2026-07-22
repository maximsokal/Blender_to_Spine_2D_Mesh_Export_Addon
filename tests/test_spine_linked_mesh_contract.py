from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import (
    AttachmentReference,
    LinkedMeshResolver,
    validate_setup_linked_meshes,
)


RAW_MESH = {
    "type": "mesh",
    "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "triangles": [0, 1, 2],
    "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "hull": 3,
}


def typed_mesh(name="parent"):
    return MeshAttachment(
        name=name,
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
    )


def linked(parent, *, skin=None, attachment_type="linkedmesh", **extras):
    value = {
        "type": attachment_type,
        "parent": parent,
        **extras,
    }
    if skin is not None:
        value["skin"] = skin
    return value


def build_document(skins, *, slots=None, setup_attachment="child"):
    if slots is None:
        slots = (Slot("slot", "root", attachment=setup_attachment),)
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=slots,
        skins=skins,
        animations={"animation": {}},
    )


def test_same_skin_linked_mesh_resolves_to_raw_mesh_and_is_preserved():
    child = linked("parent", futureField=True)
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "parent": deepcopy(RAW_MESH),
                    "child": child,
                }
            },
        ),
    )
    document = build_document(skins)
    source = deepcopy(child)

    serialized = SpineSerializer().to_dict(document)

    assert serialized["skins"][0]["attachments"]["slot"]["child"] == source


def test_omitted_and_empty_parent_skin_use_default_skin():
    for skin_value in (None, ""):
        child = linked("parent")
        if skin_value == "":
            child["skin"] = ""
        skins = (
            Skin(
                "default",
                {"slot": {"parent": deepcopy(RAW_MESH), "child": child}},
            ),
        )

        validate_setup_linked_meshes(skins)


def test_explicit_parent_skin_and_cross_skin_chain_are_supported():
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "base": typed_mesh("base"),
                    "child": linked("middle", skin="alternate"),
                }
            },
        ),
        Skin(
            "alternate",
            {
                "slot": {
                    "middle": linked("base", skin="default"),
                }
            },
        ),
    )

    resolver = LinkedMeshResolver(skins)
    result = resolver.resolve(
        AttachmentReference("default", "slot", "child")
    )

    assert result.source == AttachmentReference("default", "slot", "child")
    assert result.terminal == AttachmentReference("default", "slot", "base")
    assert isinstance(result.terminal_attachment, MeshAttachment)
    assert result.terminal_path.endswith(".attachments.slot.base")
    assert len(resolver.validate_all()) == 2


def test_parent_lookup_is_locked_to_the_same_slot():
    skins = (
        Skin(
            "default",
            {
                "slot": {"child": linked("parent")},
                "other": {"parent": deepcopy(RAW_MESH)},
            },
        ),
    )
    slots = (
        Slot("slot", "root", attachment="child"),
        Slot("other", "root"),
    )

    with pytest.raises(ValueError, match="undefined attachment 'parent'"):
        SpineSerializer().to_dict(build_document(skins, slots=slots))


@pytest.mark.parametrize(
    "child, expected",
    (
        ({"type": "linkedmesh"}, "parent is required"),
        (linked(None), "parent is required"),
        (linked(True), "parent must be str"),
        (linked(""), "parent cannot be empty"),
        (linked("parent", skin=True), "skin must be str"),
        (linked("parent", skin="   "), "skin cannot be empty"),
    ),
)
def test_linked_mesh_parent_metadata_is_strict(child, expected):
    skins = (Skin("default", {"slot": {"child": child}}),)

    with pytest.raises((TypeError, ValueError), match=expected):
        validate_setup_linked_meshes(skins)


def test_missing_default_parent_skin_is_rejected():
    skins = (
        Skin("alternate", {"slot": {"child": linked("parent")}}),
    )

    with pytest.raises(ValueError, match="undefined skin 'default'"):
        validate_setup_linked_meshes(skins)


def test_missing_explicit_parent_skin_is_rejected():
    skins = (
        Skin(
            "default",
            {"slot": {"child": linked("parent", skin="missing")}},
        ),
    )

    with pytest.raises(ValueError, match="undefined skin 'missing'"):
        validate_setup_linked_meshes(skins)


def test_parent_skin_without_source_slot_attachments_is_rejected():
    skins = (
        Skin(
            "default",
            {"slot": {"child": linked("parent", skin="alternate")}},
        ),
        Skin("alternate", {"other": {"parent": deepcopy(RAW_MESH)}}),
    )

    with pytest.raises(ValueError, match="slot 'slot' without attachments"):
        validate_setup_linked_meshes(skins)


def test_missing_parent_attachment_is_rejected():
    skins = (
        Skin("default", {"slot": {"child": linked("missing")}}),
    )

    with pytest.raises(ValueError, match="undefined attachment 'missing'"):
        validate_setup_linked_meshes(skins)


@pytest.mark.parametrize(
    "parent_type",
    ("region", "point", "boundingbox", "path", "clipping"),
)
def test_parent_must_be_mesh_compatible(parent_type):
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "parent": {"type": parent_type},
                    "child": linked("parent"),
                }
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match=rf"unsupported attachment type '{parent_type}'",
    ):
        validate_setup_linked_meshes(skins)


def test_self_cycle_is_rejected_with_rendered_chain():
    skins = (
        Skin("default", {"slot": {"child": linked("child")}}),
    )

    with pytest.raises(
        ValueError,
        match=r"default/slot/child -> default/slot/child",
    ):
        validate_setup_linked_meshes(skins)


def test_multi_skin_parent_cycle_is_rejected():
    skins = (
        Skin(
            "default",
            {"slot": {"first": linked("second", skin="alternate")}},
        ),
        Skin(
            "alternate",
            {"slot": {"second": linked("first", skin="default")}},
        ),
    )

    with pytest.raises(ValueError, match="linked mesh parent cycle"):
        validate_setup_linked_meshes(skins)


def test_duplicate_skin_name_is_fail_closed_in_direct_resolver():
    skins = (
        Skin("default", {"slot": {"parent": deepcopy(RAW_MESH)}}),
        Skin("default", {"slot": {"child": linked("parent")}}),
    )

    with pytest.raises(ValueError, match="duplicated skin 'default'"):
        validate_setup_linked_meshes(skins)


def test_legacy_mesh_with_parent_spelling_is_resolved_by_shared_resolver():
    child = {
        **deepcopy(RAW_MESH),
        "parent": "parent",
    }
    skins = (
        Skin(
            "default",
            {"slot": {"parent": deepcopy(RAW_MESH), "child": child}},
        ),
    )

    result = LinkedMeshResolver(skins).resolve(
        AttachmentReference("default", "slot", "child")
    )

    assert result.terminal == AttachmentReference("default", "slot", "parent")


def test_serializer_revalidates_mutated_linked_parent():
    child = linked("parent")
    skins = (
        Skin(
            "default",
            {"slot": {"parent": deepcopy(RAW_MESH), "child": child}},
        ),
    )
    document = build_document(skins)
    child["parent"] = "missing"

    with pytest.raises(ValueError, match="undefined attachment 'missing'"):
        SpineSerializer().to_dict(document)


def test_documents_without_linked_meshes_are_untouched():
    region = {
        "type": "region",
        "path": "image",
        "width": 64,
        "height": 64,
        "futureField": {"enabled": True},
    }
    skins = (Skin("default", {"slot": {"child": region}}),)
    document = build_document(skins)
    source = deepcopy(region)

    serialized = SpineSerializer().to_dict(document)

    assert serialized["skins"][0]["attachments"]["slot"]["child"] == source
