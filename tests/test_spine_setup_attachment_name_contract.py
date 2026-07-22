from dataclasses import FrozenInstanceError

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_attachment_contract import (
    SetupAttachmentNameIndex,
)


def test_index_retains_exact_tuple_and_unions_names_across_skins():
    skin_attachments = (
        {"slot": {"A": object()}, "other": {"X": object()}},
        {"slot": {"B": object(), "A": object()}},
    )

    index = SetupAttachmentNameIndex(skin_attachments)

    assert index.skin_attachments is skin_attachments
    assert index.names_for_slot("slot") == frozenset({"A", "B"})
    assert index.names_for_slot("other") == frozenset({"X"})


def test_index_is_a_snapshot_of_source_names():
    source = {"slot": {"A": object()}}
    index = SetupAttachmentNameIndex((source,))

    source["slot"]["B"] = object()

    assert index.names_for_slot("slot") == frozenset({"A"})


def test_attachment_names_remain_isolated_by_slot():
    index = SetupAttachmentNameIndex(
        ({"slot": {"A": object()}, "other": {"B": object()}},)
    )

    index.require("slot", "A", path="timeline.name")
    with pytest.raises(
        ValueError,
        match="undefined attachment 'B' for slot 'slot'",
    ):
        index.require("slot", "B", path="timeline.name")


def test_missing_slot_has_an_empty_name_set():
    index = SetupAttachmentNameIndex(({},))

    assert index.names_for_slot("missing") == frozenset()


def test_lookup_snapshot_is_deeply_read_only():
    index = SetupAttachmentNameIndex(({"slot": {"A": object()}},))

    with pytest.raises(TypeError):
        index._names_by_slot["slot"] = frozenset({"B"})
    assert isinstance(index._names_by_slot["slot"], frozenset)


def test_frozen_index_rejects_field_replacement():
    index = SetupAttachmentNameIndex(({},))

    with pytest.raises(FrozenInstanceError):
        index.skin_attachments = ()


@pytest.mark.parametrize("value", (None, [], {}, "skins", 1, True))
def test_skin_attachment_collection_must_be_tuple(value):
    with pytest.raises(TypeError, match="skin_attachments must be tuple"):
        SetupAttachmentNameIndex(value)


def test_each_skin_attachment_root_must_be_mapping():
    with pytest.raises(
        TypeError,
        match=r"skin_attachments\[0\] must be a mapping",
    ):
        SetupAttachmentNameIndex((None,))


def test_each_slot_attachment_group_must_be_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        SetupAttachmentNameIndex(({"slot": None},))


@pytest.mark.parametrize("slot_name", (None, 1, True, (), "   "))
def test_slot_names_are_strict(slot_name):
    expected = TypeError if slot_name != "   " else ValueError
    with pytest.raises(expected):
        SetupAttachmentNameIndex(({slot_name: {}},))


@pytest.mark.parametrize("attachment_name", (None, 1, True, (), "   "))
def test_attachment_names_are_strict(attachment_name):
    expected = TypeError if attachment_name != "   " else ValueError
    with pytest.raises(expected):
        SetupAttachmentNameIndex(({"slot": {attachment_name: object()}},))


def test_require_preserves_caller_path_and_name_validation():
    index = SetupAttachmentNameIndex(({"slot": {"A": object()}},))

    with pytest.raises(ValueError) as error:
        index.require("slot", "missing", path="document.animations.idle.name")
    assert str(error.value).startswith("document.animations.idle.name ")

    with pytest.raises(TypeError, match="document.animations.idle.name must be str"):
        index.require("slot", 1, path="document.animations.idle.name")
