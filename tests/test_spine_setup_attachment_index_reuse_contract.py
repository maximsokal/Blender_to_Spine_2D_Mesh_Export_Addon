import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_attachment_contract import (
    SetupAttachmentNameIndex,
    resolve_setup_attachment_name_index,
)


def test_direct_call_builds_index_for_exact_snapshot():
    skin_attachments = ({"slot": {"A": {}}},)

    resolved = resolve_setup_attachment_name_index(skin_attachments, None)

    assert isinstance(resolved, SetupAttachmentNameIndex)
    assert resolved.skin_attachments is skin_attachments
    assert resolved.names_for_slot("slot") == frozenset({"A"})


def test_exact_index_is_reused_without_rebuilding():
    skin_attachments = ({"slot": {"A": {}}},)
    index = SetupAttachmentNameIndex(skin_attachments)

    assert resolve_setup_attachment_name_index(skin_attachments, index) is index


def test_non_index_is_rejected():
    with pytest.raises(
        TypeError,
        match="setup_attachment_index must be SetupAttachmentNameIndex or None",
    ):
        resolve_setup_attachment_name_index((), object())


def test_equivalent_but_distinct_snapshot_is_rejected():
    first = ({"slot": {"A": {}}},)
    second = ({"slot": {"A": {}}},)
    index = SetupAttachmentNameIndex(first)

    with pytest.raises(ValueError, match="exact skin_attachments tuple"):
        resolve_setup_attachment_name_index(second, index)
