from copy import deepcopy

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract as linked_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import Skin
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    require_name,
)


def record(*, parent="parent", include_skin=False, skin=None):
    attachment = {
        "type": "linkedmesh",
        "parent": parent,
    }
    if include_skin:
        attachment["skin"] = skin
    return linked_contract.SetupAttachment(
        reference=linked_contract.AttachmentReference(
            skin_name="source",
            slot_name="slot",
            attachment_name="child",
        ),
        attachment=attachment,
        path="document.skins[0].attachments.slot.child",
    )


def test_linked_mesh_contract_holds_exact_shared_name_function():
    assert linked_contract._require_name is require_name


@pytest.mark.parametrize(
    "include_skin, skin_value",
    (
        (False, None),
        (True, None),
        (True, ""),
    ),
)
def test_default_skin_fallback_remains_limited_to_none_or_empty_string(
    include_skin,
    skin_value,
):
    reference = linked_contract.LinkedMeshResolver._parent_reference(
        record(include_skin=include_skin, skin=skin_value)
    )

    assert reference is not None
    assert reference.skin_name == "default"


def test_whitespace_parent_skin_is_not_treated_as_default():
    with pytest.raises(ValueError, match=r"\.skin cannot be empty"):
        linked_contract.LinkedMeshResolver._parent_reference(
            record(include_skin=True, skin="   ")
        )


def test_parent_name_preserves_exact_spelling_without_normalization():
    source = record(parent=" parent ")
    attachment_before = deepcopy(source.attachment)

    reference = linked_contract.LinkedMeshResolver._parent_reference(source)

    assert reference is not None
    assert reference.attachment_name == " parent "
    assert source.attachment == attachment_before


@pytest.mark.parametrize(
    "parent, expected",
    (
        (None, "parent is required"),
        (True, "parent must be str"),
        ("", "parent cannot be empty"),
        ("   ", "parent cannot be empty"),
    ),
)
def test_parent_name_preserves_exact_shared_diagnostics(parent, expected):
    with pytest.raises((TypeError, ValueError), match=expected):
        linked_contract.LinkedMeshResolver._parent_reference(record(parent=parent))


def test_require_skin_preserves_shared_name_diagnostic_before_lookup():
    skins = (Skin("default", {}),)
    resolver = linked_contract.LinkedMeshResolver(skins)

    with pytest.raises(ValueError, match="skin name cannot be empty"):
        resolver.require_skin("   ", path="animation.attachments")
