import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.animation_model_contract as animation_contract
import Blender_to_Spine2D_Mesh_Exporter.domain.spine.model as spine_model
import Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_attachment_contract as attachment_contract
import Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_slot_contract as slot_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    is_finite_number,
    require_name,
)


def test_consumers_hold_exact_shared_helper_objects():
    assert spine_model._require_name is require_name
    assert animation_contract._require_name is require_name
    assert slot_contract._require_name is require_name
    assert attachment_contract._require_name is require_name
    assert spine_model._is_finite_number is is_finite_number
    assert animation_contract._is_finite_number is is_finite_number


def test_model_name_diagnostics_are_preserved():
    with pytest.raises(TypeError, match="name must be str"):
        spine_model.Bone(1)

    with pytest.raises(ValueError, match="name cannot be empty"):
        spine_model.Bone("  ")


def test_setup_slot_name_diagnostics_are_preserved():
    with pytest.raises(TypeError, match=r"slot_names\[0\] must be str"):
        slot_contract.SetupSlotIndex((1,))

    with pytest.raises(ValueError, match=r"slot_names\[0\] cannot be empty"):
        slot_contract.SetupSlotIndex((" ",))


def test_setup_attachment_name_diagnostics_are_preserved():
    with pytest.raises(
        TypeError,
        match=r"skin_attachments\[0\] slot name must be str",
    ):
        attachment_contract.SetupAttachmentNameIndex(({1: {}},))

    with pytest.raises(
        ValueError,
        match=r"skin_attachments\[0\]\['slot'\] attachment name cannot be empty",
    ):
        attachment_contract.SetupAttachmentNameIndex(
            ({"slot": {" ": {}}},)
        )


def test_model_and_animation_share_bool_exclusion_and_finiteness():
    with pytest.raises(TypeError, match="width must be a finite number or None"):
        spine_model.MeshAttachment(
            "mesh",
            uvs=(),
            triangles=(),
            vertices=(),
            hull=0,
            width=True,
        )

    with pytest.raises(ValueError, match="document.events.step.float must be finite"):
        animation_contract._validate_event_definitions(
            {"step": {"float": float("inf")}},
            path="document.events",
        )
