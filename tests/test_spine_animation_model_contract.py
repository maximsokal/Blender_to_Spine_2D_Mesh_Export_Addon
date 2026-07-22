import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.animation_model_contract import (
    validate_animation_model_contracts,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_attachment_contract import (
    SetupAttachmentNameIndex,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_slot_contract import (
    SetupSlotIndex,
)


def snapshots():
    slot_names = ("slot",)
    skin_attachments = ({"slot": {"A": {"type": "point"}}},)
    return (
        slot_names,
        skin_attachments,
        SetupSlotIndex(slot_names),
        SetupAttachmentNameIndex(skin_attachments),
    )


def call_contract(animations, events):
    slot_names, skin_attachments, slot_index, attachment_index = snapshots()
    validate_animation_model_contracts(
        animations,
        events=events,
        slot_names=slot_names,
        skin_attachments=skin_attachments,
        setup_slot_index=slot_index,
        setup_attachment_index=attachment_index,
    )


def test_valid_model_animation_payload_is_accepted():
    call_contract(
        {
            "idle": {
                "events": [{"name": "step"}],
                "drawOrder": [{"offsets": [{"slot": "slot", "offset": 0}]}],
                "slots": {"slot": {"attachment": [{"name": "A"}]}},
            }
        },
        {"step": {}},
    )


def test_mutated_event_reference_is_rejected():
    with pytest.raises(ValueError, match="undefined event 'missing'"):
        call_contract({"idle": {"events": [{"name": "missing"}]}}, {"step": {}})


def test_mutated_draw_order_reference_is_rejected():
    with pytest.raises(ValueError, match="undefined slot 'missing'"):
        call_contract(
            {
                "idle": {
                    "drawOrder": [
                        {"offsets": [{"slot": "missing", "offset": 0}]}
                    ]
                }
            },
            {},
        )


def test_mutated_attachment_reference_is_rejected():
    with pytest.raises(
        ValueError,
        match="undefined attachment 'missing' for slot 'slot'",
    ):
        call_contract(
            {
                "idle": {
                    "slots": {
                        "slot": {"attachment": [{"name": "missing"}]}
                    }
                }
            },
            {},
        )


def test_stale_slot_index_is_rejected():
    slot_names = ("slot",)
    equivalent = tuple(["slot"])
    skin_attachments = ({"slot": {"A": {}}},)

    with pytest.raises(ValueError, match="exact slot_names tuple"):
        validate_animation_model_contracts(
            {},
            events={},
            slot_names=equivalent,
            skin_attachments=skin_attachments,
            setup_slot_index=SetupSlotIndex(slot_names),
        )


def test_stale_attachment_index_is_rejected():
    slot_names = ("slot",)
    first = ({"slot": {"A": {}}},)
    second = ({"slot": {"A": {}}},)

    with pytest.raises(ValueError, match="exact skin_attachments tuple"):
        validate_animation_model_contracts(
            {},
            events={},
            slot_names=slot_names,
            skin_attachments=second,
            setup_attachment_index=SetupAttachmentNameIndex(first),
        )
