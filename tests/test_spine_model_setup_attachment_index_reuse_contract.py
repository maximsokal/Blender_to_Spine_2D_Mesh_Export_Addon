import Blender_to_Spine2D_Mesh_Exporter.domain.spine.model as spine_model
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_attachment_contract import (
    SetupAttachmentNameIndex,
)


def test_document_builds_one_cross_skin_attachment_index(monkeypatch):
    original_require = SetupAttachmentNameIndex.require
    calls = []

    def recording_require(self, slot_name, attachment_name, *, path):
        calls.append((self, slot_name, attachment_name, path))
        return original_require(
            self,
            slot_name,
            attachment_name,
            path=path,
        )

    monkeypatch.setattr(SetupAttachmentNameIndex, "require", recording_require)

    default_attachments = {"slot": {"A": {"type": "point"}}}
    alternate_attachments = {"slot": {"B": {"type": "point"}}}
    document = spine_model.SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(spine_model.Bone("root"),),
        slots=(spine_model.Slot("slot", "root"),),
        skins=(
            spine_model.Skin("default", default_attachments),
            spine_model.Skin("alternate", alternate_attachments),
        ),
        animations={
            "idle": {
                "slots": {
                    "slot": {
                        "attachment": [
                            {"name": "A"},
                            {"time": 1, "name": "B"},
                        ]
                    }
                }
            }
        },
    )

    assert document.animations["idle"]["slots"]["slot"]["attachment"][1] == {
        "time": 1,
        "name": "B",
    }
    assert len({id(index) for index, *_ in calls}) == 1
    index = calls[0][0]
    assert index.skin_attachments[0] is default_attachments
    assert index.skin_attachments[1] is alternate_attachments
    assert [call[1:] for call in calls] == [
        (
            "slot",
            "A",
            "document.animations.idle.slots.slot.attachment[0].name",
        ),
        (
            "slot",
            "B",
            "document.animations.idle.slots.slot.attachment[1].name",
        ),
    ]
