import Blender_to_Spine2D_Mesh_Exporter.domain.spine.model as spine_model
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_attachment_contract import (
    SetupAttachmentNameIndex,
)


def test_document_builds_one_cross_skin_attachment_index(monkeypatch):
    real_index = SetupAttachmentNameIndex
    instances = []

    class RecordingSetupAttachmentNameIndex:
        def __init__(self, skin_attachments):
            self.skin_attachments = skin_attachments
            self._delegate = real_index(skin_attachments)
            self.lookups = []
            instances.append(self)

        def require(self, slot_name, attachment_name, *, path):
            self.lookups.append((slot_name, attachment_name, path))
            self._delegate.require(slot_name, attachment_name, path=path)

    monkeypatch.setattr(
        spine_model,
        "SetupAttachmentNameIndex",
        RecordingSetupAttachmentNameIndex,
    )

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
    assert len(instances) == 1
    assert instances[0].skin_attachments[0] is default_attachments
    assert instances[0].skin_attachments[1] is alternate_attachments
    assert instances[0].lookups == [
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
