import Blender_to_Spine2D_Mesh_Exporter.domain.spine.model as spine_model
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_slot_contract import (
    SetupSlotIndex,
)


def test_document_reuses_one_index_object_across_model_animation_boundaries(
    monkeypatch,
):
    real_index = SetupSlotIndex
    instances = []

    class RecordingSetupSlotIndex:
        def __init__(self, slot_names):
            self.slot_names = slot_names
            self._delegate = real_index(slot_names)
            self.paths = []
            instances.append(self)

        def require(self, slot_name, *, path):
            self.paths.append(path)
            return self._delegate.require(slot_name, path=path)

    monkeypatch.setattr(
        spine_model,
        "SetupSlotIndex",
        RecordingSetupSlotIndex,
    )

    document = spine_model.SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(spine_model.Bone("root"),),
        slots=(
            spine_model.Slot("a", "root"),
            spine_model.Slot("b", "root"),
        ),
        skins=(
            spine_model.Skin(
                "default",
                {"a": {"A": {"type": "point"}}},
            ),
        ),
        animations={
            "idle": {
                "drawOrder": [
                    {
                        "offsets": [
                            {"slot": "a", "offset": 1},
                            {"slot": "b", "offset": -1},
                        ]
                    }
                ],
                "slots": {
                    "a": {"attachment": [{"name": "A"}]},
                },
            }
        },
    )

    assert document.animations["idle"]["slots"]["a"]["attachment"] == [
        {"name": "A"}
    ]
    assert len(instances) == 1
    assert instances[0].paths == [
        "document.animations.idle.drawOrder[0].offsets[0].slot",
        "document.animations.idle.drawOrder[0].offsets[1].slot",
        "document.animations.idle.slots.a",
    ]
