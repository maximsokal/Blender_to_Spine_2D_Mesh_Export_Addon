import Blender_to_Spine2D_Mesh_Exporter.domain.spine.model as spine_model
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_slot_contract import (
    SetupSlotIndex,
)


def test_document_reuses_one_index_object_across_model_animation_boundaries(
    monkeypatch,
):
    original_require = SetupSlotIndex.require
    calls = []

    def recording_require(self, slot_name, *, path):
        calls.append((self, path))
        return original_require(self, slot_name, path=path)

    monkeypatch.setattr(SetupSlotIndex, "require", recording_require)

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
    assert len({id(index) for index, _ in calls}) == 1
    assert [path for _, path in calls] == [
        "document.animations.idle.drawOrder[0].offsets[0].slot",
        "document.animations.idle.drawOrder[0].offsets[1].slot",
        "document.animations.idle.slots.a",
    ]
