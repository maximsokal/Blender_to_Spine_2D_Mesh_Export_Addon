from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import Skin
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.deform_timeline_contract import validate_animation_deform_timelines
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import validate_setup_linked_meshes
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.sequence_timeline_contract import validate_animation_sequence_timelines
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_slot_contract import SetupSlotIndex
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.slot_color_timeline_contract import validate_animation_slot_color_timelines


RAW_MESH = {
    "type": "mesh",
    "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "triangles": [0, 1, 2],
    "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "hull": 3,
}


def setup_skins():
    return (
        Skin(
            "default",
            {
                "slot": {
                    "mesh": deepcopy(RAW_MESH),
                    "sequence": {"type": "region", "sequence": {"count": 2}},
                }
            },
        ),
    )


def animation(kind, slot_name="slot"):
    if kind == "color":
        return {"animation": {"slots": {slot_name: {"rgba": [{"color": "FFFFFFFF"}]}}}}
    timeline_name = "mesh" if kind == "deform" else "sequence"
    timeline = {"deform": [{"vertices": [0.1, -0.1]}]} if kind == "deform" else {"sequence": [{}]}
    return {
        "animation": {
            "attachments": {
                "default": {slot_name: {timeline_name: timeline}}
            }
        }
    }


def validate(kind, animations, slot_names, index=None):
    kwargs = {
        "slot_names": slot_names,
        "path": "document.animations",
        "setup_slot_index": index,
    }
    if kind == "color":
        validate_animation_slot_color_timelines(animations, **kwargs)
        return
    skins = setup_skins()
    kwargs["skins"] = skins
    if index is not None:
        kwargs["linked_mesh_resolver"] = validate_setup_linked_meshes(skins)
    function = validate_animation_deform_timelines if kind == "deform" else validate_animation_sequence_timelines
    function(animations, **kwargs)


@pytest.mark.parametrize("kind", ("color", "deform", "sequence"))
def test_three_boundaries_accept_one_exact_index_and_direct_fallback(kind):
    slot_names = ("slot",)
    validate(kind, animation(kind), slot_names, SetupSlotIndex(slot_names))
    validate(kind, animation(kind), slot_names)


@pytest.mark.parametrize("kind", ("color", "deform", "sequence"))
def test_three_boundaries_reject_stale_index(kind):
    slot_names = ("slot",)
    stale = SetupSlotIndex(tuple(["slot"]))
    with pytest.raises(ValueError, match="exact slot_names tuple"):
        validate(kind, animation(kind), slot_names, stale)


@pytest.mark.parametrize("kind", ("color", "deform", "sequence"))
def test_three_boundaries_share_path_aware_undefined_slot_error(kind):
    slot_names = ("slot",)
    with pytest.raises(ValueError, match="undefined slot 'missing'") as error:
        validate(kind, animation(kind, "missing"), slot_names, SetupSlotIndex(slot_names))
    expected = ".slots.missing" if kind == "color" else ".attachments.default.missing"
    assert expected in str(error.value)
