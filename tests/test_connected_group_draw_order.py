from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_draw_order import (
    apply_connected_setup_draw_order,
    connected_draw_order_component_ids,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_error import (
    ConnectedGroupBuildError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    Slot,
    SpineDocument,
)


def _document(prefix, slot_suffixes):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=tuple(
            Slot(f"{prefix}_{suffix}", "root") for suffix in slot_suffixes
        ),
        skins=(),
        animations={},
    )


def _object(component_id, prefix, slot_suffixes, z):
    return ConnectedObjectDocument(
        component_id=component_id,
        prefix=prefix,
        document=_document(prefix, slot_suffixes),
        world_position=(0.0, 0.0, float(z)),
    )


def _placement(item, *, layer_index, relative_z):
    return ConnectedObjectPlacement(
        component_id=item.component_id,
        prefix=item.prefix,
        relative_x=0.0,
        relative_y=0.0,
        relative_z=float(relative_z),
        layer_index=layer_index,
        main_bone_name=f"{item.prefix}_main",
        parent_layer_bone_name=f"all_objects_layer_{layer_index}",
    )


def test_connected_component_draw_order_is_back_to_front_by_z_layer():
    front = _object("front", "Front", ("control", "mesh"), 2.0)
    middle = _object("middle", "Middle", ("mesh",), 0.0)
    back = _object("back", "Back", ("mesh",), -1.0)
    objects = (front, middle, back)
    placements = (
        _placement(front, layer_index=0, relative_z=2.0),
        _placement(middle, layer_index=1, relative_z=0.0),
        _placement(back, layer_index=2, relative_z=-1.0),
    )
    composed = _document("Combined", ())
    composed = SpineDocument(
        skeleton=composed.skeleton,
        bones=composed.bones,
        slots=tuple(
            slot for item in objects for slot in item.document.slots
        ),
        skins=(),
        animations={},
    )

    reordered = apply_connected_setup_draw_order(
        composed,
        objects,
        placements,
    )

    assert connected_draw_order_component_ids(placements) == (
        "back",
        "middle",
        "front",
    )
    assert tuple(slot.name for slot in reordered.slots) == (
        "Back_mesh",
        "Middle_mesh",
        "Front_control",
        "Front_mesh",
    )


def test_same_z_layer_preserves_source_input_order():
    first = _object("first", "First", ("mesh",), 0.0)
    second = _object("second", "Second", ("mesh",), 0.00001)
    placements = (
        _placement(first, layer_index=0, relative_z=0.0),
        _placement(second, layer_index=0, relative_z=0.00001),
    )

    assert connected_draw_order_component_ids(placements) == (
        "first",
        "second",
    )


def test_draw_order_rejects_unowned_composed_slots():
    item = _object("object", "Object", ("mesh",), 0.0)
    placement = _placement(item, layer_index=0, relative_z=0.0)
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("Object_mesh", "root"), Slot("foreign", "root")),
        skins=(),
        animations={},
    )

    with pytest.raises(ConnectedGroupBuildError, match="cannot account"):
        apply_connected_setup_draw_order(
            document,
            (item,),
            (placement,),
        )


def test_draw_order_rejects_missing_or_unknown_placement_ownership():
    first = _object("first", "First", ("mesh",), 0.0)
    second = _object("second", "Second", ("mesh",), 1.0)

    with pytest.raises(ConnectedGroupBuildError, match="ownership mismatch"):
        apply_connected_setup_draw_order(
            SpineDocument(
                skeleton={"spine": "4.2.43"},
                bones=(Bone("root"),),
                slots=(Slot("First_mesh", "root"), Slot("Second_mesh", "root")),
                skins=(),
                animations={},
            ),
            (first, second),
            (_placement(first, layer_index=0, relative_z=0.0),),
        )


def test_connected_draw_order_rejects_existing_unrebased_draworder_timeline():
    item = _object("object", "Object", ("mesh",), 0.0)
    animated_document = replace(
        item.document,
        animations={
            "animation": {
                "draworder": [
                    {
                        "time": 0.0,
                        "offsets": [
                            {"slot": "Object_mesh", "offset": 0},
                        ],
                    },
                ],
            },
        },
    )
    animated_item = replace(item, document=animated_document)

    with pytest.raises(ConnectedGroupBuildError, match="explicitly rebased"):
        apply_connected_setup_draw_order(
            animated_document,
            (animated_item,),
            (_placement(animated_item, layer_index=0, relative_z=0.0),),
        )
