from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_composition_result import (
    replace_a1_composition_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    ConnectedConstraintSchedule,
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectPlacement,
    ConnectedZLayer,
    SpineDocument,
    SpineDocumentCompositionResult,
)


def _document(name):
    return SpineDocument(
        skeleton={"spine": "4.2.43", "name": name},
        bones=(Bone("root"),),
        slots=(),
        skins=(),
        animations={},
    )


def _composition(document):
    return SpineDocumentCompositionResult(
        document=document,
        components=(),
        bone_index_maps=(),
        constraint_orders=(),
        animation_names=(),
    )


def _schedule():
    component = "object"
    return ConnectedConstraintSchedule(
        global_rotation_x=0,
        global_rotation_y=1,
        global_rotation_z=2,
        object_rotation_x=((component, 3),),
        object_rotation_y=((component, 4),),
        global_scale_ik=5,
        object_scale_ik=((component, 6),),
        global_scale=7,
        object_scale=((component, 8),),
        object_rotation_z=((component, 9),),
        object_scale_compensator=((component, 10),),
    )


def _connected_result(document):
    return ConnectedGroupBuildResult(
        document=document,
        composition=_composition(document),
        settings=ConnectedGroupSettings(
            texture_width=64,
            texture_height=64,
        ),
        layers=(
            ConnectedZLayer(
                layer_index=0,
                representative_relative_z=0.0,
                component_ids=("object",),
                scale_bone_name="all_objects_scale_0",
                layer_bone_name="all_objects_layer_0",
            ),
        ),
        placements=(
            ConnectedObjectPlacement(
                component_id="object",
                prefix="Object",
                relative_x=0.0,
                relative_y=0.0,
                relative_z=0.0,
                layer_index=0,
                main_bone_name="Object_main",
                parent_layer_bone_name="all_objects_layer_0",
            ),
        ),
        constraint_schedule=_schedule(),
        uniform_scale=1.0,
    )


def test_plain_composition_replacement_updates_its_single_document_owner():
    original = _composition(_document("original"))
    replacement = _document("replacement")

    result = replace_a1_composition_document(original, replacement)

    assert result.document is replacement


def test_connected_replacement_updates_top_level_and_nested_document_owners():
    original = _connected_result(_document("original"))
    replacement = _document("replacement")

    result = replace_a1_composition_document(original, replacement)

    assert isinstance(result, ConnectedGroupBuildResult)
    assert result.document is replacement
    assert result.composition.document is replacement
