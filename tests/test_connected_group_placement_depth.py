from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_assembly import (
    apply_object_placements,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (
    ConnectedGroupSettings,
    ConnectedObjectPlacement,
    ConnectedPlacementSpace,
    ConnectedZLayer,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_global_rig import (
    build_global_bones_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_profile import LegacyRigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    SpineDocument,
)


def _bone_by_name(document):
    return {bone.name: bone for bone in document.bones}


def test_connected_object_main_xy_adds_relative_world_to_document_local_offset():
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(
            Bone("root"),
            Bone("Anchor_main", parent="root", x=25.0, y=-40.0),
            Bone("Other_main", parent="root", x=-10.0, y=15.0),
        ),
        slots=(),
        skins=(),
        animations={},
    )
    placements = (
        ConnectedObjectPlacement(
            component_id="anchor",
            prefix="Anchor",
            relative_x=0.0,
            relative_y=0.0,
            relative_z=0.0,
            layer_index=0,
            main_bone_name="Anchor_main",
            parent_layer_bone_name="all_objects_layer_0",
        ),
        ConnectedObjectPlacement(
            component_id="other",
            prefix="Other",
            relative_x=2.0,
            relative_y=3.0,
            relative_z=-4.0,
            layer_index=1,
            main_bone_name="Other_main",
            parent_layer_bone_name="all_objects_layer_1",
        ),
    )

    result = apply_object_placements(document, placements, uniform_scale=100.0)
    bones = _bone_by_name(result)

    assert bones["Anchor_main"].parent == "all_objects_layer_0"
    assert bones["Anchor_main"].x == 25.0
    assert bones["Anchor_main"].y == -40.0
    assert bones["Other_main"].parent == "all_objects_layer_1"
    assert bones["Other_main"].x == 190.0
    assert bones["Other_main"].y == 315.0


def test_camera_projection_main_xy_is_preserved_while_reparenting():
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(
            Bone("root"),
            Bone("Camera_main", parent="root", x=0.0, y=0.0),
        ),
        slots=(),
        skins=(),
        animations={},
    )
    placement = ConnectedObjectPlacement(
        component_id="camera",
        prefix="Camera",
        relative_x=12.0,
        relative_y=-8.0,
        relative_z=3.0,
        layer_index=0,
        main_bone_name="Camera_main",
        parent_layer_bone_name="all_objects_layer_0",
        placement_space=ConnectedPlacementSpace.PRESERVE_DOCUMENT,
    )

    result = apply_object_placements(
        document,
        (placement,),
        uniform_scale=100.0,
    )
    camera_main = _bone_by_name(result)["Camera_main"]

    assert camera_main.parent == "all_objects_layer_0"
    assert camera_main.x == 0.0
    assert camera_main.y == 0.0


def test_connected_global_layers_encode_relative_z_with_single_rig_bone_pattern():
    layers = (
        ConnectedZLayer(
            layer_index=0,
            representative_relative_z=2.0,
            component_ids=("front",),
            scale_bone_name="all_objects_0_scale",
            layer_bone_name="all_objects_layer_0",
        ),
        ConnectedZLayer(
            layer_index=1,
            representative_relative_z=-1.5,
            component_ids=("back",),
            scale_bone_name="all_objects_1_scale",
            layer_bone_name="all_objects_layer_1",
        ),
    )
    settings = ConnectedGroupSettings(
        texture_width=100,
        texture_height=100,
        group_prefix="all_objects",
    )

    document = build_global_bones_document(
        {"spine": "4.2.43"},
        layers,
        settings,
        LegacyRigProfile(),
        uniform_scale=100.0,
    )
    bones = _bone_by_name(document)

    front_scale = bones["all_objects_0_scale"]
    front_layer = bones["all_objects_layer_0"]
    back_scale = bones["all_objects_1_scale"]
    back_layer = bones["all_objects_layer_1"]

    assert front_scale.y == 200.0
    assert back_scale.y == -150.0
    for scale_bone in (front_scale, back_scale):
        assert scale_bone.rotation == 90.0
        assert scale_bone.extras["inherit"] == "onlyTranslation"
    assert front_layer.rotation == -90.0
    assert back_layer.rotation == -90.0
