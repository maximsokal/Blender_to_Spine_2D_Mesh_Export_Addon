from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_contracts import (
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedPlacementSpace,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_error import (
    ConnectedGroupBuildError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_validation import (
    validate_connected_group_inputs,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_profile import LegacyRigProfile
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import Bone, SpineDocument


ROOT = Path(__file__).resolve().parents[1]
COMPOSITION_SOURCE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_multi_object_composition.py"
)


def _minimal_document():
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(),
        skins=(),
        animations={},
    )


def test_connected_group_rejects_mixed_object_and_screen_space_components_first():
    document = _minimal_document()
    objects = (
        ConnectedObjectDocument(
            component_id="object",
            prefix="Object",
            document=document,
            world_position=(0.0, 0.0, 0.0),
            placement_space=ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD,
        ),
        ConnectedObjectDocument(
            component_id="camera",
            prefix="Camera",
            document=document,
            world_position=(1.0, 2.0, 3.0),
            placement_space=ConnectedPlacementSpace.PRESERVE_DOCUMENT,
        ),
    )

    with pytest.raises(ConnectedGroupBuildError, match="cannot mix") as captured:
        validate_connected_group_inputs(
            objects,
            ConnectedGroupSettings(texture_width=100, texture_height=100),
            LegacyRigProfile(),
        )

    assert "static grouped camera flattening" in str(captured.value)


def test_adapter_routes_camera_projection_from_actual_bake_plan_type():
    source = COMPOSITION_SOURCE.read_text(encoding="utf-8")

    assert "from ..domain.baking import CameraProjectionPlan" in source
    assert "isinstance(prepared.bake_plan, CameraProjectionPlan)" in source
    assert "ConnectedPlacementSpace.PRESERVE_DOCUMENT" in source
    assert "ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD" in source
    assert "placement_space=_connected_placement_space(item)" in source
