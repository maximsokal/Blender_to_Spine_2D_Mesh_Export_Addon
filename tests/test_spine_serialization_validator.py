import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.serializer import SpineSerializer
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.validator import SpineValidationError


def _raw_mesh(*, edges):
    return {
        "type": "mesh",
        "uvs": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
        "triangles": [0, 1, 2],
        "vertices": [-50.0, 50.0, 50.0, 50.0, 0.0, -50.0],
        "hull": 3,
        "edges": list(edges),
        "width": 100.0,
        "height": 100.0,
    }


def _document(attachment):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("mesh_slot", "root", attachment="mesh"),),
        skins=(
            Skin(
                name="default",
                attachments={"mesh_slot": {"mesh": attachment}},
            ),
        ),
        animations={"animation": {}},
    )


def test_serializer_accepts_raw_mesh_edges_in_official_spine_offset_space():
    raw = _raw_mesh(edges=(0, 2, 2, 4, 4, 0))

    data = SpineSerializer().to_dict(_document(raw))
    serialized = data["skins"][0]["attachments"]["mesh_slot"]["mesh"]

    # Raw mappings are already serialized JSON and must not be doubled again.
    assert serialized["edges"] == [0, 2, 2, 4, 4, 0]


def test_serializer_rejects_odd_raw_mesh_edge_offsets_with_specific_issue():
    raw = _raw_mesh(edges=(0, 1))

    with pytest.raises(SpineValidationError) as captured:
        SpineSerializer().to_dict(_document(raw))

    assert any(
        issue.code == "INVALID_SERIALIZED_MESH_EDGES"
        and issue.path.endswith(".edges")
        for issue in captured.value.issues
    )


def test_typed_validator_override_remains_supported():
    from Blender_to_Spine2D_Mesh_Exporter.domain.spine.validator import SpineValidator

    serializer = SpineSerializer(validator=SpineValidator())
    assert serializer is not None
