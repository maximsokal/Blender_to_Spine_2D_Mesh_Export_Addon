"""Sequence rejection must identify the exact Spine 3.8 target."""

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def test_spine38_sequence_error_names_exact_target_and_attachment_path() -> None:
    mesh = MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        sequence={"count": 2, "start": 0, "digits": 4, "setup": 0},
    )
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", bone="root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": mesh}}),),
        animations={},
    )

    with pytest.raises(ValueError, match="Spine 3.8.99") as exc_info:
        serialize_spine_document(document, SpineJsonTarget.SPINE_3_8)

    message = str(exc_info.value)
    assert "document.skins[0].attachments.slot.mesh.sequence" in message
    assert "Spine 4.0.64" not in message
