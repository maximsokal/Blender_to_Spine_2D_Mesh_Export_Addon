from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.serializer import SpineSerializer
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.staged_text import (
    write_staged_utf8_text,
)


def _document() -> SpineDocument:
    attachment = MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        edges=(0, 1),
        width=64.0,
        height=64.0,
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", bone="root", attachment="mesh"),),
        skins=(Skin("default", {"slot": {"mesh": attachment}}),),
        animations={"animation": {}},
    )


def test_staged_utf8_writer_forces_lf_bytes_and_trailing_newline(tmp_path: Path):
    output = tmp_path / "unicode_герой_日本語.json"

    write_staged_utf8_text(
        output,
        "{\n  \"value\": 1\n}",
        ensure_trailing_newline=True,
    )

    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    assert payload.decode("utf-8") == "{\n  \"value\": 1\n}\n"


def test_spine_serializer_write_json_forces_lf_bytes(tmp_path: Path):
    output = tmp_path / "document.json"

    SpineSerializer().write_json(_document(), output, indent=2)

    payload = output.read_bytes()
    assert b"\r\n" not in payload
    assert payload.decode("utf-8") == SpineSerializer().to_json(_document(), indent=2)
