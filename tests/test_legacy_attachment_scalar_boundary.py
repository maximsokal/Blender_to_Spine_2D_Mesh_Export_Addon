from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ATTACHMENT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "legacy_attachment_builder.py"
)


def read_source() -> str:
    return LEGACY_ATTACHMENT.read_text(encoding="utf-8")


def test_legacy_attachment_uses_strict_integer_and_finite_pair_helpers():
    source = read_source()

    assert "def _require_integer(" in source
    assert "if isinstance(value, bool) or not isinstance(value, int):" in source
    assert "def _require_finite_number(" in source
    assert "if isinstance(value, bool) or not isinstance(value, (int, float)):" in source
    assert "def _require_finite_pair(" in source


def test_all_attachment_indices_use_the_strict_integer_helper():
    source = read_source()

    assert '_require_integer(self.index, "index", minimum=0)' in source
    assert '_require_integer(self.z_group_index, "z_group_index", minimum=0)' in source
    assert '_require_integer(self.count, "count", minimum=1)' in source
    assert '_require_integer(self.start, "start", minimum=0)' in source
    assert '_require_integer(self.digits, "digits", minimum=1, maximum=12)' in source
    assert 'self.hull,\n            "hull"' in source
    assert 'f"{field_name}[{value_index}]"' in source
    assert 'first_vertex_bone_index,\n        "first_vertex_bone_index"' in source


def test_valid_spine_payload_mapping_code_remains_unchanged():
    source = read_source()

    assert "uvs = tuple(" in source
    assert "triangles=request.triangles" in source
    assert "hull=request.hull" in source
    assert "edges=request.edges" in source
    assert "sequence=sequence_mapping" in source
    assert "SpineValidator().validate_or_raise(document)" in source
