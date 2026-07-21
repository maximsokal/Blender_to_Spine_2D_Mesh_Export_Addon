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


def sequence_source() -> str:
    source = read_source()
    sequence_start = source.index("class LegacyAttachmentSequence:")
    request_start = source.index("class LegacyMeshAttachmentRequest:", sequence_start)
    return source[sequence_start:request_start]


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
    assert '_require_integer(self.start, "start")' in source
    assert '_require_integer(self.digits, "digits", minimum=0)' in source
    assert 'self.hull,\n            "hull"' in source
    assert 'f"{field_name}[{value_index}]"' in source
    assert 'first_vertex_bone_index,\n        "first_vertex_bone_index"' in source


def test_legacy_sequence_has_runtime_ranges_but_preserves_legacy_defaults():
    source = sequence_source()

    assert "digits: int = 4" in source
    assert '_require_integer(self.start, "start")' in source
    assert '_require_integer(self.digits, "digits", minimum=0)' in source
    assert 'minimum=0,\n                maximum=self.count - 1' in source
    assert '_require_integer(self.start, "start", minimum=0)' not in source
    assert "minimum=1, maximum=12" not in source


def test_legacy_sequence_mapping_policy_remains_explicit_and_unchanged():
    source = sequence_source()

    assert "return 1 if self.count > 1 else 0" in source
    assert '"count": self.count' in source
    assert '"start": self.start' in source
    assert '"digits": self.digits' in source
    assert '"setup": self.resolved_setup' in source
    assert "setdefault(" not in source


def test_valid_spine_payload_mapping_code_remains_unchanged():
    source = read_source()

    assert "uvs = tuple(" in source
    assert "triangles=request.triangles" in source
    assert "hull=request.hull" in source
    assert "edges=request.edges" in source
    assert "sequence=sequence_mapping" in source
    assert "SpineValidator().validate_or_raise(document)" in source
