from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "model.py"
)
SERIALIZER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "serializer.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_typed_and_raw_attachments_share_one_metadata_helper():
    source = read(MODEL)

    helper_index = source.index("def _validate_attachment_metadata(")
    mesh_index = source.index("class MeshAttachment:")
    skin_index = source.index("class Skin:")

    assert source.count("_validate_attachment_metadata(") == 3
    assert helper_index < mesh_index < skin_index
    assert '_validate_attachment_metadata(self.extras, path="mesh.extras")' in source
    assert "_validate_attachment_metadata(attachment, path=attachment_path)" in source


def test_attachment_color_contract_matches_runtime_string_forms():
    source = read(MODEL)
    helper_index = source.index("def _validate_attachment_metadata(")
    finite_sequence_index = source.index("def _validate_finite_sequence(")
    helper_source = source[helper_index:finite_sequence_index]

    assert 'color.startswith("#")' in helper_source
    assert "len(normalized) not in (6, 8)" in helper_source
    assert 'fullmatch(r"[0-9A-Fa-f]+", normalized)' in helper_source


def test_metadata_validation_does_not_consume_or_rewrite_extra_fields():
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    mesh_index = model_source.index("class MeshAttachment:")
    skin_index = model_source.index("class Skin:", mesh_index)
    mesh_source = model_source[mesh_index:skin_index]

    assert '"name",' not in mesh_source[mesh_source.index("known_fields=("):]
    assert '"color",' not in mesh_source[mesh_source.index("known_fields=("):]
    assert "known.update(extras)" in serializer_source


def test_runtime_zero_defaults_are_not_replaced_by_positive_only_contracts():
    source = read(MODEL)
    mesh_index = source.index("class MeshAttachment:")
    skin_index = source.index("class Skin:", mesh_index)
    mesh_source = source[mesh_index:skin_index]

    assert '_require_non_negative_int(self.hull, "hull")' in mesh_source
    assert '_require_finite_number(self.width, "width")' in mesh_source
    assert '_require_finite_number(self.height, "height")' in mesh_source
    assert "hull must be at least 3" not in mesh_source
    assert "width must be positive" not in mesh_source
    assert "height must be positive" not in mesh_source
