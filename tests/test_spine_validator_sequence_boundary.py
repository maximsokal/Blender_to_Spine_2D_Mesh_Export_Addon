from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "validator.py"
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


def test_raw_and_typed_meshes_share_one_sequence_validator():
    source = read(VALIDATOR)

    raw_index = source.index("def _validate_raw_attachment(")
    typed_index = source.index("def _validate_mesh_attachment(")
    sequence_index = source.index("def _validate_attachment_sequence(")

    assert source.count("self._validate_attachment_sequence(") == 2
    assert raw_index < sequence_index
    assert typed_index < sequence_index


def test_sequence_validator_requires_only_count_and_start():
    source = read(VALIDATOR)
    sequence_start = source.index("def _validate_attachment_sequence(")
    payload_start = source.index("def _validate_mesh_payload(", sequence_start)
    sequence_source = source[sequence_start:payload_start]

    assert 'for required_field in ("count", "start"):' in sequence_source
    assert 'for required_field in ("count", "start", "digits", "setup"):' not in sequence_source
    assert 'if "digits" in sequence:' in sequence_source
    assert 'if "setup" in sequence:' in sequence_source


def test_sequence_validator_owns_strict_scalar_issue_codes():
    source = read(VALIDATOR)

    for code in (
        "INVALID_SEQUENCE_MAPPING",
        "MISSING_SEQUENCE_FIELD",
        "INVALID_SEQUENCE_COUNT",
        "INVALID_SEQUENCE_START",
        "INVALID_SEQUENCE_DIGITS",
        "INVALID_SEQUENCE_SETUP",
    ):
        assert f'"{code}"' in source

    assert "isinstance(raw_count, bool)" in source
    assert "isinstance(raw_start, bool)" in source
    assert "isinstance(raw_digits, bool)" in source
    assert "not isinstance(raw_setup, bool)" in source


def test_setup_range_depends_on_a_valid_count():
    source = read(VALIDATOR)
    sequence_start = source.index("def _validate_attachment_sequence(")
    payload_start = source.index("def _validate_mesh_payload(", sequence_start)
    sequence_source = source[sequence_start:payload_start]

    assert "elif count is not None and raw_setup >= count:" in sequence_source


def test_serializer_preserves_sequence_mapping_instead_of_inserting_defaults():
    source = read(SERIALIZER)
    attachment_start = source.index("def attachment_to_dict(")
    skin_start = source.index("def skin_to_dict(", attachment_start)
    attachment_source = source[attachment_start:skin_start]

    assert "dict(attachment.sequence) if attachment.sequence else None" in attachment_source
    assert "setdefault" not in attachment_source
