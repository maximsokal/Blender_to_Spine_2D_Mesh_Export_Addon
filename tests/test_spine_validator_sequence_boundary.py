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


def sequence_source() -> str:
    source = read(VALIDATOR)
    sequence_start = source.index("def _validate_attachment_sequence(")
    payload_start = source.index("def _validate_mesh_payload(", sequence_start)
    return source[sequence_start:payload_start]


def raw_attachment_source() -> str:
    source = read(VALIDATOR)
    raw_start = source.index("def _validate_raw_attachment(")
    typed_start = source.index("def _validate_mesh_attachment(", raw_start)
    return source[raw_start:typed_start]


def test_typed_mesh_and_supported_raw_types_share_one_sequence_validator():
    source = read(VALIDATOR)
    raw_source = raw_attachment_source()

    assert (
        '_SEQUENCE_ATTACHMENT_TYPES = frozenset('
        '{"region", "mesh", "linkedmesh"}'
        ')' in source
    )
    assert source.count("self._validate_attachment_sequence(") == 2
    assert "attachment_type in _SEQUENCE_ATTACHMENT_TYPES" in raw_source


def test_raw_sequence_validation_runs_before_mesh_only_return():
    source = raw_attachment_source()

    sequence_index = source.index('sequence = attachment.get("sequence")')
    type_filter_index = source.index('if attachment_type != "mesh":')
    mesh_payload_index = source.index("self._validate_mesh_payload(")

    assert sequence_index < type_filter_index < mesh_payload_index


def test_count_is_the_only_required_setup_sequence_field():
    source = sequence_source()

    assert 'if "count" not in sequence:' in source
    assert 'if "start" in sequence:' in source
    assert 'if "digits" in sequence:' in source
    assert 'if "setup" in sequence:' in source
    assert 'for required_field in ("count", "start"):' not in source
    assert 'f"{path}.start"' not in source[source.index('if "count" not in sequence:'):source.index('count: int | None')]


def test_runtime_defaults_are_documented_but_never_inserted():
    source = sequence_source()

    assert "Spine defaults ``start`` to 1" in source
    assert "``digits`` to 0" in source
    assert "``setup`` to 0" in source
    assert "setdefault(" not in source
    assert 'sequence["start"] =' not in source
    assert 'sequence["digits"] =' not in source
    assert 'sequence["setup"] =' not in source


def test_start_accepts_any_strict_integer_without_unproven_range():
    source = sequence_source()
    start_begin = source.index('if "start" in sequence:')
    digits_begin = source.index('if "digits" in sequence:', start_begin)
    start_source = source[start_begin:digits_begin]

    assert "isinstance(raw_start, bool)" in start_source
    assert "not isinstance(raw_start, int)" in start_source
    assert "raw_start < 0" not in start_source
    assert "raw_start >" not in start_source
    assert "Sequence start must be an integer" in start_source


def test_digits_accepts_zero_and_has_no_arbitrary_upper_bound():
    source = sequence_source()
    digits_begin = source.index('if "digits" in sequence:')
    setup_begin = source.index('if "setup" in sequence:', digits_begin)
    digits_source = source[digits_begin:setup_begin]

    assert "isinstance(raw_digits, bool)" in digits_source
    assert "not isinstance(raw_digits, int)" in digits_source
    assert "raw_digits < 0" in digits_source
    assert "raw_digits < 1" not in digits_source
    assert "raw_digits > 12" not in digits_source
    assert "non-negative integer" in digits_source


def test_setup_range_depends_on_a_valid_count():
    source = sequence_source()

    assert "elif count is not None and raw_setup >= count:" in source
    assert "raw_setup < 0" in source


def test_sequence_validator_preserves_stable_issue_codes():
    source = sequence_source()

    for code in (
        "INVALID_SEQUENCE_MAPPING",
        "MISSING_SEQUENCE_FIELD",
        "INVALID_SEQUENCE_COUNT",
        "INVALID_SEQUENCE_START",
        "INVALID_SEQUENCE_DIGITS",
        "INVALID_SEQUENCE_SETUP",
    ):
        assert f'"{code}"' in source


def test_unsupported_attachment_types_remain_outside_sequence_contract():
    source = raw_attachment_source()

    assert "attachment_type in _SEQUENCE_ATTACHMENT_TYPES" in source
    for unsupported in ("point", "boundingbox", "path", "clipping"):
        assert f'"{unsupported}"' not in read(VALIDATOR).split(
            "_SEQUENCE_ATTACHMENT_TYPES =", 1
        )[1].split("\n\n", 1)[0]


def test_serializer_preserves_sequence_mapping_instead_of_inserting_defaults():
    source = read(SERIALIZER)
    attachment_start = source.index("def attachment_to_dict(")
    skin_start = source.index("def skin_to_dict(", attachment_start)
    attachment_source = source[attachment_start:skin_start]

    assert "dict(attachment.sequence) if attachment.sequence else None" in attachment_source
    assert "setdefault" not in attachment_source
