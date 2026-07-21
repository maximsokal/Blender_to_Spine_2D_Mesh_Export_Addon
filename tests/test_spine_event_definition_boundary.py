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


def test_event_validation_runs_after_recursive_json_validation():
    source = read(MODEL)
    document_start = source.index("class SpineDocument:")
    document_source = source[document_start:]

    json_index = document_source.index(
        'validate_json_mapping(self.events, path="document.events")'
    )
    event_index = document_source.index(
        '_validate_event_definitions(self.events, path="document.events")'
    )

    assert json_index < event_index


def test_event_known_field_sets_match_runtime_scalar_types():
    source = read(MODEL)

    assert '_EVENT_STRING_FIELDS = ("string", "audio")' in source
    assert '_EVENT_NUMBER_FIELDS = ("float", "volume", "balance")' in source
    assert '_EVENT_INT_MIN = -(2**31)' in source
    assert '_EVENT_INT_MAX = 2**31 - 1' in source


def test_event_entries_require_mappings_and_use_json_paths():
    source = read(MODEL)
    helper_start = source.index("def _validate_event_definitions(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    helper_source = source[helper_start:next_helper]

    assert "event_path = json_path_key(path, event_name)" in helper_source
    assert "if not isinstance(event_metadata, Mapping):" in helper_source
    assert 'raise TypeError(f"{event_path} must be a mapping")' in helper_source


def test_event_int_is_strict_and_cross_runtime_bounded():
    source = read(MODEL)
    helper_start = source.index("def _validate_event_definitions(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    helper_source = source[helper_start:next_helper]

    assert "isinstance(int_value, bool)" in helper_source
    assert "not isinstance(int_value, int)" in helper_source
    assert "int_value < _EVENT_INT_MIN" in helper_source
    assert "int_value > _EVENT_INT_MAX" in helper_source


def test_event_numbers_reuse_finite_number_semantics_without_ranges():
    source = read(MODEL)
    helper_start = source.index("def _validate_event_definitions(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    helper_source = source[helper_start:next_helper]

    assert "isinstance(value, bool)" in helper_source
    assert "not isinstance(value, (int, float))" in helper_source
    assert "if not _is_finite_number(value):" in helper_source
    assert "volume <" not in helper_source
    assert "volume >" not in helper_source
    assert "balance <" not in helper_source
    assert "balance >" not in helper_source


def test_event_helper_does_not_require_fields_or_insert_defaults():
    source = read(MODEL)
    helper_start = source.index("def _validate_event_definitions(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    helper_source = source[helper_start:next_helper]

    assert 'if "int" in event_metadata:' in helper_source
    assert "if field_name in event_metadata" in helper_source
    assert "if field_name not in event_metadata:" in helper_source
    assert "setdefault(" not in helper_source
    assert '"MISSING_' not in helper_source


def test_serializer_passes_events_through_without_defaults():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["events"] = dict(document.events)' in to_dict_source
    assert 'document.events.setdefault(' not in to_dict_source
