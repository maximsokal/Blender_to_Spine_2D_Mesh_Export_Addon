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


def timeline_helper_source() -> str:
    source = read(MODEL)
    helper_start = source.index("def _validate_animation_event_timelines(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    return source[helper_start:next_helper]


def test_event_timeline_validation_runs_after_recursive_json_and_setup_events():
    source = read(MODEL)
    document_start = source.index("class SpineDocument:")
    document_source = source[document_start:]

    animation_json_index = document_source.index(
        'validate_json_mapping(self.animations, path="document.animations")'
    )
    event_json_index = document_source.index(
        'validate_json_mapping(self.events, path="document.events")'
    )
    setup_index = document_source.index(
        '_validate_event_definitions(self.events, path="document.events")'
    )
    timeline_index = document_source.index(
        "_validate_animation_event_timelines("
    )

    assert animation_json_index < timeline_index
    assert event_json_index < setup_index < timeline_index


def test_event_timeline_reuses_setup_event_scalar_contracts():
    source = read(MODEL)

    assert '_EVENT_TIMELINE_STRING_FIELDS = ("string",)' in source
    assert '_EVENT_NUMBER_FIELDS = ("float", "volume", "balance")' in source
    assert '_EVENT_INT_MIN = -(2**31)' in source
    assert '_EVENT_INT_MAX = 2**31 - 1' in source

    helper = timeline_helper_source()
    assert "int_value < _EVENT_INT_MIN" in helper
    assert "int_value > _EVENT_INT_MAX" in helper
    assert "for field_name in _EVENT_NUMBER_FIELDS:" in helper


def test_animation_and_event_containers_are_fail_closed():
    helper = timeline_helper_source()

    assert "if not isinstance(animation_metadata, Mapping):" in helper
    assert 'if "events" not in animation_metadata:' in helper
    assert "if not isinstance(timeline, (list, tuple)):" in helper
    assert "if not timeline:" in helper
    assert "if not isinstance(keyframe, Mapping):" in helper


def test_event_names_use_json_paths_and_reference_setup_definitions():
    helper = timeline_helper_source()

    assert "animation_path = json_path_key(path, animation_name)" in helper
    assert 'if "name" not in keyframe:' in helper
    assert '_require_name(event_name, f"{keyframe_path}.name")' in helper
    assert "if event_name not in event_names:" in helper
    assert "references undefined event" in helper


def test_setup_event_mapping_keys_are_non_empty_names():
    source = read(MODEL)
    helper_start = source.index("def _validate_event_definitions(")
    next_helper = source.index(
        "def _validate_animation_event_timelines(",
        helper_start,
    )
    helper = source[helper_start:next_helper]

    assert '_require_name(event_name, f"{event_path} event name")' in helper


def test_event_timeline_times_are_finite_and_non_decreasing():
    helper = timeline_helper_source()

    assert 'time_value = keyframe.get("time", 0)' in helper
    assert "isinstance(time_value, bool)" in helper
    assert "not isinstance(" in helper
    assert "if not _is_finite_number(time_value):" in helper
    assert "time_value < previous_time" in helper
    assert "time_value <= previous_time" not in helper


def test_event_timeline_does_not_insert_defaults_or_unproven_ranges():
    helper = timeline_helper_source()

    assert "setdefault(" not in helper
    assert "keyframe[\"time\"] =" not in helper
    assert "volume <" not in helper
    assert "volume >" not in helper
    assert "balance <" not in helper
    assert "balance >" not in helper


def test_serializer_keeps_animation_mapping_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
