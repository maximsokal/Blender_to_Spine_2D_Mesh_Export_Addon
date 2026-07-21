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


def draw_order_helper_source() -> str:
    source = read(MODEL)
    helper_start = source.index("def _validate_animation_draw_order_timelines(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    return source[helper_start:next_helper]


def test_draw_order_validation_runs_after_recursive_animation_validation():
    source = read(MODEL)
    document_start = source.index("class SpineDocument:")
    document_source = source[document_start:]

    json_index = document_source.index(
        'validate_json_mapping(self.animations, path="document.animations")'
    )
    event_index = document_source.index(
        "_validate_animation_event_timelines("
    )
    draw_order_index = document_source.index(
        "_validate_animation_draw_order_timelines("
    )

    assert json_index < event_index < draw_order_index


def test_draw_order_receives_setup_slot_order_without_hidden_sorting():
    source = read(MODEL)
    document_start = source.index("class SpineDocument:")
    document_source = source[document_start:]

    assert "slot_names=tuple(slot.name for slot in self.slots)" in document_source
    assert "slot_names=tuple(sorted(" not in document_source


def test_draw_order_timeline_and_keyframes_are_fail_closed():
    helper = draw_order_helper_source()

    assert 'if "drawOrder" not in animation_metadata:' in helper
    assert "if not isinstance(timeline, (list, tuple)):" in helper
    assert "if not timeline:" in helper
    assert "if not isinstance(keyframe, Mapping):" in helper
    assert 'if "offsets" not in keyframe:' in helper
    assert "if not isinstance(offsets, (list, tuple)):" in helper
    assert "if not isinstance(offset_entry, Mapping):" in helper


def test_draw_order_time_is_finite_and_non_decreasing():
    helper = draw_order_helper_source()

    assert 'time_value = keyframe.get("time", 0)' in helper
    assert "isinstance(time_value, bool)" in helper
    assert "if not _is_finite_number(time_value):" in helper
    assert "time_value < previous_time" in helper
    assert "time_value <= previous_time" not in helper


def test_draw_order_slot_references_are_exact_and_unambiguous():
    helper = draw_order_helper_source()

    assert "slot_index_by_name: dict[str, int] = {}" in helper
    assert "ambiguous_slot_names: set[str] = set()" in helper
    assert '_require_name(slot_name, f"{entry_path}.slot")' in helper
    assert "if slot_name in seen_slot_names:" in helper
    assert "if slot_name not in slot_index_by_name:" in helper
    assert "if slot_name in ambiguous_slot_names:" in helper


def test_draw_order_offset_entries_preserve_runtime_setup_order():
    helper = draw_order_helper_source()

    assert "previous_source_index = -1" in helper
    assert "source_index <= previous_source_index" in helper
    assert "must follow setup slot order" in helper
    assert ".sort(" not in helper
    assert "sorted(" not in helper


def test_draw_order_offsets_are_strict_integers_and_valid_targets():
    helper = draw_order_helper_source()

    assert 'if "offset" not in offset_entry:' in helper
    assert "isinstance(offset_value, bool)" in helper
    assert "not isinstance(offset_value, int)" in helper
    assert "target_index = source_index + offset_value" in helper
    assert "target_index < 0 or target_index >= slot_count" in helper
    assert "target_to_entry_index.get(target_index)" in helper
    assert "already used by" in helper


def test_draw_order_contract_does_not_normalize_or_insert_defaults():
    helper = draw_order_helper_source()

    assert "setdefault(" not in helper
    assert 'keyframe["time"] =' not in helper
    assert 'keyframe["offsets"] =' not in helper
    assert ".strip()" not in helper
    assert ".lower()" not in helper


def test_serializer_keeps_animation_mapping_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
