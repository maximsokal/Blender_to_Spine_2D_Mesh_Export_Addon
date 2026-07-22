from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "slot_color_timeline_contract.py"
)
SERIALIZER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "serializer.py"
)
MODEL = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "model.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_slot_color_contract_owns_exact_known_timeline_schema():
    source = read(CONTRACT)

    assert '"rgba": (("color", 8),)' in source
    assert '"rgb": (("color", 6),)' in source
    assert '"rgba2": (("light", 8), ("dark", 6))' in source
    assert '"rgb2": (("light", 6), ("dark", 6))' in source
    assert 'frozenset((*_SLOT_COLOR_FIELDS, "alpha"))' in source


def test_serializer_runs_color_contract_after_general_validator():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    validator_index = to_dict_source.index(
        "self._validator.validate_or_raise(document)"
    )
    color_index = to_dict_source.index(
        "validate_animation_slot_color_timelines("
    )
    data_index = to_dict_source.index('data: dict[str, Any] = {')

    assert validator_index < color_index < data_index
    assert "slot_names=slot_names" in to_dict_source
    assert "setup_slot_index=setup_slot_index" in to_dict_source
    assert 'path="document.animations"' in to_dict_source


def test_color_contract_is_a_separate_output_boundary_not_model_mutation():
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    assert "slot_color_timeline_contract" not in model_source
    assert (
        "from .slot_color_timeline_contract import ("
        in serializer_source
    )


def test_known_timeline_containers_and_keyframes_are_fail_closed():
    source = read(CONTRACT)

    assert "if not isinstance(timeline, (list, tuple)):" in source
    assert "if not timeline:" in source
    assert "if not isinstance(keyframe, Mapping):" in source
    assert "if field_name not in keyframe:" in source
    assert "is required" in source


def test_rgb_and_rgba_formats_are_exact_without_normalization():
    source = read(CONTRACT)

    assert 'fullmatch(' in source
    assert 'rf"[0-9A-Fa-f]{{{digits}}}"' in source
    assert 'color_kind = "RGBA" if digits == 8 else "RGB"' in source
    assert 'value.startswith("#")' not in source
    assert ".upper()" not in source
    assert ".lower()" not in source
    assert ".strip()" not in source[source.index("def _require_hex_color("):]


def test_alpha_value_is_optional_finite_and_has_no_invented_range():
    source = read(CONTRACT)

    assert 'if timeline_name == "alpha":' in source
    assert 'if "value" in keyframe:' in source
    assert 'keyframe["value"]' in source
    assert "_require_finite_number(" in source
    assert "0 <=" not in source
    assert "<= 1" not in source
    assert "min(" not in source
    assert "max(" not in source


def test_color_timeline_time_is_finite_and_non_decreasing():
    source = read(CONTRACT)

    assert 'time_value = keyframe.get("time", 0)' in source
    assert "previous_time: float | int | None = None" in source
    assert "time_value < previous_time" in source
    assert "time_value <= previous_time" not in source
    assert "_require_finite_number(" in source


def test_unknown_timelines_and_curves_are_not_rejected_or_rewritten():
    source = read(CONTRACT)

    assert "for timeline_name in _SLOT_COLOR_TIMELINE_NAMES:" in source
    assert "if timeline_name not in slot_metadata:" in source
    assert "Invalid timeline type" not in source
    assert 'keyframe["curve"]' not in source
    assert 'keyframe["time"] =' not in source
    assert 'slot_metadata[timeline_name] =' not in source
    assert "setdefault(" not in source
    assert ".sort(" not in source
    assert "sorted(" not in source


def test_serializer_keeps_animation_mapping_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
