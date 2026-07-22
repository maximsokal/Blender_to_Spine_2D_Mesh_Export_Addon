from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "curve_timeline_contract.py"
)
SCALAR = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "spine_scalar_contract.py"
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
LEGACY_VISUALS = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "legacy_visuals.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_curve_contract_owns_exact_spine_42_channel_counts():
    source = read(CONTRACT)

    for expected in (
        '"rgba": 4',
        '"rgb": 3',
        '"alpha": 1',
        '"rgba2": 7',
        '"rgb2": 6',
        '"rotate": 1',
        '"translate": 2',
        '"scale": 2',
        '"shear": 2',
        '"position": 1',
        '"spacing": 1',
        '"mix": 3',
        '"ik": 2',
        '"transform": 6',
    ):
        assert expected in source

    assert '"physics": _PHYSICS_CURVE_CHANNELS' in source
    for timeline_name in (
        "inertia",
        "strength",
        "damping",
        "mass",
        "wind",
        "gravity",
    ):
        assert f'"{timeline_name}": 1' in source


def test_curve_contract_accepts_only_stepped_or_exact_bezier_sequence():
    source = read(CONTRACT)

    assert 'if curve == "stepped":' in source
    assert 'must be exactly "stepped" or a Bezier number sequence' in source
    assert "if not isinstance(curve, (list, tuple)):" in source
    assert "expected_length = channel_count * 4" in source
    assert "if len(curve) != expected_length:" in source
    assert "for value_index, value in enumerate(curve):" in source


def test_curve_numbers_and_times_reuse_shared_strict_finite_requirement():
    source = read(CONTRACT)
    scalar_source = read(SCALAR)

    assert (
        "from .spine_scalar_contract import require_finite_number as "
        "_require_finite_number"
    ) in source
    assert "def _require_finite_number(" not in source
    assert "from math import isfinite" not in source
    assert "def require_finite_number(" in scalar_source
    assert "isinstance(value, bool)" in scalar_source
    assert "not isinstance(value, (int, float))" in scalar_source
    assert "if not is_finite_number(value):" in scalar_source
    assert 'time_value = keyframe.get("time", 0)' in source
    assert "time_value < previous_time" in source
    assert "time_value <= previous_time" not in source


def test_curve_contract_does_not_clamp_or_normalize_absolute_controls():
    source = read(CONTRACT)

    for forbidden in (
        ".lower()",
        ".upper()",
        ".strip()",
        "min(",
        "max(",
        "0 <=",
        "<= 1",
        ".sort(",
        "sorted(",
        'keyframe["curve"] =',
        'keyframe["time"] =',
        "setdefault(",
    ):
        assert forbidden not in source


def test_discrete_and_attachment_specific_timelines_remain_outside_contract():
    source = read(CONTRACT)

    for discrete_name in (
        '"attachment"',
        '"inherit"',
        '"reset"',
        '"events"',
        '"drawOrder"',
        '"deform"',
        '"sequence"',
    ):
        assert discrete_name not in source

    assert "unknown" in source
    assert "future timeline kinds" in source


def test_serializer_runs_curve_contract_after_existing_boundaries():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    validator_index = to_dict_source.index(
        "self._validator.validate_or_raise(document)"
    )
    color_index = to_dict_source.index(
        "validate_animation_slot_color_timelines("
    )
    curve_index = to_dict_source.index("validate_animation_curves(")
    data_index = to_dict_source.index('data: dict[str, Any] = {')

    assert validator_index < color_index < curve_index < data_index
    assert 'path="document.animations"' in to_dict_source


def test_curve_contract_is_output_boundary_not_model_mutation():
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    assert "curve_timeline_contract" not in model_source
    assert "from .curve_timeline_contract import validate_animation_curves" in (
        serializer_source
    )


def test_legacy_preview_keeps_absolute_four_number_curves_and_stepped():
    source = read(LEGACY_VISUALS)

    assert '"curve": [0.667, 0, 1.333, -360]' in source
    assert '"curve": [2.667, -360, 3.333, -360]' in source
    assert '"curve": "stepped"' in source
    assert "normalized_curve" not in source


def test_serializer_keeps_animation_mapping_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
    assert 'document.animations["' not in to_dict_source
