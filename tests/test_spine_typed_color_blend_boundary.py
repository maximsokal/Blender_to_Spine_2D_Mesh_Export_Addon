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


def test_bone_slot_and_attachment_share_one_rgba_helper():
    source = read(MODEL)

    assert source.count("_require_rgba_hex_string(") == 3
    assert '_require_optional_rgba_hex_string(self.color, "color")' in source
    assert source.count(
        '_require_optional_rgba_hex_string(self.color, "color")'
    ) == 2
    assert '_require_rgba_hex_string(metadata["color"], f"{path}.color")' in source


def test_rgba_helper_requires_exact_cross_runtime_representation():
    source = read(MODEL)
    rgba_start = source.index("def _require_rgba_hex_string(")
    optional_start = source.index("def _require_optional_rgba_hex_string(")
    rgba_source = source[rgba_start:optional_start]

    assert "len(value) != 8" in rgba_source
    assert 'fullmatch(r"[0-9A-Fa-f]{8}", value)' in rgba_source
    assert 'value.startswith("#")' not in rgba_source
    assert ".upper()" not in rgba_source
    assert ".lower()" not in rgba_source


def test_slot_blend_set_matches_official_four_modes():
    source = read(MODEL)

    assert (
        '_SLOT_BLEND_VALUES = frozenset('
        '{"normal", "additive", "multiply", "screen"}'
        ')' in source
    )
    assert "if value not in _SLOT_BLEND_VALUES:" in source
    assert '_require_optional_slot_blend(self.blend, "blend")' in source


def test_model_and_serializer_do_not_normalize_color_or_blend():
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    assert ".strip() in _SLOT_BLEND_VALUES" not in model_source
    assert ".lower() in _SLOT_BLEND_VALUES" not in model_source
    assert '_put_optional(data, "color", bone.color)' in serializer_source
    assert '_put_optional(data, "color", slot.color)' in serializer_source
    assert '_put_optional(data, "blend", slot.blend)' in serializer_source
