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


def test_skeleton_metadata_validation_runs_after_recursive_json_validation():
    source = read(MODEL)
    document_start = source.index("class SpineDocument:")
    document_source = source[document_start:]

    json_index = document_source.index(
        'validate_json_mapping(self.skeleton, path="document.skeleton")'
    )
    metadata_index = document_source.index(
        '_validate_skeleton_metadata(self.skeleton, path="document.skeleton")'
    )

    assert json_index < metadata_index


def test_skeleton_known_field_sets_match_runtime_metadata_types():
    source = read(MODEL)

    assert '_SKELETON_STRING_FIELDS = ("hash", "images", "audio")' in source
    assert '"referenceScale",' in source
    assert '"fps",' in source
    for field_name in ("x", "y", "width", "height"):
        assert f'"{field_name}",' in source


def test_skeleton_metadata_helper_does_not_require_optional_fields():
    source = read(MODEL)
    helper_start = source.index("def _validate_skeleton_metadata(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    helper_source = source[helper_start:next_helper]

    assert 'if "spine" in metadata:' in helper_source
    assert 'if field_name in metadata' in helper_source
    assert 'if field_name not in metadata:' in helper_source
    assert '"MISSING_' not in helper_source
    assert 'metadata.setdefault(' not in helper_source


def test_skeleton_numeric_metadata_rejects_bool_and_non_finite_values():
    source = read(MODEL)
    helper_start = source.index("def _validate_skeleton_metadata(")
    next_helper = source.index("def _validate_finite_sequence(", helper_start)
    helper_source = source[helper_start:next_helper]

    assert "isinstance(value, bool)" in helper_source
    assert "not isinstance(value, (int, float))" in helper_source
    assert "if not _is_finite_number(value):" in helper_source


def test_bone_icon_remains_string_typed_without_invented_enum():
    source = read(MODEL)
    bone_start = source.index("class Bone:")
    slot_start = source.index("class Slot:", bone_start)
    bone_source = source[bone_start:slot_start]

    assert '_require_optional_string(self.icon, "icon")' in bone_source
    assert "_BONE_ICON_VALUES" not in source


def test_serializer_passes_skeleton_mapping_through_without_defaults():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert '"skeleton": dict(document.skeleton)' in to_dict_source
    assert 'document.skeleton.setdefault(' not in to_dict_source
