from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "validator.py"
)
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "linked_mesh_contract.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def raw_attachment_source() -> str:
    source = read(VALIDATOR)
    start = source.index("def _validate_raw_attachment(")
    end = source.index("def _validate_mesh_attachment(", start)
    return source[start:end]


def test_parent_bearing_mesh_returns_before_unlinked_geometry_validation():
    source = raw_attachment_source()

    sequence_index = source.index('sequence = attachment.get("sequence")')
    parent_index = source.index('parent = attachment.get("parent")')
    linked_return_index = source.index(
        'if attachment_type in {"mesh", "linkedmesh"} and parent:'
    )
    mesh_filter_index = source.index('if attachment_type != "mesh":')
    missing_fields_index = source.index("missing_fields = tuple(")

    assert (
        sequence_index
        < parent_index
        < linked_return_index
        < mesh_filter_index
        < missing_fields_index
    )


def test_validator_matches_runtime_parent_truthiness_without_rewriting_payload():
    source = raw_attachment_source()

    assert 'parent = attachment.get("parent")' in source
    assert 'and parent:' in source
    assert 'parent is not None' not in source
    assert 'attachment["parent"] =' not in source
    assert "bool(parent)" not in source


def test_shared_resolver_uses_the_same_truthiness_for_mesh_parent_spelling():
    source = read(CONTRACT)

    assert (
        'return attachment_type == "mesh" and '
        'bool(attachment.get("parent"))'
    ) in source
    assert 'attachment.get("parent") is not None' not in source


def test_canonical_linkedmesh_still_requires_parent_name():
    source = read(CONTRACT)

    assert "if attachment_type in _LINKED_MESH_TYPES:" in source
    assert 'if "parent" not in attachment or attachment["parent"] is None:' in source
    assert "parent is required for a linked mesh" in source
    assert "_require_name(" in source


def test_mesh_payload_validator_remains_unchanged_for_unlinked_meshes():
    source = raw_attachment_source()

    for field_name in ("uvs", "triangles", "vertices", "hull"):
        assert f'"{field_name}"' in source
    assert "self._validate_mesh_payload(" in source
