from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINKED = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "linked_mesh_contract.py"
)
DEFORM = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "deform_timeline_contract.py"
)
VALIDATOR = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "validator.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_linked_metadata_contract_owns_only_runtime_consumed_fields():
    source = read(LINKED)

    assert 'for field_name in ("name", "path"):' in source
    assert 'if "timelines" in attachment' in source
    assert 'attachment["timelines"],\n        bool' in source
    assert 'sequence = attachment.get("sequence")' in source
    assert "isinstance(sequence, Mapping)" in source
    assert '_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")' in source
    assert "len(hexadecimal) not in (6, 8)" in source
    assert 'color.startswith("#")' in source


def test_linked_metadata_defaults_are_never_materialized():
    source = read(LINKED)

    for forbidden in (
        'attachment["timelines"] =',
        'attachment["name"] =',
        'attachment["path"] =',
        'attachment["color"] =',
        'attachment["sequence"] =',
        "setdefault(",
        ".lower()",
        ".upper()",
        ".sort(",
        "sorted(",
    ):
        assert forbidden not in source


def test_sequence_scalar_contract_remains_in_spine_validator():
    linked_source = read(LINKED)
    validator_source = read(VALIDATOR)

    assert "INVALID_SEQUENCE_COUNT" not in linked_source
    assert "INVALID_SEQUENCE_START" not in linked_source
    assert "INVALID_SEQUENCE_DIGITS" not in linked_source
    assert "INVALID_SEQUENCE_SETUP" not in linked_source

    for issue_code in (
        "INVALID_SEQUENCE_COUNT",
        "INVALID_SEQUENCE_START",
        "INVALID_SEQUENCE_DIGITS",
        "INVALID_SEQUENCE_SETUP",
    ):
        assert issue_code in validator_source


def test_setup_resolver_exposes_one_shared_attachment_index():
    source = read(LINKED)

    assert "class SetupAttachment:" in source
    assert "def require_skin(" in source
    assert "def get_attachment(" in source
    assert "terminal_path: str" in source
    assert '"SetupAttachment"' in source
    assert '"is_linked_mesh_attachment"' in source
    assert '"raw_attachment_type"' in source


def test_deform_contract_reuses_shared_setup_resolution():
    source = read(DEFORM)

    assert "from .linked_mesh_contract import (" in source
    assert "AttachmentReference," in source
    assert "LinkedMeshResolver," in source
    assert "is_linked_mesh_attachment," in source
    assert "raw_attachment_type," in source
    assert 'resolver = LinkedMeshResolver(skins, path="document.skins")' in source
    assert "resolver.require_skin(skin_name, path=skin_path)" in source
    assert "setup = resolver.get_attachment(reference, path=path)" in source
    assert "resolved = resolver.resolve(reference)" in source
    assert "resolved.terminal_attachment" in source
    assert "resolved.terminal_path" in source


def test_deform_contract_has_no_second_skin_or_parent_resolver():
    source = read(DEFORM)

    for forbidden in (
        "def _build_skin_index(",
        "def _resolve_attachment(",
        "skin_by_name:",
        "ambiguous_skin_names:",
        "resolving:",
        'raw_parent_skin = attachment.get("skin")',
        'parent_name = _require_name(parent',
        'parent_skin_name = "default"',
        "linked mesh parent cycle",
    ):
        assert forbidden not in source


def test_deform_capacity_remains_owned_by_deform_contract():
    linked_source = read(LINKED)
    deform_source = read(DEFORM)

    assert "decode_weighted_vertices" not in linked_source
    assert "_deform_capacity_from_vertices" not in linked_source
    assert "decode_weighted_vertices" in deform_source
    assert "_deform_capacity_from_vertices" in deform_source
    assert "_deform_capacity_for_attachment" in deform_source
    assert "expected_vertex_count=expected_coordinate_count // 2" in deform_source
