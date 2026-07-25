from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "linked_mesh_contract.py"
)
SERIALIZER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "serializer.py"
)
DEFORM = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "deform_timeline_contract.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_linked_mesh_contract_owns_exact_setup_types_and_default_skin():
    source = read(CONTRACT)

    assert '_LINKED_MESH_TYPES = frozenset({"linkedmesh"})' in source
    assert '_MESH_PARENT_TYPES = frozenset({"mesh", "linkedmesh"})' in source
    assert '_DEFAULT_SKIN_NAME = "default"' in source
    assert 'if raw_skin_name in (None, ""):' in source
    assert "parent_skin_name = _DEFAULT_SKIN_NAME" in source


def test_linked_mesh_contract_aliases_shared_name_requirement():
    source = read(CONTRACT)

    assert (
        "from .spine_scalar_contract import require_name as _require_name"
        in source
    )
    assert "def _require_name(" not in source
    assert "_require_name(" in source


def test_parent_reference_is_forced_to_the_source_slot():
    source = read(CONTRACT)

    assert "slot_name=record.reference.slot_name" in source
    assert 'attachment.get("slot")' not in source
    assert 'attachment["slot"]' not in source


def test_linked_mesh_contract_has_recursive_cache_and_cycle_detection():
    source = read(CONTRACT)

    assert "self._cache: dict[AttachmentReference, ResolvedLinkedMesh]" in source
    assert "cached = self._cache.get(reference)" in source
    assert "if reference in stack:" in source
    assert "stack=stack + (reference,)" in source
    assert "linked mesh parent cycle" in source
    assert "terminal_path=terminal.terminal_path" in source


def test_terminal_parent_must_be_mesh_compatible():
    source = read(CONTRACT)

    assert "if parent_type not in _MESH_PARENT_TYPES:" in source
    assert "resolves to unsupported attachment " in source
    assert "resolves to non-mesh attachment type" in source


def test_resolver_validates_every_setup_linked_mesh_not_only_animated_ones():
    source = read(CONTRACT)

    assert "def validate_all(" in source
    assert "for record in self._records.values():" in source
    assert "if not is_linked_mesh_attachment" in source
    assert "resolved.append(self.resolve(record.reference))" in source
    assert "animations" not in source


def test_public_lookup_is_shared_with_deform_contract():
    linked_source = read(CONTRACT)
    deform_source = read(DEFORM)

    assert "def require_skin(" in linked_source
    assert "def get_attachment(" in linked_source
    assert "SetupAttachment" in linked_source
    assert "resolver.require_skin(" in deform_source
    assert "resolver.get_attachment(" in deform_source
    assert "resolver.resolve(reference)" in deform_source


def test_contract_never_normalizes_or_rewrites_attachments():
    source = read(CONTRACT)

    for forbidden in (
        ".sort(",
        "sorted(",
        "setdefault(",
        'attachment["parent"] =',
        'attachment["skin"] =',
        'attachment["type"] =',
        'attachment["timelines"] =',
        'attachment["name"] =',
        'attachment["path"] =',
        'attachment["color"] =',
        ".lower()",
        ".upper()",
    ):
        assert forbidden not in source


def test_serializer_runs_setup_linked_mesh_contract_before_animation_contracts():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    validator_index = to_dict_source.index(
        "self._validator.validate_or_raise(document)"
    )
    linked_index = to_dict_source.index("validate_setup_linked_meshes(")
    color_index = to_dict_source.index(
        "validate_animation_slot_color_timelines("
    )
    curve_index = to_dict_source.index("validate_animation_curves(")
    deform_index = to_dict_source.index(
        "validate_animation_deform_timelines("
    )
    sequence_index = to_dict_source.index(
        "validate_animation_sequence_timelines("
    )
    data_index = to_dict_source.index('data: dict[str, Any] = {')

    assert (
        validator_index
        < linked_index
        < color_index
        < curve_index
        < deform_index
        < sequence_index
        < data_index
    )
    assert 'path="document.skins"' in to_dict_source


def test_deform_capacity_logic_remains_animation_specific():
    linked_source = read(CONTRACT)
    deform_source = read(DEFORM)

    assert "deform" not in linked_source
    assert "_resolve_deform_capacity" in deform_source
    assert "decode_weighted_vertices" in deform_source
    assert "decode_weighted_vertices" not in linked_source


def test_serializer_keeps_skin_attachment_mappings_as_supplied():
    source = read(SERIALIZER)
    attachment_start = source.index("def attachment_to_dict(")
    skin_start = source.index("def skin_to_dict(", attachment_start)
    attachment_source = source[attachment_start:skin_start]

    assert "if isinstance(attachment, Mapping):" in attachment_source
    assert "return dict(attachment)" in attachment_source
    assert 'attachment["parent"] =' not in attachment_source
    assert 'attachment["skin"] =' not in attachment_source
