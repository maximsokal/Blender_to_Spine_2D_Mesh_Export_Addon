from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "setup_slot_contract.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_setup_slot_index_is_frozen_and_retains_exact_tuple():
    source = read(CONTRACT)

    assert "@dataclass(frozen=True, slots=True)" in source
    assert "slot_names: tuple[str, ...]" in source
    assert "self.slot_names" in source
    assert "tuple(self.slot_names)" not in source
    assert "list(self.slot_names)" not in source


def test_setup_slot_index_owns_strict_names_and_ambiguity():
    source = read(CONTRACT)

    assert 'raise TypeError("slot_names must be tuple")' in source
    assert 'f"slot_names[{slot_index}]"' in source
    assert "ambiguous_names.add(slot_name)" in source
    assert "frozenset(ambiguous_names)" in source
    assert "references undefined slot" in source
    assert "references duplicated setup slot" in source


def test_internal_lookup_mapping_is_read_only():
    source = read(CONTRACT)

    assert "MappingProxyType(index_by_name)" in source
    assert "_index_by_name: Mapping[str, int]" in source
    assert "_ambiguous_names: frozenset[str]" in source


def test_reuse_requires_exact_slot_tuple_identity():
    source = read(CONTRACT)

    assert "def resolve_setup_slot_index(" in source
    assert "setup_slot_index is None" in source
    assert "isinstance(setup_slot_index, SetupSlotIndex)" in source
    assert "setup_slot_index.slot_names is not slot_names" in source
    assert "built from the exact slot_names tuple" in source
